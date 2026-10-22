"""Fine-tune RT-DETRv2 on LLVIP for one modality (rgb / thermal / fused4).

Single GPU:   uv run python -m src.train --modality rgb --out runs/rgb_s1
Two GPUs:     torchrun --nproc_per_node=2 -m src.train --modality rgb --out runs/rgb_s1 (Kaggle)

Fairness protocol (same for every arm): same checkpoint, input size, epochs, LR schedule and
*effective* batch size (per-GPU batch x GPUs x accumulation), head initialised from COCO person.
Model selection on val mAP@0.5:0.95 of the EMA weights; the test set is never touched here.

Outputs in --out: best/ (save_pretrained of the best EMA model), last.pt (resume state),
log.jsonl (per-epoch metrics), run.json (config, versions, hardware).
"""

import argparse
import json
import math
import os
import platform
import random
import subprocess
import time
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from src.data.llvip import LLVIP, ROOT
from src.eval.cocoish import average_precision, match_image
from src.models.rtdetr import INPUT_SIZE, MODALITIES, build_model, postprocess, read_input, to_tensor

ANN = ROOT / "data" / "annotations"


# ----------------------------------------------------------------------------- data

class LLVIPDetection(Dataset):
    def __init__(self, coco: dict, modality: str, root: Path, train: bool, limit: int | None = None):
        self.images = sorted(coco["images"], key=lambda i: i["id"])[:limit]
        self.boxes = {i["id"]: [] for i in self.images}
        for a in coco["annotations"]:
            if a["image_id"] in self.boxes:
                self.boxes[a["image_id"]].append(a["bbox"])
        self.modality, self.root, self.train = modality, root, train

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        im = self.images[i]
        img = read_input(im["file_name"], self.modality, self.root, replicate=False)  # HxWxC uint8
        boxes = np.array(self.boxes[im["id"]], np.float32).reshape(-1, 4)  # xywh px
        if self.train:
            img, boxes = augment(img, boxes, self.modality)
        h, w = img.shape[:2]  # boxes are normalised by the (possibly cropped) native size
        if self.train:
            img = photometric(cv2.resize(img, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR),
                              self.modality)
        x = to_tensor(img)
        cxcywh = np.stack([(boxes[:, 0] + boxes[:, 2] / 2) / w, (boxes[:, 1] + boxes[:, 3] / 2) / h,
                           boxes[:, 2] / w, boxes[:, 3] / h], 1) if len(boxes) else np.zeros((0, 4), np.float32)
        target = {"class_labels": torch.zeros(len(boxes), dtype=torch.long),
                  "boxes": torch.from_numpy(np.clip(cxcywh, 0, 1)).float()}
        return x, target, im["id"]


def augment(img: np.ndarray, boxes: np.ndarray, modality: str):
    """Geometric ops (flip, crop) at native resolution; they hit all channels identically."""
    h, w = img.shape[:2]
    if random.random() < 0.5:  # horizontal flip
        img = img[:, ::-1]
        boxes = boxes.copy()
        boxes[:, 0] = w - boxes[:, 0] - boxes[:, 2]
    if random.random() < 0.5 and len(boxes):  # random crop keeping >= 1 box centre
        for _ in range(10):
            s = random.uniform(0.5, 1.0)
            cw, ch = int(w * s), int(h * s)
            x0, y0 = random.randint(0, w - cw), random.randint(0, h - ch)
            cx, cy = boxes[:, 0] + boxes[:, 2] / 2, boxes[:, 1] + boxes[:, 3] / 2
            keep = (cx > x0) & (cx < x0 + cw) & (cy > y0) & (cy < y0 + ch)
            if keep.any():
                b = boxes[keep].copy()
                x1 = np.clip(b[:, 0] - x0, 0, cw); y1 = np.clip(b[:, 1] - y0, 0, ch)
                x2 = np.clip(b[:, 0] + b[:, 2] - x0, 0, cw); y2 = np.clip(b[:, 1] + b[:, 3] - y0, 0, ch)
                b = np.stack([x1, y1, x2 - x1, y2 - y1], 1)
                ok = (b[:, 2] > 2) & (b[:, 3] > 2)
                if ok.any():
                    img, boxes = img[y0:y0 + ch, x0:x0 + cw], b[ok]
                    break
    img = np.ascontiguousarray(img)
    return img, boxes


def photometric(img: np.ndarray, modality: str) -> np.ndarray:
    """Colour jitter on the (already resized) image, via uint8 lookup tables. RGB channels get
    hue/saturation/value jitter; the thermal channel only a mild brightness/contrast change (hue
    and saturation mean nothing there)."""
    if modality in ("rgb", "fused4") and random.random() < 0.8:
        hue, sat, val = cv2.split(cv2.cvtColor(np.ascontiguousarray(img[..., :3]), cv2.COLOR_RGB2HSV))
        x = np.arange(256, dtype=np.float32)
        lut_h = ((x + random.uniform(-5, 5)) % 180).astype(np.uint8)
        lut_s = np.clip(x * random.uniform(0.7, 1.3), 0, 255).astype(np.uint8)
        lut_v = np.clip(x * random.uniform(0.7, 1.3), 0, 255).astype(np.uint8)
        hsv = cv2.merge([cv2.LUT(hue, lut_h), cv2.LUT(sat, lut_s), cv2.LUT(val, lut_v)])
        img[..., :3] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    ir_ch = {"thermal": slice(0, 1), "fused4": slice(3, 4)}.get(modality)
    if ir_ch is not None and random.random() < 0.5:
        lut = np.clip(np.arange(256) * random.uniform(0.9, 1.1) + random.uniform(-10, 10), 0, 255).astype(np.uint8)
        img[..., ir_ch] = cv2.LUT(np.ascontiguousarray(img[..., ir_ch]), lut).reshape(img[..., ir_ch].shape)
    return img


def collate(batch):
    xs, targets, ids = zip(*batch)
    return torch.stack(xs), list(targets), list(ids)


# ----------------------------------------------------------------------------- training utils

class EMA:
    """Exponential moving average of weights with warmup (decay ramps up over ~2k updates)."""

    def __init__(self, model, decay=0.9999, tau=2000):
        self.model = deepcopy(model).eval()
        self.decay, self.tau, self.updates = decay, tau, 0
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / self.tau))
        msd = model.state_dict()
        for k, v in self.model.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)


@torch.no_grad()
def evaluate_map(model, coco: dict, modality: str, root: Path, device, batch_size=16, workers=4):
    """Val mAP50 / mAP50-95 with the project's matcher (single process)."""
    model.eval()
    ds = LLVIPDetection(coco, modality, root, train=False)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=workers, collate_fn=collate)
    gts = {i: np.array(b, float).reshape(-1, 4) for i, b in ds.boxes.items()}
    sizes = {im["id"]: (im["width"], im["height"]) for im in ds.images}
    matches = []
    for x, _, ids in loader:
        out = model(pixel_values=x.to(device))
        for image_id, dets in zip(ids, postprocess(out.logits.float(), out.pred_boxes.float(), (1, 1))):
            w, h = sizes[image_id]
            db = np.array([[d["bbox"][0] * w, d["bbox"][1] * h, d["bbox"][2] * w, d["bbox"][3] * h] for d in dets]).reshape(-1, 4)
            sc = np.array([d["score"] for d in dets])
            matches.append(match_image(gts[image_id], np.zeros(len(gts[image_id]), bool), db, sc))
    ap = average_precision(matches)
    return float(ap[0]), float(np.mean(ap))


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return os.environ.get("GIT_COMMIT", "unknown")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modality", choices=MODALITIES, required=True)
    p.add_argument("--checkpoint", default="PekingU/rtdetr_v2_r18vd")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--root", type=Path, default=LLVIP, help="LLVIP folder (visible/, infrared/)")
    p.add_argument("--ann-dir", type=Path, default=ANN)
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--effective-batch", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=8, help="per GPU")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-backbone", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--clip-grad", type=float, default=0.1)
    p.add_argument("--amp", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--limit", type=int, help="use only the first N train images (overfit tests)")
    p.add_argument("--val-split", default="val", help="annotation file to select on (val / train)")
    p.add_argument("--time-limit-hours", type=float, default=0, help="stop cleanly after this; resume later")
    p.add_argument("--name", default=None)
    args = p.parse_args()

    distributed = "LOCAL_RANK" in os.environ
    if distributed:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    else:
        rank, world = 0, 1
    device = torch.device("cuda", torch.cuda.current_device())
    main_proc = rank == 0
    accum = max(1, args.effective_batch // (args.batch_size * world))
    assert args.batch_size * world * accum == args.effective_batch, "effective batch must divide evenly"
    set_seed(args.seed + rank)

    amp = args.amp
    if amp == "auto":
        amp = "bf16" if torch.cuda.is_bf16_supported(including_emulation=False) else "fp16"
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(amp)
    scaler = torch.amp.GradScaler("cuda", enabled=amp == "fp16")

    train_coco = json.loads((args.ann_dir / "train.json").read_text())
    val_coco = json.loads((args.ann_dir / f"{args.val_split}.json").read_text())
    if args.limit and args.val_split == "train":  # overfit test: select on the same images
        keep = {i["id"] for i in sorted(train_coco["images"], key=lambda i: i["id"])[:args.limit]}
        val_coco = {**train_coco, "images": [i for i in train_coco["images"] if i["id"] in keep]}
    train_ds = LLVIPDetection(train_coco, args.modality, args.root, train=True, limit=args.limit)
    sampler = DistributedSampler(train_ds, world, rank, shuffle=True, seed=args.seed) if distributed else None
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=sampler is None, sampler=sampler,
                        num_workers=args.workers, collate_fn=collate, drop_last=True,
                        pin_memory=True, persistent_workers=args.workers > 0)

    model = build_model(args.checkpoint, args.modality).to(device)
    backbone = [p_ for n, p_ in model.named_parameters() if "backbone" in n]
    others = [p_ for n, p_ in model.named_parameters() if "backbone" not in n]
    opt = torch.optim.AdamW([{"params": backbone, "lr": args.lr_backbone},
                             {"params": others, "lr": args.lr}], weight_decay=args.weight_decay)
    steps_per_epoch = len(loader) // accum
    total_steps = steps_per_epoch * args.epochs

    def lr_factor(step):
        if step < args.warmup_steps:
            return (step + 1) / args.warmup_steps
        t = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return args.min_lr_ratio + (1 - args.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * min(t, 1)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    ema = EMA(model)
    start_epoch, best, step = 0, -1.0, 0

    args.out.mkdir(parents=True, exist_ok=True)
    last = args.out / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        ema.model.load_state_dict(ck["ema"])
        ema.updates = ck["ema_updates"]
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best, step = ck["epoch"] + 1, ck["best"], ck["step"]
        if main_proc:
            print(f"resumed from epoch {ck['epoch']} (best val mAP50-95 {best:.4f})")

    ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index],
                                                          find_unused_parameters=True) if distributed else model
    if main_proc:
        meta = {"name": args.name or args.out.name, **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "world_size": world, "accum": accum, "amp_resolved": amp, "git_commit": git_commit(),
                "torch": torch.__version__, "transformers": __import__("transformers").__version__,
                "device": torch.cuda.get_device_name(device), "host": platform.node(),
                "train_images": len(train_ds), "steps_per_epoch": steps_per_epoch}
        (args.out / "run.json").write_text(json.dumps(meta, indent=1))
        print(json.dumps(meta))

    t_start = time.time()
    for epoch in range(start_epoch, args.epochs):
        if sampler:
            sampler.set_epoch(epoch)
        ddp_model.train()
        t0, losses = time.time(), []
        for i, (x, targets, _) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            sync = (i + 1) % accum == 0
            ctx = ddp_model.no_sync() if distributed and not sync else torch.enable_grad()
            with ctx:
                with torch.autocast("cuda", dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
                    loss = ddp_model(pixel_values=x, labels=targets).loss
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss {loss.item()} at epoch {epoch} iter {i}")
                scaler.scale(loss / accum).backward()
            losses.append(loss.item())
            if sync:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                ema.update(model)
                step += 1
            if main_proc and i % 200 == 0:
                print(f"epoch {epoch} iter {i}/{len(loader)} loss {np.mean(losses[-200:]):.4f} "
                      f"lr {sched.get_last_lr()[1]:.2e} {(time.time() - t0) / (i + 1):.3f}s/it", flush=True)
        train_time = time.time() - t0

        if main_proc:
            m50, m5095 = evaluate_map(ema.model, val_coco, args.modality, args.root, device, workers=args.workers)
            improved = m5095 > best
            if improved:
                best = m5095
                ema.model.save_pretrained(args.out / "best")
            rec = {"epoch": epoch, "step": step, "loss": float(np.mean(losses)), "val_mAP50": m50,
                   "val_mAP50_95": m5095, "best": best, "epoch_min": train_time / 60,
                   "img_per_s": len(loader) * args.batch_size * world / train_time}
            with (args.out / "log.jsonl").open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print(json.dumps(rec), flush=True)
            torch.save({"model": model.state_dict(), "ema": ema.model.state_dict(), "ema_updates": ema.updates,
                        "opt": opt.state_dict(), "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                        "epoch": epoch, "best": best, "step": step}, last)
        if distributed:
            dist.barrier()
        if args.time_limit_hours and (time.time() - t_start) / 3600 > args.time_limit_hours \
                and epoch + 1 < args.epochs:
            if main_proc:
                print(f"time limit reached after epoch {epoch}; resume to continue", flush=True)
            break
    else:
        if main_proc:
            (args.out / "DONE").write_text(json.dumps({"best_val_mAP50_95": best}))
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
