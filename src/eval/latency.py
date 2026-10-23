"""End-to-end single-image latency on the local GPU.

Times preprocessing (resize + tensor + host-to-device), the forward pass(es), post-processing and,
for late fusion, box fusion, at batch size 1 after a warmup. Images are decoded beforehand (disk
I/O excluded) and cycled. Reports p50/p95/mean in ms plus device and precision.

    uv run python -m src.eval.latency --run runs/rgb_s1/best --modality rgb
    uv run python -m src.eval.latency --late runs/late_s1/run.json   # both models + fusion, sequential
"""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from src.models.rtdetr import MODALITIES, postprocess, read_input, to_tensor
from src.predict import load_model

GOLDEN = Path(__file__).resolve().parents[2] / "data" / "annotations" / "golden.json"


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def measure(stages: list[tuple], fuse=None, n_images: int = 32, warmup: int = 50, iters: int = 500,
            device: str = "cuda", amp: str = "bf16") -> dict:
    """stages: list of (model, modality). Returns latency stats in ms."""
    images = json.loads(GOLDEN.read_text())["images"][:n_images]
    raw = [{m: read_input(im["file_name"], m) for _, m in stages} for im in images]
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(amp)
    for model, _ in stages:
        model.eval().to(device)
    times = []
    for i in range(warmup + iters):
        sample = raw[i % len(raw)]
        _sync()
        t0 = time.perf_counter()
        outs = []
        for model, modality in stages:
            x = to_tensor(sample[modality])[None].to(device)
            with torch.autocast(device, dtype=dtype or torch.float32, enabled=dtype is not None):
                out = model(pixel_values=x)
            outs.append(postprocess(out.logits.float(), out.pred_boxes.float(), (1280, 1024))[0])
        if fuse:
            fuse(outs)
        _sync()
        if i >= warmup:
            times.append((time.perf_counter() - t0) * 1000)
    t = np.array(times)
    return {"p50_ms": round(float(np.percentile(t, 50)), 2), "p95_ms": round(float(np.percentile(t, 95)), 2),
            "mean_ms": round(float(t.mean()), 2), "iters": iters, "batch": 1, "amp": amp or "fp32",
            "device": torch.cuda.get_device_name(0) if device == "cuda" else platform.processor(),
            "host": platform.node(), "torch": torch.__version__}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", help="model dir or HF checkpoint")
    parser.add_argument("--modality", choices=MODALITIES)
    parser.add_argument("--late", type=Path, help="late-fusion run.json: time RGB + thermal passes + fusion")
    parser.add_argument("--amp", choices=["bf16", "fp16", "fp32"], default="fp32",
                        help="fp32 is fastest at batch 1 here: the HF model is launch-bound, autocast adds casts")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    amp = None if args.amp == "fp32" else args.amp
    if args.late:
        from src.fusion.late import fuse_image
        cfg = json.loads(args.late.read_text())
        root = Path(__file__).resolve().parents[2]
        stages = [(load_model(str(root / cfg[k] / "best"), m, coco80=False), m)
                  for k, m in (("rgb_run", "rgb"), ("thermal_run", "thermal"))]
        fusion = cfg["fusion"]
        res = measure(stages, fuse=lambda outs: fuse_image(outs[0], outs[1], **fusion), amp=amp)
        res["mode"] = "sequential, both models + fusion"
        # One GPU: "parallel" is the lower bound max(single model) + fusion; fusion is timed alone on CPU.
        rgb_out, th_out = [[{"bbox": [100.0 * i, 100.0, 50.0, 120.0], "score": 0.9 - 0.008 * i} for i in range(100)]] * 2
        t = []
        for _ in range(200):
            t0 = time.perf_counter()
            fuse_image(rgb_out, th_out, **fusion)
            t.append((time.perf_counter() - t0) * 1000)
        res["fusion_only_p50_ms"] = round(float(np.percentile(t, 50)), 3)
    else:
        model = load_model(args.run, args.modality, coco80=False)
        res = measure([(model, args.modality)], amp=amp)
    print(json.dumps(res))
    if args.out:
        args.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
