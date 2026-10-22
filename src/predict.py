"""Run a detector over an annotation file's images and write COCO-format predictions.

    # zero-shot COCO model (person = class 0 of 80)
    uv run python -m src.predict --gt data/annotations/test.json --modality rgb \
        --checkpoint PekingU/rtdetr_v2_r18vd --coco80 --out runs/zeroshot/rgb/pred_test.json
    # fine-tuned run
    uv run python -m src.predict --gt data/annotations/test.json --modality thermal \
        --checkpoint runs/thermal_s1/best --out runs/thermal_s1/pred_test.json
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import RTDetrV2ForObjectDetection

from src.data.llvip import LLVIP
from src.models.rtdetr import COCO_PERSON, MODALITIES, build_model, postprocess, read_input, to_tensor


class InferenceSet(Dataset):
    def __init__(self, images: list[dict], modality: str, root: Path = LLVIP):
        self.images, self.modality, self.root = images, modality, root

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        im = self.images[i]
        return to_tensor(read_input(im["file_name"], self.modality, self.root)), im["id"]


def load_model(checkpoint: str, modality: str, coco80: bool) -> RTDetrV2ForObjectDetection:
    if coco80:
        return RTDetrV2ForObjectDetection.from_pretrained(checkpoint)
    if Path(checkpoint).exists():
        return RTDetrV2ForObjectDetection.from_pretrained(checkpoint)
    return build_model(checkpoint, modality)  # pretrained, converted to 1 class, no fine-tuning


@torch.no_grad()
def predict(model, images: list[dict], modality: str, class_index: int = 0, batch_size: int = 8,
            workers: int = 4, device: str = "cuda", amp: bool = True, root: Path = LLVIP) -> list[dict]:
    model.eval().to(device)
    loader = DataLoader(InferenceSet(images, modality, root), batch_size=batch_size,
                        num_workers=workers, pin_memory=device == "cuda")
    sizes = {im["id"]: (im["width"], im["height"]) for im in images}
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    preds = []
    for x, ids in tqdm(loader, desc=f"predict {modality}", mininterval=30):
        with torch.autocast(device, dtype=dtype, enabled=amp and device == "cuda"):
            out = model(pixel_values=x.to(device, non_blocking=True))
        # postprocess in normalised coords, then scale per image (sizes may differ in a batch)
        per_image = postprocess(out.logits.float(), out.pred_boxes.float(), (1, 1), class_index)
        for image_id, dets in zip(ids.tolist(), per_image):
            w, h = sizes[image_id]
            for d in dets:
                x0, y0, bw, bh = d["bbox"]
                preds.append({"image_id": image_id, "category_id": 1, "score": round(d["score"], 5),
                              "bbox": [round(x0 * w, 2), round(y0 * h, 2), round(bw * w, 2), round(bh * h, 2)]})
    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--modality", choices=MODALITIES, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--coco80", action="store_true", help="checkpoint is the 80-class COCO model")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--root", type=Path, default=LLVIP)
    args = parser.parse_args()

    images = json.loads(args.gt.read_text())["images"]
    model = load_model(args.checkpoint, args.modality, args.coco80)
    preds = predict(model, images, args.modality, COCO_PERSON if args.coco80 else 0,
                    args.batch_size, args.workers, root=args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(preds))
    print(f"{len(preds)} detections for {len(images)} images -> {args.out}")


if __name__ == "__main__":
    main()
