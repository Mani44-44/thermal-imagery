"""RT-DETRv2 wrapper: model construction, input building and post-processing.

Modalities (all scaled to [0, 1], resized to 640x640 like the HF image processor, no mean/std):
- rgb: visible image, 3 channels
- thermal: infrared image (grayscale) replicated to 3 channels, so pretrained weights apply
- fused4: visible RGB + infrared as a 4th channel (early fusion)

Single-class (person) models are initialised from the COCO checkpoint's *person* row of every
classification head instead of a random re-init, identically for all modalities.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from transformers import RTDetrV2ForObjectDetection

from src.data.llvip import LLVIP

MODALITIES = ("rgb", "thermal", "fused4")
COCO_PERSON = 0
INPUT_SIZE = 640


def in_channels(modality: str) -> int:
    return 4 if modality == "fused4" else 3


def _slice_rows(module: nn.Module, rows: list[int]):
    """Keep only the given output rows of a Linear/Embedding layer, in place."""
    w = module.weight.data[rows].clone()
    if isinstance(module, nn.Embedding):
        module.num_embeddings = len(rows)
        if module.padding_idx is not None:
            module.padding_idx = len(rows) - 1
    else:
        module.out_features = len(rows)
        module.bias = nn.Parameter(module.bias.data[rows].clone())
    module.weight = nn.Parameter(w)


def to_single_class(model: RTDetrV2ForObjectDetection) -> RTDetrV2ForObjectDetection:
    """Convert an 80-class COCO RT-DETR to person-only, keeping the person head weights."""
    n = model.config.num_labels
    seen = set()
    for name, mod in model.named_modules():
        if id(mod) in seen:
            continue
        if isinstance(mod, nn.Linear) and mod.out_features == n and (
                "class_embed" in name or "enc_score_head" in name):
            _slice_rows(mod, [COCO_PERSON])
            seen.add(id(mod))
        elif isinstance(mod, nn.Embedding) and "denoising_class_embed" in name:
            _slice_rows(mod, [COCO_PERSON, n])  # person + the padding/"no class" row
            seen.add(id(mod))
    model.config.num_labels = 1
    model.config.id2label = {0: "person"}
    model.config.label2id = {"person": 0}
    return model


def expand_first_conv(model: RTDetrV2ForObjectDetection, channels: int = 4):
    """Make the backbone's first conv take `channels` inputs. Extra channels are initialised with
    the mean of the pretrained RGB filters."""
    conv = model.model.backbone.model.embedder.embedder[0].convolution
    w = conv.weight.data
    extra = w.mean(dim=1, keepdim=True).repeat(1, channels - w.shape[1], 1, 1)
    new = nn.Conv2d(channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                    bias=conv.bias is not None)
    new.weight.data = torch.cat([w, extra], dim=1)
    if conv.bias is not None:
        new.bias.data = conv.bias.data.clone()
    embedder = model.model.backbone.model.embedder
    embedder.embedder[0].convolution = new
    embedder.num_channels = channels  # the embedder checks input channels against this
    model.config.backbone_config.num_channels = channels
    return model


def num_channels(model: RTDetrV2ForObjectDetection) -> int:
    return model.config.backbone_config.num_channels


def build_model(checkpoint: str = "PekingU/rtdetr_v2_r18vd", modality: str = "rgb",
                single_class: bool = True) -> RTDetrV2ForObjectDetection:
    model = RTDetrV2ForObjectDetection.from_pretrained(checkpoint)
    if single_class and model.config.num_labels != 1:
        model = to_single_class(model)
    if in_channels(modality) == 4 and num_channels(model) != 4:
        model = expand_first_conv(model, 4)
    return model


def load_trained(run_dir: Path, modality: str) -> RTDetrV2ForObjectDetection:
    """Load a fine-tuned model saved with save_pretrained (config already single-class/4-ch)."""
    return RTDetrV2ForObjectDetection.from_pretrained(run_dir)


def read_input(file_name: str, modality: str, root: Path = LLVIP) -> np.ndarray:
    """HxWxC uint8 array at native resolution for one image (file_name like 'test/190001.jpg')."""
    if modality in ("rgb", "fused4"):
        vis = cv2.cvtColor(cv2.imread(str(root / "visible" / file_name)), cv2.COLOR_BGR2RGB)
    if modality in ("thermal", "fused4"):
        ir = cv2.imread(str(root / "infrared" / file_name), cv2.IMREAD_GRAYSCALE)
    if modality == "rgb":
        return vis
    if modality == "thermal":
        return np.repeat(ir[..., None], 3, axis=2)
    return np.concatenate([vis, ir[..., None]], axis=2)


def to_tensor(img: np.ndarray, size: int = INPUT_SIZE) -> torch.Tensor:
    """HxWxC uint8 -> CxSxS float in [0, 1] (bilinear resize, like the HF processor)."""
    resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    if resized.ndim == 2:
        resized = resized[..., None]
    return torch.from_numpy(resized).permute(2, 0, 1).float().div_(255)


@torch.no_grad()
def postprocess(logits: torch.Tensor, pred_boxes: torch.Tensor, orig_wh: tuple[int, int],
                class_index: int = 0, top_k: int = 100) -> list[list[dict]]:
    """Batch outputs -> per image list of {"bbox": xywh in original pixels, "score"}.
    Uses sigmoid scores (RT-DETR is trained with focal/VFL loss) for one class."""
    w, h = orig_wh
    scores = logits[..., class_index].sigmoid()                     # B x Q
    k = min(top_k, scores.shape[1])
    top, idx = scores.topk(k, dim=1)
    boxes = torch.gather(pred_boxes, 1, idx[..., None].expand(-1, -1, 4))  # cx, cy, bw, bh (0-1)
    out = []
    for s, b in zip(top.float().cpu().numpy(), boxes.float().cpu().numpy()):
        x = (b[:, 0] - b[:, 2] / 2) * w
        y = (b[:, 1] - b[:, 3] / 2) * h
        out.append([{"bbox": [float(xi), float(yi), float(bw * w), float(bh * h)], "score": float(si)}
                    for xi, yi, bw, bh, si in zip(x, y, b[:, 2], b[:, 3], s)])
    return out
