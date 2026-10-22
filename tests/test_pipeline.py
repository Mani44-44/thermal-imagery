"""Round trip: GT boxes -> training targets -> (as if predicted) post-processing -> metrics = 1.0.
Box-format bugs (xywh vs cxcywh, normalisation, w/h order) would silently ruin every run."""

import numpy as np
import torch

from src.eval.cocoish import average_precision, match_image
from src.models.rtdetr import postprocess, read_input, to_tensor
from src.train import LLVIPDetection, augment
from tests.conftest import needs_data

pytestmark = needs_data


def test_targets_roundtrip_to_perfect_map(cocos):
    coco = cocos["golden_ci"]
    ds = LLVIPDetection(coco, "rgb", root=__import__("src.data.llvip", fromlist=["LLVIP"]).LLVIP, train=False)
    matches = []
    for i in range(0, len(ds), 10):
        x, target, image_id = ds[i]
        assert x.shape == (3, 640, 640) and 0 <= x.min() and x.max() <= 1
        n = len(target["boxes"])
        logits = torch.full((1, max(n, 1), 1), -20.0)
        logits[0, :n, 0] = 20.0  # confident "predictions" exactly at the targets
        boxes = torch.zeros(1, max(n, 1), 4)
        boxes[0, :n] = target["boxes"]
        dets = postprocess(logits, boxes, (1280, 1024))[0]
        db = np.array([d["bbox"] for d in dets]).reshape(-1, 4)
        sc = np.array([d["score"] for d in dets])
        gt = np.array(ds.boxes[image_id], float).reshape(-1, 4)
        matches.append(match_image(gt, np.zeros(len(gt), bool), db, sc))
    np.testing.assert_allclose(average_precision(matches), 1.0, atol=1e-6)


def test_augment_keeps_channels_aligned(cocos):
    im = cocos["golden_ci"]["images"][0]
    img = read_input(im["file_name"], "fused4")
    boxes = np.array([a["bbox"] for a in cocos["golden_ci"]["annotations"] if a["image_id"] == im["id"]], np.float32)
    for seed in range(5):
        import random
        random.seed(seed)
        out, b = augment(img.copy(), boxes.copy(), "fused4")
        assert out.shape[2] == 4 and len(b) >= 1
        assert (b[:, 2] > 0).all() and (b[:, 3] > 0).all()
        assert (b[:, 0] >= 0).all() and (b[:, 0] + b[:, 2] <= out.shape[1] + 1e-3).all()


def test_input_shapes(cocos):
    im = cocos["golden_ci"]["images"][0]
    assert to_tensor(read_input(im["file_name"], "rgb")).shape == (3, 640, 640)
    assert to_tensor(read_input(im["file_name"], "thermal")).shape == (3, 640, 640)
    assert to_tensor(read_input(im["file_name"], "fused4")).shape == (4, 640, 640)
