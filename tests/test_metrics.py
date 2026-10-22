"""The numpy matcher must agree with pycocotools, and behave sanely on edge cases."""

import contextlib
import io

import numpy as np
import pytest
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from src.eval.cocoish import average_precision, match_image, precision_recall_at


def synthetic(seed=0, n_images=40):
    rng = np.random.default_rng(seed)
    images, anns, preds = [], [], []
    for i in range(1, n_images + 1):
        images.append({"id": i, "width": 1280, "height": 1024})
        for _ in range(rng.integers(0, 6)):
            x, y = rng.uniform(0, 1000), rng.uniform(0, 700)
            w, h = rng.uniform(20, 150), rng.uniform(60, 300)
            anns.append({"id": len(anns) + 1, "image_id": i, "category_id": 1,
                         "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0})
            if rng.random() < 0.8:  # a jittered true detection
                j = rng.normal(0, 8, 4)
                preds.append({"image_id": i, "category_id": 1, "score": float(rng.uniform(0.3, 1)),
                              "bbox": [x + j[0], y + j[1], max(w + j[2], 5), max(h + j[3], 5)]})
        for _ in range(rng.integers(0, 4)):  # false positives
            preds.append({"image_id": i, "category_id": 1, "score": float(rng.uniform(0, 0.8)),
                          "bbox": [rng.uniform(0, 1100), rng.uniform(0, 800), rng.uniform(20, 150),
                                   rng.uniform(60, 300)]})
    return {"images": images, "annotations": anns, "categories": [{"id": 1, "name": "person"}]}, preds


def pycoco_ap(gt, preds, area_rng=None):
    with contextlib.redirect_stdout(io.StringIO()):
        cg = COCO()
        cg.dataset = gt
        cg.createIndex()
        cd = cg.loadRes(preds)
        ev = COCOeval(cg, cd, "bbox")
        if area_rng:
            ev.params.areaRng = [area_rng]
            ev.params.areaRngLbl = ["custom"]
        ev.params.maxDets = [100]
        ev.evaluate()
        ev.accumulate()
    p = ev.eval["precision"][:, :, 0, 0, 0]  # T x R
    return np.array([row[row > -1].mean() if (row > -1).any() else np.nan for row in p])


def ours(gt, preds, area_rng=None):
    ms = []
    for im in gt["images"]:
        g = np.array([a["bbox"] for a in gt["annotations"] if a["image_id"] == im["id"]]).reshape(-1, 4)
        d = [p for p in preds if p["image_id"] == im["id"]]
        db = np.array([p["bbox"] for p in d]).reshape(-1, 4)
        ignore, oor = np.zeros(len(g), bool), None
        if area_rng:
            ga, da = g[:, 2] * g[:, 3], db[:, 2] * db[:, 3]
            ignore = (ga < area_rng[0]) | (ga > area_rng[1])
            oor = (da < area_rng[0]) | (da > area_rng[1])
        ms.append(match_image(g, ignore, db, np.array([p["score"] for p in d]), det_out_of_range=oor))
    return average_precision(ms)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_pycocotools(seed):
    gt, preds = synthetic(seed)
    np.testing.assert_allclose(ours(gt, preds), pycoco_ap(gt, preds), atol=1e-6)


@pytest.mark.parametrize("seed", [0, 1])
def test_matches_pycocotools_area_range(seed):
    gt, preds = synthetic(seed)
    rng = [3000, 15000]
    np.testing.assert_allclose(ours(gt, preds, rng), pycoco_ap(gt, preds, rng), atol=1e-6)


def test_perfect_and_empty():
    gt, _ = synthetic(3)
    perfect = [{"image_id": a["image_id"], "category_id": 1, "bbox": a["bbox"], "score": 1.0}
               for a in gt["annotations"]]
    np.testing.assert_allclose(ours(gt, perfect), 1.0)
    np.testing.assert_allclose(ours(gt, []), 0.0)


def test_precision_recall_hand_case():
    g = np.array([[0, 0, 10, 10], [100, 100, 10, 10]], float)
    d = np.array([[0, 0, 10, 10], [50, 50, 10, 10], [100, 100, 10, 10]], float)
    m = match_image(g, np.zeros(2, bool), d, np.array([0.9, 0.8, 0.3]))
    assert precision_recall_at([m], conf=0.5) == (0.5, 0.5)   # 1 TP, 1 FP above 0.5; 1 of 2 GT
    assert precision_recall_at([m], conf=0.0) == (2 / 3, 1.0)
    # AP@0.5: precision envelope 1.0 up to recall 0.5, then 2/3 to recall 1.0
    ap = average_precision([m])[0]
    assert ap == pytest.approx((51 * 1.0 + 50 * 2 / 3) / 101)


def test_ignored_gt_not_penalised():
    g = np.array([[0, 0, 10, 10], [100, 100, 10, 10]], float)
    m = match_image(g, np.array([False, True]), np.array([[0, 0, 10, 10]], float), np.array([0.9]))
    assert m.n_gt == 1 and average_precision([m])[0] == pytest.approx(1.0)
    # a detection on the ignored box is neither TP nor FP
    m = match_image(g, np.array([False, True]), g.copy(), np.array([0.9, 0.8]))
    assert average_precision([m])[0] == pytest.approx(1.0)
