"""Fast COCO-style detection matching and AP, in numpy.

Matching follows pycocotools (COCOeval.evaluateImg) for a single category:
- detections sorted by score, top `max_dets` per image;
- each detection takes the highest-IoU still-unmatched GT at or above the IoU threshold, preferring
  non-ignored GTs (a match to an ignored GT is only taken if no non-ignored GT qualifies);
- detections matched to ignored GTs are dropped. So are unmatched detections whose own "size" is
  outside the evaluated range (the area-range rule).
AP uses COCO's 101-point interpolated precision envelope.

Unlike pycocotools, ignored GTs here use plain IoU (no iscrowd semantics), and the size of a box
is whatever `size_fn` returns (we use box height, see params.yaml).

Matching is done once per image (`match_images`); any subset or bootstrap resample of images is
then scored by concatenating per-image results (`average_precision`), which makes location
bootstraps cheap.
"""

from dataclasses import dataclass

import numpy as np

IOU_THRS = np.linspace(0.5, 0.95, 10)
RECALL_PTS = np.linspace(0, 1, 101)


def iou_xywh(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between boxes a (N,4) and b (M,4), xywh."""
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2[None]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = iw * ih
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None] - inter
    return np.where(union > 0, inter / union, 0.0)


@dataclass
class ImageMatch:
    scores: np.ndarray   # (D,) kept detections, sorted by score desc
    tp: np.ndarray       # (T, D) bool: true positive at each IoU threshold
    valid: np.ndarray    # (T, D) bool: counts at all (not matched-to-ignored, not out-of-range)
    n_gt: int            # non-ignored GT boxes


def match_image(gt: np.ndarray, gt_ignore: np.ndarray, det: np.ndarray, scores: np.ndarray,
                det_out_of_range: np.ndarray | None = None, iou_thrs=IOU_THRS,
                max_dets: int = 100) -> ImageMatch:
    order = np.argsort(-scores, kind="mergesort")[:max_dets]
    det, scores = det[order], scores[order]
    det_oor = (np.zeros(len(det), bool) if det_out_of_range is None
               else np.asarray(det_out_of_range)[order])
    g_order = np.argsort(gt_ignore, kind="mergesort")  # non-ignored first
    gt, gt_ignore = gt[g_order], gt_ignore[g_order].astype(bool)
    T, D, G = len(iou_thrs), len(det), len(gt)
    tp = np.zeros((T, D), bool)
    matched_ignore = np.zeros((T, D), bool)
    if D and G:
        ious = iou_xywh(det, gt)
        for t, thr in enumerate(iou_thrs):
            gt_taken = np.zeros(G, bool)
            for d in range(D):
                best, best_iou = -1, min(thr, 1 - 1e-10)
                for g in range(G):
                    if gt_taken[g]:
                        continue
                    # once matched to a real GT, stop when reaching ignored GTs (sorted last)
                    if best > -1 and not gt_ignore[best] and gt_ignore[g]:
                        break
                    if ious[d, g] < best_iou:
                        continue
                    best, best_iou = g, ious[d, g]
                if best == -1:
                    continue
                gt_taken[best] = True
                if gt_ignore[best]:
                    matched_ignore[t, d] = True
                else:
                    tp[t, d] = True
    unmatched_oor = ~tp & ~matched_ignore & det_oor[None]
    valid = ~matched_ignore & ~unmatched_oor
    return ImageMatch(scores, tp, valid, int((~gt_ignore).sum()))


def average_precision(matches: list[ImageMatch]) -> np.ndarray:
    """COCO AP per IoU threshold (T,), over the concatenation of the given images.
    Returns NaN where there are no non-ignored GTs."""
    n_gt = sum(m.n_gt for m in matches)
    T = matches[0].tp.shape[0] if matches else len(IOU_THRS)
    if n_gt == 0:
        return np.full(T, np.nan)
    scores = np.concatenate([m.scores for m in matches]) if matches else np.zeros(0)
    order = np.argsort(-scores, kind="mergesort")
    aps = np.zeros(T)
    for t in range(T):
        tp = np.concatenate([m.tp[t] for m in matches])[order]
        valid = np.concatenate([m.valid[t] for m in matches])[order]
        tp, fp = tp[valid], ~tp[valid]
        tpc, fpc = np.cumsum(tp), np.cumsum(fp)
        recall = tpc / n_gt
        precision = tpc / np.maximum(tpc + fpc, np.finfo(float).eps)
        precision = np.maximum.accumulate(precision[::-1])[::-1]  # envelope
        idx = np.searchsorted(recall, RECALL_PTS, side="left")
        q = np.zeros(len(RECALL_PTS))
        ok = idx < len(precision)
        q[ok] = precision[idx[ok]]
        aps[t] = q.mean()
    return aps


def precision_recall_at(matches: list[ImageMatch], conf: float, t: int = 0) -> tuple[float, float]:
    """Precision and recall at a confidence threshold, IoU threshold index t (0 -> IoU 0.5)."""
    n_gt = sum(m.n_gt for m in matches)
    tp = fp = 0
    for m in matches:
        keep = (m.scores >= conf) & m.valid[t]
        tp += int(m.tp[t][keep].sum())
        fp += int((~m.tp[t][keep]).sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / n_gt if n_gt else float("nan")
    return precision, recall
