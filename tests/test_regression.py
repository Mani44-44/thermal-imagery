"""Golden-set regression gate (Phase 5).

Runs the candidate model on the frozen golden CI subset and fails if any tracked metric falls
more than its tolerance below golden/baseline.json. The candidate defaults to the baseline's
own model; set CANDIDATE_MODEL (a save_pretrained dir) and CANDIDATE_MODALITY to gate a new one.
The baseline only changes through a reviewed commit (scripts/promote.py).

    uv run pytest -m regression
"""

import json
import os
from pathlib import Path

import pytest
import torch

from src.eval.metrics import evaluate
from src.predict import load_model, predict
from tests.conftest import ANNOTATIONS, needs_data

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "golden" / "baseline.json"

pytestmark = [pytest.mark.regression, needs_data,
              pytest.mark.skipif(not BASELINE.exists(), reason="no golden/baseline.json yet")]


def metric_values(res: dict) -> dict:
    return {"mAP50": res["overall"]["mAP50"], "mAP50_95": res["overall"]["mAP50_95"],
            "recall": res["overall"]["recall"], "dark_mAP50": res["by_condition"]["dark"]["mAP50"]}


@pytest.fixture(scope="module")
def candidate_metrics():
    base = json.loads(BASELINE.read_text())
    model_dir = Path(os.environ.get("CANDIDATE_MODEL", ROOT / base["model"]["path"]))
    modality = os.environ.get("CANDIDATE_MODALITY", base["model"]["modality"])
    assert model_dir.exists(), f"{model_dir} missing (dvc pull / dvc checkout?)"
    gt = json.loads((ANNOTATIONS / f"{base['set']}.json").read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    preds = predict(load_model(str(model_dir), modality, coco80=False), gt["images"], modality,
                    batch_size=8, workers=0, device=device)
    res = evaluate(gt, preds, conf=base["conf"], n_boot=0)
    out = ROOT / "reports" / "regression_last.json"
    out.write_text(json.dumps({"model": str(model_dir), "modality": modality, "device": device,
                               "metrics": metric_values(res)}, indent=1))
    return base, metric_values(res)


@pytest.mark.parametrize("metric", ["mAP50", "mAP50_95", "recall", "dark_mAP50"])
def test_no_regression(candidate_metrics, metric):
    base, got = candidate_metrics
    floor = base["metrics"][metric] - base["tolerance"][metric]
    assert got[metric] >= floor, (f"{metric} regressed: {got[metric]:.4f} < baseline "
                                  f"{base['metrics'][metric]:.4f} - tol {base['tolerance'][metric]}")
