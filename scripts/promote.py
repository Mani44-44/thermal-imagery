"""Promote a trained run to be the CI-gated model and (re)write golden/baseline.json.

Copies <run>/best to models/<name>, tracks it with DVC, scores it on the golden CI subset with the
same code path as tests/test_regression.py, and writes the baseline with tolerances. Commit the
result (baseline.json + models/<name>.dvc) in a reviewed commit; CI never updates it.

    uv run python scripts/promote.py --run runs/kaggle/out/tir-fused4-s1/run --modality fused4 --name fused4_s1
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.eval.metrics import evaluate  # noqa: E402
from src.predict import load_model, predict  # noqa: E402
from tests.test_regression import metric_values  # noqa: E402

TOLERANCE = {"mAP50": 0.01, "mAP50_95": 0.01, "recall": 0.02, "dark_mAP50": 0.02}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--modality", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--conf", type=float, default=0.5)
    a = p.parse_args()

    dst = ROOT / "models" / a.name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(a.run / "best", dst)
    subprocess.run(["uv", "run", "dvc", "add", str(dst.relative_to(ROOT))], cwd=ROOT, check=True)

    gt = json.loads((ROOT / "data" / "annotations" / "golden_ci.json").read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    preds = predict(load_model(str(dst), a.modality, coco80=False), gt["images"], a.modality,
                    batch_size=8, workers=0, device=device)
    metrics = metric_values(evaluate(gt, preds, conf=a.conf, n_boot=0))
    run_meta = json.loads((a.run / "run.json").read_text()) if (a.run / "run.json").exists() else {}
    baseline = {"description": "CI regression baseline. Change only via scripts/promote.py in a reviewed commit.",
                "model": {"path": f"models/{a.name}", "modality": a.modality, "name": a.name,
                          "trained_commit": run_meta.get("git_commit")},
                "set": "golden_ci", "conf": a.conf, "scored_on": device,
                "metrics": {k: round(v, 4) for k, v in metrics.items()}, "tolerance": TOLERANCE}
    (ROOT / "golden" / "baseline.json").write_text(json.dumps(baseline, indent=1))
    print(json.dumps(baseline, indent=1))


if __name__ == "__main__":
    main()
