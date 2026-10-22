"""Side-by-side Markdown report for evaluated runs.

Each run directory holds metrics_<split>.json (from src.eval.metrics) and optionally latency.json
and run.json (training metadata). Optionally adds paired-bootstrap deltas against a reference run.

    uv run python -m src.eval.report --runs runs/zeroshot/rgb runs/zeroshot/thermal --split test \
        --out reports/zeroshot.md
"""

import argparse
import json
from pathlib import Path

from src.eval.metrics import paired_delta

ROOT = Path(__file__).resolve().parents[2]


def _f(x, nd=3):
    return "–" if x is None or x != x else f"{x:.{nd}f}"


def _ci(ci):
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ""


def load(run: Path, split: str) -> dict:
    m = json.loads((run / f"metrics_{split}.json").read_text())
    lat = json.loads((run / "latency.json").read_text()) if (run / "latency.json").exists() else None
    meta = json.loads((run / "run.json").read_text()) if (run / "run.json").exists() else {}
    return {"name": meta.get("name", run.name), "m": m, "lat": lat, "meta": meta, "dir": run}


def render(runs: list[dict], split: str, deltas: dict | None = None, ref: str | None = None) -> str:
    L = [f"## Results on `{split}`", "",
         "| Model | mAP@0.5 | 95% CI | mAP@0.5:0.95 | 95% CI | P | R | p50 ms | p95 ms |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        o, ci = r["m"]["overall"], r["m"].get("ci95") or {}
        lat = r["lat"] or {}
        L.append(f"| {r['name']} | {_f(o['mAP50'])} | {_ci(ci.get('mAP50'))} | {_f(o['mAP50_95'])} | "
                 f"{_ci(ci.get('mAP50_95'))} | {_f(o['precision'])} | {_f(o['recall'])} | "
                 f"{_f(lat.get('p50_ms'), 1)} | {_f(lat.get('p95_ms'), 1)} |")
    conf = runs[0]["m"]["conf"]
    L += ["", f"P/R at confidence ≥ {conf}, IoU 0.5. CIs: location-level bootstrap (1000 resamples).", ""]

    L += ["### By lighting condition (mAP@0.5 / mAP@0.5:0.95)", "",
          "| Model | dark | dim | lit |", "|---|---|---|---|"]
    for r in runs:
        c = r["m"]["by_condition"]
        L.append(f"| {r['name']} | " + " | ".join(
            f"{_f(c[k]['mAP50'])} / {_f(c[k]['mAP50_95'])}" if k in c else "–" for k in ("dark", "dim", "lit")) + " |")
    L += ["", "### By box height (mAP@0.5): small <150 px, medium 150–250, large ≥250", "",
          "| Model | small | medium | large | RGB-visible boxes only |", "|---|---|---|---|---|"]
    for r in runs:
        s = r["m"]["by_size"]
        L.append(f"| {r['name']} | {_f(s['small']['mAP50'])} | {_f(s['medium']['mAP50'])} | "
                 f"{_f(s['large']['mAP50'])} | {_f(r['m']['rgb_visible']['mAP50'])} |")
    locs = sorted(runs[0]["m"]["by_location"], key=int)
    L += ["", "### By location (mAP@0.5)", "", "| Model | " + " | ".join(locs) + " |",
          "|---|" + "---|" * len(locs)]
    for r in runs:
        b = r["m"]["by_location"]
        L.append(f"| {r['name']} | " + " | ".join(_f(b[k]["mAP50"]) for k in locs) + " |")
    if deltas:
        L += ["", f"### Paired differences vs {ref} (location bootstrap, 95% CI)", "",
              "| Model | Δ mAP@0.5 | CI | Δ mAP@0.5:0.95 | CI | Δ mAP@0.5 dark | CI |", "|---|---|---|---|---|---|---|"]
        for name, d in deltas.items():
            a, dk = d["all"], d["dark"]
            L.append(f"| {name} | {a['mAP50']['delta']:+.3f} | {_ci(a['mAP50']['ci95'])} | "
                     f"{a['mAP50_95']['delta']:+.3f} | {_ci(a['mAP50_95']['ci95'])} | "
                     f"{dk['mAP50']['delta']:+.3f} | {_ci(dk['mAP50']['ci95'])} |")
    return "\n".join(L) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--ref", type=Path, help="run to compute paired deltas against")
    parser.add_argument("--title", default="Evaluation report")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    runs = [load(r, args.split) for r in args.runs]
    deltas = None
    if args.ref:
        gt = json.loads((ROOT / "data" / "annotations" / f"{args.split}.json").read_text())
        ref_pred = json.loads((args.ref / f"pred_{args.split}.json").read_text())
        deltas = {}
        for r in runs:
            if r["dir"] == args.ref:
                continue
            pred = json.loads((r["dir"] / f"pred_{args.split}.json").read_text())
            deltas[r["name"]] = {"all": paired_delta(gt, ref_pred, pred),
                                 "dark": paired_delta(gt, ref_pred, pred, condition="dark")}
    ref_name = load(args.ref, args.split)["name"] if args.ref else None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(f"# {args.title}\n\n" + render(runs, args.split, deltas, ref_name),
                        encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
