"""Phase 0 GPU sanity check: find the largest RT-DETR training batch that fits, and its speed.

Runs real forward+backward+optimizer steps with AMP on random images and dummy boxes, doubling the
batch size until CUDA runs out of memory. Results are printed and appended to docs/compute.md.

    uv run python scripts/gpu_check.py --model PekingU/rtdetr_v2_r18vd --size 640
"""

import argparse
import platform
import time
from datetime import date
from pathlib import Path

import torch
from transformers import RTDetrV2ForObjectDetection

ROOT = Path(__file__).resolve().parents[1]


def make_batch(batch_size: int, size: int, device: torch.device):
    pixel_values = torch.rand(batch_size, 3, size, size, device=device)
    labels = []
    for _ in range(batch_size):
        n = 8  # LLVIP images typically hold a handful of pedestrians
        centers = torch.rand(n, 2, device=device) * 0.8 + 0.1
        wh = torch.rand(n, 2, device=device) * 0.1 + 0.02
        labels.append({"class_labels": torch.zeros(n, dtype=torch.long, device=device),
                       "boxes": torch.cat([centers, wh], dim=1)})
    return pixel_values, labels


def bench(model, optimizer, batch_size, size, device, amp_dtype, steps=10, warmup=3):
    """Return (iterations/s, peak GB) for training at this batch size, or None on OOM."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        pixel_values, labels = make_batch(batch_size, size, device)
        for i in range(warmup + steps):
            if i == warmup:
                torch.cuda.synchronize()
                start = time.perf_counter()
            with torch.autocast("cuda", dtype=amp_dtype):
                loss = model(pixel_values=pixel_values, labels=labels).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    except (torch.OutOfMemoryError, torch.AcceleratorError) as e:
        if "out of memory" not in str(e):
            raise
        optimizer.zero_grad(set_to_none=True)
        return None
    peak = torch.cuda.max_memory_allocated()
    # On Windows (WDDM) the driver may spill into shared system RAM instead of raising OOM, which
    # "works" but is several times slower. Treat exceeding physical VRAM as a failure.
    if peak > torch.cuda.get_device_properties(device).total_memory:
        return None
    return steps / elapsed, peak / 1e9


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="PekingU/rtdetr_v2_r18vd")
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default=None,
                        help="default: bf16 if supported, else fp16")
    parser.add_argument("--max-batch", type=int, default=64)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA not available: check the PyTorch install"
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    dtype_name = args.dtype or ("bf16" if torch.cuda.is_bf16_supported() else "fp16")
    amp_dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float16

    print(f"torch {torch.__version__} | CUDA {torch.version.cuda} | {props.name} "
          f"| {props.total_memory / 1e9:.1f} GB | AMP {dtype_name}")

    model = RTDetrV2ForObjectDetection.from_pretrained(
        args.model, num_labels=1, ignore_mismatched_sizes=True).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # fp16 needs a GradScaler in real training; for a memory/speed probe plain steps are enough.

    rows, batch_size = [], 1
    while batch_size <= args.max_batch:
        result = bench(model, optimizer, batch_size, args.size, device, amp_dtype)
        if result is None:
            print(f"batch {batch_size:>3}: OOM")
            break
        its, peak = result
        rows.append((batch_size, its, its * batch_size, peak))
        print(f"batch {batch_size:>3}: {its:6.2f} it/s  {its * batch_size:6.1f} img/s  peak {peak:.2f} GB")
        batch_size *= 2

    if not rows:
        print("Even batch 1 does not fit.")
        return
    best = rows[-1]
    out = ROOT / "docs" / "compute.md"
    if not out.exists():
        out.write_text("# Compute measurements\n\nTraining probe from `scripts/gpu_check.py` "
                       "(random data, AMP, AdamW).\n\n| Date | Device | Model | Size | AMP | Max batch "
                       "| it/s | img/s | Peak GB |\n|---|---|---|---|---|---|---|---|---|\n")
    with out.open("a") as f:
        f.write(f"| {date.today()} | {props.name} ({platform.node()}) | {args.model} | {args.size} "
                f"| {dtype_name} | {best[0]} | {best[1]:.2f} | {best[2]:.1f} | {best[3]:.2f} |\n")
    print(f"Max batch {best[0]} -> recorded in {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
