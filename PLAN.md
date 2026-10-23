# RGB + Thermal Detection: Execution Plan

**Goal:** fine-tune a detector three ways (RGB only, thermal only, and RGB+thermal fusion) and compare them fairly, especially in low light. The project also builds an evaluation harness, a CI regression gate, and a versioned data-quality loop around the models.

**Decisions:**
- **Detector: RT-DETR**, using the Hugging Face Transformers implementation (`RTDetrV2ForObjectDetection`, Apache-2.0) with COCO-pretrained R18 as the default and R50 if time allows. HF over Ultralytics because it's a new stack (YOLO was done already), the licence is permissive, and patching the backbone for 4-channel input is straightforward.
- **Personal project, public on GitHub.** Dataset files never go in git. They live in a private DVC remote.
- **Compute:** local RTX 4050 Laptop (6 GB) plus Kaggle. See the compute plan below.
- No fixed deadline. The phases are sequential, but the deliverable at the end of each phase is usable on its own.
- Dev machine: Windows 11. Python 3.11 in a pinned env (uv or conda).

## Status (2026-10-23)

| Phase | Exit criteria | Status | Deviations |
|---|---|---|---|
| 0 Setup | licences, alignment, DVC raw, lockfile, GPU numbers | met | `docs/licences.md`, `docs/alignment/`, `docs/compute.md` |
| 1 Data | `dvc repro`, leakage tests, golden manifest, EDA | met except EDA notebook | findings written to `docs/data-checks.md` instead of a notebook |
| 2 Eval harness | zero-shot scored, metric unit tests, latency | met | `reports/zeroshot.md`, `tests/test_metrics.py` |
| 3 Baselines | 2 models in DVC with CIs + latency | met | 2 seeds per arm, not 3 (s3 runs skipped/failed; see ledger); training ran on Kaggle, not as a DVC stage; outputs tracked in `results.dvc`, `models/*.dvc` |
| 4 Fusion | `reports/comparison.md`, numbers traceable to DVC | met | late fusion uses NMS (beat WBF on val); FiftyOne error gallery not produced |
| 5 Regression/CI | degraded model makes CI fail | met locally; GitHub run pending | `pytest -m regression` fails the fusion model on 4/4 checks (`reports/regression_gate.md`). `.github/workflows/ci.yml` (self-hosted runner) is written but not pushed: the gh token lacks the `workflow` scope |
| 6 Data quality | `data-v2` tag, v1-vs-v2 table with CIs, flagging precision | met | review done by the model (Claude), not a human; CVAT export provided but CVAT not hosted; SAM box tightening not used |
| 7 Extensions | optional | not started | |

## Compute plan

| | Local: RTX 4050 Laptop | Kaggle: T4 ×2 | Kaggle: P100 |
|---|---|---|---|
| Architecture | Ada (2023) | Turing (2018) | Pascal (2016) |
| VRAM | **6 GB** | 16 GB per card (×2) | 16 GB |
| Mixed precision | fp16 + **bf16** tensor cores | fp16 tensor cores, no bf16 | no tensor cores |
| Power | 75 W cap (as reported by `nvidia-smi`) | 70 W per card | 250 W |
| System RAM | 15.3 GB | 30 GB | 30 GB |
| Disk | **25 GB free on C:** | 20 GB persisted + scratch | same |
| Limits | none, but the laptop may throttle on long runs | 30 GPU-h/week, 12 h/session, idle timeout | same quota |

Measured (see `docs/compute.md`): **one Kaggle T4 is ~1.4× faster than the 4050** (power-capped at 75 W) and fits a 4× larger batch. R18 trains at 19 img/s locally vs 27 img/s on a T4. **VRAM is the bigger limit locally.** RT-DETR's transformer decoder is memory-hungry, so on 6 GB the max batch at 640 px is 8 for R18 and 4 for R50 (measured in Phase 0). Use gradient accumulation to reach the effective batch size.

**Split the work like this:**
- **Local:** all development and debugging, data tooling (FiftyOne, CVAT, DVC), the eval harness, short training runs, R18 trainings, and **all latency benchmarks**. Latency must come from one fixed device, and the 4050 is the one you always have.
- **Kaggle:** the bulk runs. That's 3 arms × 3 seeds, R50 models, and anything that doesn't fit in 6 GB. Run each training as a script with checkpoint/resume so the 12 h session limit doesn't matter. Use 1 T4 per run for simplicity. Use `torchrun` across both T4s only if a run is too slow. Pull data with `dvc pull`, keeping the remote credentials in Kaggle Secrets. T4 has no bf16, so use fp16 AMP there and watch for NaN losses.
- **Disk (25 GB free) is the tightest local constraint.** Don't materialise the 4-channel dataset; stack RGB+T on the fly in the dataloader. Set `dvc config cache.type hardlink,symlink` so the DVC cache doesn't duplicate the data. Free up space or use an external drive if the raw download plus extraction is close to the limit.

---

## Phase 0: Setup and data due diligence

**Goal:** decide on the dataset and toolchain before writing any modelling code.

Tasks
1. **Licence gate** (checklist; record the answers in `docs/licences.md`):
   - [ ] LLVIP terms of use: is it research-only? What attribution does it require? Can derived labels be redistributed?
   - [ ] FLIR ADAS terms: does it require registration? Is it research-only? Can it be redistributed?
   - [ ] Is it OK to publish trained weights and metrics publicly under the dataset terms? (The repo is public.)
   - [ ] HF Transformers / RT-DETR weights, SAM, FiftyOne, CVAT and DVC licences (all expected to be fine; confirm).
2. **Alignment check (this decides the primary dataset).** Early fusion only works if the RGB and IR images line up pixel for pixel.
   - LLVIP is distributed as registered pairs. Confirm it by overlaying about 20 random pairs (alpha blend) and checking that the edges line up.
   - FLIR ADAS uses separate RGB and thermal cameras with different fields of view, so pairs are **not** pixel-aligned. Early fusion on FLIR would need a registration step (homography per sequence). Otherwise FLIR supports late fusion only.
   - **Expected decision:** LLVIP is the primary dataset. FLIR is optional and secondary (Phase 7).
3. Download the data. Record checksums and the source URL and date.
4. Set up the repo: `git init`, `dvc init`, a pre-commit hook (ruff), a pinned lockfile, and a `README`.
5. Choose a DVC remote. Start with a local folder remote. Move to private S3-compatible storage (Backblaze B2, Cloudflare R2 or S3) before Phase 5, because CI and Kaggle need to pull data from it.
6. **GPU sanity check.** Install the CUDA build of PyTorch, then load `RTDetrV2ForObjectDetection` (R18, COCO-pretrained). Run a few training steps at 640 px with AMP and increase the batch size until you run out of memory. Record the maximum batch size and it/s locally and on a Kaggle T4 in `docs/compute.md`.

**Exit criteria:** licences are recorded, alignment is verified with images saved to `docs/`, the raw data is tracked by DVC, the environment reproduces from the lockfile, and the GPU memory/speed numbers are recorded.

---

## Phase 1: Data pipeline, splits and golden set

**Goal:** a reproducible, leakage-free dataset in three formats, with a frozen golden set.

Tasks
1. **Conversion script** (`src/data/convert.py`). Turns the source annotations (LLVIP is VOC XML) into **COCO JSON**, one file per split. This is what the HF image processor and the eval harness both consume. All modalities share one annotation file because the pairs are registered.
   - The dataset class takes a `modality` argument: `rgb` (visible), `thermal` (IR replicated to 3 channels so it works with pretrained weights) or `fused4` (RGB+T **stacked on the fly**, not saved to disk; see the compute plan).
2. **Scene IDs.** *(Found in Phase 0: the two-digit filename prefix 01–26 matches the 26 scenes; see `docs/data-checks.md`.)* Confirm by viewing a few images per prefix in FiftyOne, then save `scene_id` per image to `data/meta/scenes.csv`.
3. **Splits** (`src/data/split.py`). *(Found in Phase 0: the official test split is already scene-disjoint: 19–24 and 26.)* Keep the official test set, which also keeps results comparable with papers. Carve **val** out of the train scenes (01–18, 25) by holding out 2–3 whole scenes, ideally including one of the daytime scenes if they fall in train. Re-measure the alignment offset per scene (scene 08 showed ~9 px).
   - **Leakage checks** (as tests in `tests/test_splits.py`):
     - No `scene_id` appears in more than one split.
     - A near-duplicate scan across splits (perceptual hash or embedding cosine > threshold) returns nothing, or every hit is explained.
   - Add an **"RGB-visible" subset** flag on test boxes (e.g. by local contrast inside the box in the visible image). Some dark-scene pedestrians are nearly invisible in RGB, so RGB-only recall needs to be reported on both all boxes and RGB-visible boxes.
4. **Condition bins.** Measure the mean luminance of each RGB image (optionally also the local contrast). Bin it into `dark / dim / bright` using fixed thresholds saved in `params.yaml`. Don't rely on metadata. LLVIP is mostly night, so check how many images end up in each bin. If the bright bin is tiny, the "low light vs normal" contrast needs FLIR (Phase 7). Note this as a limitation.
5. **Golden set.** Carve out about 300–500 images from **test scenes only**, covering every condition bin and object size bucket. Freeze it **now**, before any training or label cleaning happens:
   - Save the file list plus the SHA-256 of each image and label to `golden/manifest.json`, tracked in DVC.
   - A test asserts the manifest hash, so the golden set can't drift silently.
6. **DVC pipeline** (`dvc.yaml`): `convert → split → golden` stages, parameterised from `params.yaml`.
7. **FiftyOne + EDA.** Load the data as a *grouped* dataset (`rgb` / `thermal` slices) with ground truth, `scene_id`, `split` and `condition` fields. Write a notebook covering box count per image, box size histogram (COCO small/medium/large), condition distribution and class balance.

**Exit criteria:** `dvc repro` rebuilds every split from raw data, the leakage tests pass, the golden manifest is committed, and the EDA notebook is saved with its findings.

---

## Phase 2: Evaluation harness (built before any training)

**Goal:** one command that scores any set of predictions and produces a comparable report.

Tasks
1. **Prediction format.** Use COCO-style JSON (`image_id, bbox, score, category_id`) for all three arms. That way the harness doesn't care which model produced the predictions.
2. **Metrics** (`src/eval/`), using `pycocotools` or `faster-coco-eval`:
   - mAP@0.5, mAP@0.5:0.95, and precision/recall at a fixed operating confidence
   - **Per size:** COCO small/medium/large area ranges. Also report custom pixel-height bins if almost no pedestrians at 1280×1024 count as COCO "small".
   - **Per condition:** the `dark / dim / bright` bins from Phase 1
   - **Per scene:** so a single bad scene can be spotted
   - **Uncertainty:** a bootstrap 95% CI computed by resampling *scenes*, not images
3. **Latency** (`src/eval/latency.py`):
   - Fixed hardware, batch size 1, 50 warmup iterations, then 500 timed.
   - Report p50 and p95, and state the image size, precision (fp16 or fp32) and device.
   - Time end to end: preprocessing + inference + NMS (+ box fusion for late fusion).
4. **Report.** Write `reports/<run_id>/metrics.json` (machine-readable) and `report.md` (tables and PR curves). Add `compare.py` to put N runs side by side.
5. **Smoke test.** Run COCO-pretrained RT-DETR (person class only) zero-shot on RGB and on thermal test images. This checks the harness end to end and gives you a "no fine-tuning" reference row.

**Exit criteria:** zero-shot baselines are scored. Unit tests on toy fixtures check the metrics (perfect predictions give mAP=1, empty predictions give 0, and a known case gives a hand-computed number). Latency numbers are reproducible within about 5% across runs.

---

## Phase 3: Single-modality baselines

**Goal:** fine-tuned RGB-only and thermal-only models, trained under identical settings.

**Fairness protocol** (written down once in `params.yaml` and reused by Phase 4):
- Same architecture and size (RT-DETRv2-R18 first; R50 on Kaggle later), same image size, epochs, optimiser, LR schedule and pretrained init. Keep the **effective** batch size the same everywhere (per-device batch × gradient accumulation), since the per-device batch differs between local and Kaggle.
- Write a custom training loop or use HF `Trainer` with a COCO collate function. RT-DETR is sensitive to LR: use the reference recipe (AdamW, lower LR for the backbone than for the head, warmup) as a starting point. Hold out a short convergence check (e.g. 10 epochs) before full runs.
- Same evaluation conf/NMS/IoU thresholds for every arm.
- Model selection on **val** only. Look at test **once per final model**.
- **Augmentation:** HSV/colour jitter is fine for RGB. For thermal, turn off hue and saturation jitter (they mean nothing there) and keep only a mild brightness/contrast change.
- **Seeds:** at least 3 seeds per arm, or rely on the scene bootstrap CIs. A single-run gap in mAP of about 1 point is noise.

Tasks
1. Train script `src/train/train.py --modality {rgb,thermal}`, run as a DVC stage, with metrics logged (DVC experiments, or MLflow/W&B if preferred).
2. Score both models with the Phase 2 harness.
3. Short write-up covering where each modality wins, by condition and by size.

**Exit criteria:** two trained models are tracked in DVC, and their reports include CIs and latency.

---

## Phase 4: Fusion and the comparison

**Goal:** one early-fusion model and one late-fusion system, compared with the baselines.

### 4a. Early fusion (4-channel input)
- Patch the backbone's **first conv layer** to accept 4 inputs. Copy the pretrained RGB weights into channels 0–2 and initialise channel 3 with the **mean of the RGB filters**. Also extend the image processor's normalisation mean/std to 4 channels (compute the thermal channel's stats from train).
- Check: overfit 10 images with the patched model before full training, to make sure the 4-channel path is wired correctly.
- **Augmentation:** geometric transforms (flip, scale, mosaic, crop) must apply **identically** to both modalities. Colour jitter applies to the RGB channels only.
- Train under the same fairness protocol as Phase 3.

### 4b. Late fusion (box-level)
- Run the Phase 3 RGB and thermal models and merge their outputs with **Weighted Boxes Fusion** (the `ensemble-boxes` package). Also try NMS-merge as a simpler baseline.
- Tune the fusion weights and IoU threshold on **val**.
- Latency counts **both** forward passes plus the fusion step. Report both sequential and parallel timings.

### 4c. Comparison report
- One table: RGB / thermal / early / late, with mAP@0.5, mAP@0.5:0.95, P, R, latency p50/p95 and parameter count.
- Breakdowns by condition and by size, with CIs. Headline question: *how much does fusion help in `dark`, and what does it cost in latency?*
- An error gallery from FiftyOne showing the cases each arm uniquely gets right or wrong.

**Exit criteria:** `reports/comparison.md` exists and every number traces back to a DVC-tracked run.

---

## Phase 5: Regression test and CI

**Goal:** CI fails when a new model performs worse on the golden set.

Tasks
1. **Baseline file** `golden/baseline.json`, committed to git. It records per-arm mAP@0.5, mAP@0.5:0.95, recall@conf and per-condition mAP on the golden set, plus a **tolerance** for each, e.g. `max_drop: 0.01`.
2. **Test** `tests/test_regression.py`:
   - Verify the golden manifest hash, run inference, compute metrics, and assert `new >= baseline - tolerance` for each metric.
   - Also assert that inference doesn't crash and the output format is valid.
3. **CI** (GitHub Actions). Hosted runners have **no GPU**:
   - Pull the golden set and the candidate weights with `dvc pull`, using DVC remote credentials stored as **repo secrets**.
   - Run on CPU with the R18 model and a capped golden subset (about 100–200 images, fixed in the manifest) so the job finishes in about 10 minutes. Cache pip packages and DVC data. Measure CPU time per image first and size the subset from that.
   - The repo is public: check that CI secrets aren't exposed to workflows triggered by forks' pull requests.
   - Run the full golden set on GPU locally or on a self-hosted runner if one is available. That run is optional.
   - Also run the fast unit tests on every push (metrics, splits, leakage, conversion).
4. **Baseline updates** happen deliberately, through a PR that changes `baseline.json`, never automatically.

**Exit criteria:** a deliberately degraded model makes CI **fail**. Train for 1 epoch, or add noise to the weights, and confirm the job goes red. The current model passes.

---

## Phase 6: Data-quality loop (FiftyOne → pre-label → CVAT → DVC v2)

**Goal:** find label errors and hard examples, fix them, and **measure** whether fixing them helps.

Tasks
1. **Find problems** with FiftyOne:
   - Load the Phase 3/4 predictions on **train and val**. Run `compute_mistakenness` (likely label errors and missing boxes), `compute_hardness`, `compute_uniqueness` and embedding visualisation.
   - Write saved views for: high-confidence false positives (often missing labels), low-IoU true positives (loose boxes), and thermal-only detections that RGB misses.
2. **Pre-label.** The fine-tuned detector's boxes are the practical pre-labels. SAM produces *masks* and needs prompts, so use **box-prompted SAM only to tighten boxes**, not as a detector. Try Grounding DINO + SAM only if you need labels for a new class.
3. **Review in CVAT.** Use `dataset.annotate(backend="cvat")` from FiftyOne. Host CVAT with Docker Desktop on Windows, or use app.cvat.ai. A human accepts or edits every change. Log reviewer decisions so you can report "X% of flagged samples were real errors".
4. **Version it.** Record the corrected labels as **dataset v2** in DVC (git tag `data-v2`). Keep v1 untouched.
5. **Measure.** Retrain the best arm on v2 with the same protocol. Compare v1 and v2 on the same test set and the golden set.

**Rules against circular cleaning:**
- Never auto-label or "clean" the **golden** or **test** sets with the model being evaluated.
- If test labels are fixed at all, do it in a separate blind review pass. Report metrics on **both** the original and the corrected test labels.

**Exit criteria:** dataset v2 is tagged, and a v1-vs-v2 table with CIs exists, along with a precision figure for the flagging step.

---

## Phase 7: Extensions (optional)

- **R18 vs R50**, or a newer DETR variant (e.g. D-FINE), run under the same harness and protocol.
- **YOLO as a reference row:** you've used it before, so a quick YOLO baseline through the same harness shows how RT-DETR compares.
- **FLIR ADAS** as a second dataset. It has real day/night variation, so the low-light comparison gets a proper control. Use late fusion only, unless you build per-sequence registration.
- **Export:** ONNX → TensorRT/OpenVINO. Re-run latency and check the accuracy drop on the golden set.
- **Write-up:** a final README or blog post with the comparison table, error gallery and lessons learned.

---

## Proposed repo layout

```
Thermal Imagery/
├── PLAN.md
├── README.md
├── params.yaml            # all thresholds, split ratios, train hyperparams
├── dvc.yaml               # convert → split → golden → train → eval stages
├── data/                  # DVC-tracked: raw/, annotations/ (COCO JSON per split), meta/
├── golden/                # manifest.json (DVC), baseline.json (git)
├── src/
│   ├── data/              # convert.py, split.py, conditions.py, golden.py
│   ├── train/             # train.py, fusion_model.py (4-ch patch)
│   ├── fusion/            # late_fusion.py (WBF)
│   └── eval/              # metrics.py, breakdowns.py, latency.py, compare.py
├── tests/                 # test_metrics, test_splits, test_regression, ...
├── notebooks/             # EDA, error analysis
├── reports/               # per-run metrics.json + report.md, comparison.md
├── docs/                  # licences.md, alignment checks, decisions
└── .github/workflows/ci.yml
```

## Phase dependencies

```
0 ─► 1 ─► 2 ─► 3 ─► 4 ─► 6
               │    └──► 5 (can start once the first model from Phase 3 exists)
               └────────► 7 (optional, any time after 4)
```

## Top risks

| Risk | Mitigation |
|---|---|
| Too few bright images in LLVIP for a low-light contrast | Measured condition bins; add FLIR in Phase 7; state the limitation |
| Scene IDs not recoverable cleanly | Embedding clustering plus manual spot check; leakage tests |
| 4-channel patch breaks pretrained behaviour | Mean-RGB init for channel 3, overfit check on 10 images (Phase 4a) |
| 6 GB VRAM too small for RT-DETR | Small batch + grad accumulation + AMP locally; larger runs on Kaggle |
| RT-DETR slow to converge / LR-sensitive | Reference recipe, short convergence check before full runs |
| Low local disk (25 GB free) | On-the-fly 4-channel stacking, DVC hardlink/symlink cache, external drive if needed |
| Kaggle session limits (12 h, 30 h/week) | Checkpoint/resume; plan the seed sweep across weeks |
| Differences between arms are just noise | Seeds plus scene bootstrap CIs, and a fixed protocol |

## Open questions

1. Do you have a deadline? If so, Phases 0–5 are the core project and 6–7 can be cut down.
