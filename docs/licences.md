# Licences and terms (Phase 0 gate)

Checked 2026-10-22.

## LLVIP (primary dataset)

Source: https://github.com/bupt-ai-cz/LLVIP, "Term of Use and License.md" (last updated 2023-01-03).

- **Allowed:** free use by academic and non-academic entities for **non-commercial** purposes, explicitly
  including "personal experimentation". This project qualifies (personal, public GitHub, no commercial use).
- **Must:** reference the LLVIP dataset in any work using it, i.e. cite the paper or link the LLVIP website
  (README does this).
- **If we release new annotations** (e.g. the corrected labels from Phase 6), we must state how they relate to
  the original LLVIP annotations, tell users to cite LLVIP, and agree that the LLVIP authors may republish them.
- **Must not:** use the dataset or any derivative work for commercial gain, or use it to identify people.
- **Not addressed explicitly:** publishing trained weights. Treated as a non-commercial derivative work: OK to
  publish with attribution and a non-commercial note. Revisit if that changes.
- **Repo policy:** no LLVIP images or labels in git. Data lives in a private DVC remote only.

Dataset facts from the same files: 26 scenes (24 dark, 2 daytime). The README notes that a small part of the
annotations was corrected on 2023-02-21. Download links are in the repo's `download_dataset.md`: Google
Drive or Baidu Yun, registered pairs. Raw unregistered pairs and videos are offered separately.

## Software

| Component | Licence | Notes |
|---|---|---|
| Hugging Face Transformers (RT-DETR / RT-DETRv2) | Apache-2.0 | COCO-pretrained checkpoints: confirm the model card licence when one is picked |
| PyTorch / torchvision | BSD-3 | |
| DVC | Apache-2.0 | |
| FiftyOne | Apache-2.0 | |
| CVAT | MIT | |
| SAM / SAM 2 | Apache-2.0 | |
| ensemble-boxes (WBF) | MIT | |

## FLIR ADAS (optional, Phase 7)

Not checked yet. Do it before Phase 7.
