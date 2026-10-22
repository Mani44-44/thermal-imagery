# Thermal Imagery: RGB vs thermal vs fusion pedestrian detection

RT-DETR fine-tuned on LLVIP three ways (RGB only, thermal only, RGB+thermal fusion), with an evaluation
harness, a golden-set regression test in CI, and a DVC-versioned data-quality loop.
See [PLAN.md](PLAN.md) for the phased plan.

## Setup

```bash
uv sync                     # Python 3.11 env with CUDA PyTorch, from uv.lock
uv run python scripts/gpu_check.py
```

## Dataset

This project uses **LLVIP: A Visible-infrared Paired Dataset for Low-light Vision**
([website](https://bupt-ai-cz.github.io/LLVIP/), [paper](https://arxiv.org/abs/2108.10831)).
LLVIP is licensed for non-commercial use only; see [docs/licences.md](docs/licences.md). The images are
not redistributed in this repository.

```bibtex
@inproceedings{jia2021llvip,
  title={LLVIP: A visible-infrared paired dataset for low-light vision},
  author={Jia, Xinyu and Zhu, Chuang and Li, Minzhen and Tang, Wenqi and Zhou, Wenli},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={3496--3504},
  year={2021}
}
```
