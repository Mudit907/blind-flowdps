# Blind-FlowDPS

**Blind Inverse Problem Solving with Flow-Matching Diffusion Models**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **Research project** — UTM Vicube Lab × UNSW Sydney  
> Status: Active development | Target venue: CVPR / NeurIPS 2027

---

## What this is

All existing flow-matching inverse solvers (FlowDPS, FlowLPS, MS-Flow, D-Flow) assume the degradation operator is **known at inference time**. Real-world images are degraded by unknown blur kernels, unknown noise levels, and unknown compression artifacts.

This project extends flow-matching posterior sampling to the **blind** setting — where neither the kernel shape nor its parameters are known — through online kernel estimation coupled with the diffusion data-consistency loop.

**Key findings (so far):**
- Blind kernel estimation via gradient descent on `L(σ) = ||y - A(x; σ)||²` recovers degradation parameters in ~15 Adam steps for well-conditioned images
- Convergence depends on image conditioning properties — we study *when* and *why* estimation succeeds or fails
- A discrete optimization phenomenon: even-valued blur parameters outperform odd-valued by ~37 percentage points (under investigation)

---

## Repository structure

```
blind-flowdps/
├── src/
│   ├── operators/          # Degradation operators (parametric + non-parametric)
│   ├── solvers/            # FlowDPS core, data consistency loop
│   ├── metrics/            # PSNR, SSIM, kernel error metrics
│   └── utils/              # Logging, image I/O, reproducibility
├── experiments/
│   ├── configs/            # YAML config per experiment
│   └── results/            # Auto-generated from W&B (not hand-edited)
├── scripts/
│   ├── evaluate.py         # Full DIV2K evaluation pipeline
│   ├── ablate_sigma.py     # Even/odd sigma sweep
│   └── benchmark.py        # vs Richardson-Lucy and baselines
├── api/
│   ├── main.py             # FastAPI inference endpoint
│   └── Dockerfile
├── notebooks/
│   └── exploration/        # Kaggle history and exploratory work
├── docs/
│   ├── method.md           # Method writeup
│   └── results.md          # Running results log
├── tests/                  # Unit tests for operators and metrics
├── paper/                  # LaTeX source
├── docker-compose.yml
└── pyproject.toml
```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/yourusername/blind-flowdps.git
cd blind-flowdps
pip install -e ".[dev]"
```

### 2. Run inference on a single image

```bash
python scripts/evaluate.py \
  --image path/to/blurry_image.png \
  --task deblur_gauss \
  --deg_scale 8 \
  --NFE 28 \
  --output_dir results/
```

### 3. Run via Docker

```bash
docker compose up api
# POST http://localhost:8000/restore with your image
```

### 4. Run the full DIV2K evaluation

```bash
python scripts/evaluate.py \
  --dataset div2k \
  --data_dir data/DIV2K_valid_HR \
  --config experiments/configs/div2k_eval.yaml \
  --log_wandb
```

---

## Method overview

```
Degraded image y = A(x; σ) + n    [σ unknown]

For each diffusion timestep t:
  1. Decode latent z_t → image estimate x̂_t
  2. Kernel estimation:  σ* = argmin_σ ||y - A(x̂_t; σ)||²   [Adam, 15 steps]
  3. Data consistency:   z_t ← z_t - η·∇_z ||A^T(y;σ*) - A^T(A(x̂_t;σ*);σ*)||²
  4. Diffusion step:     z_{t-1} = ODE_solver(z_t, score, t)

Output: x̂_0 = decode(z_0)
```

See [`docs/method.md`](docs/method.md) for full mathematical derivation.

---

## Results

| Method | PSNR (dB) | SSIM | Notes |
|--------|-----------|------|-------|
| Degraded input | 21.57 | — | σ=8, DIV2K |
| Richardson-Lucy | — | — | Pending |
| **Blind-FlowDPS (ours)** | **23.36** | — | +1.79 dB, 3-image |
| Oracle (known σ) | 22.99 | — | Upper bound |

*Full 100-image DIV2K evaluation in progress.*

---

## Reproducing results

Every experiment is logged to Weights & Biases. Config files in `experiments/configs/` are the single source of truth for hyperparameters. To reproduce:

```bash
python scripts/evaluate.py --config experiments/configs/div2k_eval.yaml
```

Results are written to `experiments/results/` as structured JSON and automatically synced to W&B.

---

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Format code
black src/ scripts/ api/
isort src/ scripts/ api/

# Type check
mypy src/
```

---

## Citation

If you use this work, please cite:

```bibtex
@article{mudit2026blindflowdps,
  title={Blind Inverse Problem Solving with Flow-Matching Diffusion Models},
  author={Mudit},
  year={2026},
  note={Under review}
}
```

---

## Acknowledgements

Built on [FlowDPS](https://github.com/FlowDPS) (Zhao et al., 2023) and Stable Diffusion 3.  
Research conducted at UTM Vicube Lab under Prof. Tarmizi bin Adam.
