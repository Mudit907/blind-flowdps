# Method: Blind-FlowDPS

**Blind inverse problem solving with flow-matching diffusion models.**

---

## Problem formulation

We observe a degraded image:

```
y = A(x; σ) + n
```

where:
- `x ∈ ℝ^(H×W×C)` is the unknown clean image
- `σ = (σ_x, σ_y) ∈ ℝ²` are the unknown degradation parameters
- `A(x; σ) = K_σ * x` is Gaussian convolution with unknown kernel
- `n ~ N(0, σ_n²)` is additive Gaussian noise

In the **blind** setting, neither `σ_x` nor `σ_y` is known at inference time.

---

## Why existing solvers fail

All existing flow-matching inverse solvers (FlowDPS, FlowLPS, MS-Flow, D-Flow, Restora-Flow) require the degradation operator `A` to be fully specified at inference time. When `σ` is unknown, these methods cannot run without a prior estimate — which in practice means either:

1. Guessing `σ` (degrades performance by up to 5.35 dB when wrong)
2. Estimating `σ` with a separate network (requires training data and retraining for each degradation type)

We propose a training-free, online approach that estimates `σ` jointly with the diffusion sampling process.

---

## Our approach

### Kernel estimation objective

The key insight is that the measurement consistency loss:

```
L(σ) = ||y - A(x̂; σ)||²
```

has a well-defined minimum at `σ_true` when `x̂` is a reasonable image estimate. This can be verified by taking the gradient:

```
∂L/∂σ = -2 * ⟨y - A(x̂; σ), ∂A(x̂; σ)/∂σ⟩
```

At `σ = σ_true`, `A(x̂; σ) ≈ y`, so the gradient is near zero — confirming a minimum.

**The wrong objective** (explored and abandoned):

```
L_wrong(σ) = ||A(x̂; σ) - A^T(y; σ)||²
```

This is monotonically decreasing in `σ` for Gaussian kernels and has no minimum — gradient descent diverges. This was discovered empirically and confirmed analytically.

### Algorithm

At each diffusion timestep `t`:

```
1. Decode latent z_t to image estimate x̂_t = decode(z_t)

2. Kernel estimation (blind mode only):
   σ* = Adam(σ₀, ∇_σ ||y - A(x̂_t; σ)||², K=15 steps)

3. Data consistency:
   L_dc = ||A^T(y; σ*) - A^T(A(x̂_t; σ*); σ*)||²
   z_t ← z_t - η · ∇_z L_dc

4. Diffusion step:
   z_{t-1} = ODE_solver(z_t, score_model, t)
```

### Why data consistency uses A^T

The data consistency loss `L_dc = ||A^T(y) - A^T(A(x))||²` is preferred over the simpler `||y - A(x)||²` because:

1. `A^T(y)` projects the measurement into image space, where the diffusion model operates
2. Gradients w.r.t. `z_t` are better conditioned (empirically observed)
3. For Gaussian blur (self-adjoint: `A = A^T`), this reduces to enforcing `A(x) ≈ y` in kernel space

---

## Operator implementation

The Gaussian blur operator is implemented as a depthwise convolution:

```python
# Forward: A(x; σ) = K_σ * x
kernel = make_gaussian_kernel(sigma_x, sigma_y)  # (1, 1, 33, 33)
kernel_expanded = kernel.expand(C, 1, 33, 33)    # depthwise: one kernel per channel
output = F.conv2d(x_padded, kernel_expanded, groups=C)
```

Key design choices:
- **Kernel size 33**: Large enough to capture blur up to σ=25 (6σ rule: 6×25/π ≈ 47; we use 33 for efficiency)
- **Reflect padding**: Avoids border artefacts better than zero padding
- **Differentiable w.r.t. σ**: The kernel is built from differentiable PyTorch ops, so gradients flow to σ_x and σ_y

---

## Discovered phenomenon: even/odd sigma discretization

During systematic ablation over `σ ∈ {5, 6, 7, ..., 15}`, we observed:

| Sigma type | Mean performance |
|------------|-----------------|
| Even (6, 8, 10, 12, 14) | ~96% |
| Odd (5, 7, 9, 11, 13, 15) | ~59% |

**Difference: ~37 percentage points.**

This is not noise — the pattern is consistent across multiple images and runs.

### Hypotheses under investigation

1. **Kernel size interaction**: Our kernel size is 33 (odd). Even-valued σ may interact more favourably with odd kernel dimensions in the depthwise convolution.

2. **VAE latent space**: SD3's VAE operates at 8× spatial downsampling (256→32). Even-valued σ may align better with the latent grid.

3. **Gradient flow**: The Gaussian kernel derivative `∂K_σ/∂σ` may have numerical properties that differ between even and odd σ values due to floating-point discretization on the coordinate grid.

4. **Optimization landscape**: The loss surface `L(σ) = ||y - A(x; σ)||²` may have different curvature near even vs. odd integers.

**Status**: Under active investigation. See `scripts/ablate_sigma.py` for the experiment that generates this data.

---

## Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Kernel size | 33 | Covers 6σ range for σ_max=25 |
| σ_init | (10.0, 10.0) | Intentionally offset from typical true σ (5–15) |
| σ_min | 0.5 | Physically minimum meaningful blur |
| σ_max | 25.0 | Maximum tested degradation |
| Kernel iterations | 15 | ~0.1 dB loss vs 80 iterations; 5× speedup |
| Kernel LR | 0.3 | Adam; higher works for coarse estimation |
| DC stepsize η | 15.0 | Empirically tuned |
| DC iterations | 7 | Per diffusion step |
| NFE | 28 | Diffusion steps |
| cfg_scale | 2.0 | Classifier-free guidance |

---

## Ablation: kernel estimation is critical

| Condition | PSNR (dB) | Δ vs blind |
|-----------|-----------|------------|
| Blind (our method) | 23.92 | — |
| Fixed σ=8 (oracle) | 22.99 | −0.93 |
| Fixed σ=5 (wrong) | 18.57 | −5.35 |
| Fixed σ=15 (wrong) | 22.26 | −1.66 |

**Using the wrong σ causes up to 5.35 dB PSNR loss.** This justifies the complexity of online kernel estimation — it is not optional.

Notably, blind estimation (+23.92) slightly exceeds oracle (+22.99). Hypothesis: measurement noise during kernel estimation acts as a regulariser, finding a blur model that better explains the noisy measurement than the true noiseless kernel would.

---

## References

- FlowDPS: Zhao et al. (2023)
- SD3: Esser et al. (2024) — Scaling Rectified Flow Transformers for High-Resolution Image Synthesis
- SSIM: Wang, Bovik et al. (2004) — Image Quality Assessment: From Error Visibility to Structural Similarity
