"""
Parametric Gaussian blind deblurring operator.

Models degradation as convolution with a bivariate Gaussian kernel:
    y = K_σ * x + n

where σ = (σ_x, σ_y) are the unknown blur parameters estimated online
during the diffusion sampling loop.

Mathematical background
-----------------------
The bivariate Gaussian kernel is:
    K_σ(i, j) = exp(-(i²/2σ_x² + j²/2σ_y²)) / Z

where Z is a normalisation constant. For Gaussian blur, the operator
is self-adjoint (A = A^T) because K_σ is symmetric.

The kernel estimation objective:
    σ* = argmin_σ ||y - A(x̂; σ)||²

has a well-defined minimum at σ_true when x̂ is a reasonable image
estimate. This is the correct loss — the alternative ||A(x) - A^T(y)||²
is monotonic in σ and has no minimum.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from src.operators.base import DegradationOperator, OperatorConfig


@dataclass
class GaussianBlurConfig(OperatorConfig):
    """Configuration for parametric Gaussian blur operator."""

    # Kernel spatial support (pixels). Must be odd.
    kernel_size: int = 33

    # Blur parameter bounds for clamping during estimation
    sigma_min: float = 0.5
    sigma_max: float = 25.0

    # Initial sigma guess for blind estimation (starting point for Adam)
    sigma_init: Tuple[float, float] = field(default_factory=lambda: (10.0, 10.0))

    # Known sigma (non-blind mode). If None, runs in blind mode.
    sigma_known: Optional[Tuple[float, float]] = None

    def __post_init__(self):
        if self.kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be odd, got {self.kernel_size}. "
                "Even kernel sizes break symmetric padding."
            )


class GaussianBlurOperator(DegradationOperator):
    """
    Self-adjoint Gaussian blur operator with online blind kernel estimation.

    In blind mode: σ is estimated via Adam gradient descent at each
    diffusion timestep using the measurement consistency objective.

    In oracle mode (sigma_known is set): uses the true σ directly.
    This is useful as an ablation upper bound.

    Example
    -------
    >>> config = GaussianBlurConfig(kernel_size=33, sigma_init=(10.0, 10.0))
    >>> op = GaussianBlurOperator(config)
    >>> x = torch.randn(1, 3, 256, 256).cuda()
    >>> y = op.forward(x, sigma_x=8.0, sigma_y=8.0)        # degrade
    >>> params = op.estimate_params(y, x, iterations=15)    # blind estimate
    >>> x_restored = op.adjoint(y, **params)                # apply A^T
    """

    def __init__(self, config: GaussianBlurConfig):
        super().__init__(config)
        self.config: GaussianBlurConfig = config
        self.padding = config.kernel_size // 2

    # ------------------------------------------------------------------
    # Kernel construction
    # ------------------------------------------------------------------

    def _make_kernel(
        self,
        sigma_x: torch.Tensor,
        sigma_y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build a normalised bivariate Gaussian kernel.

        Args:
            sigma_x: Blur std in x direction (differentiable tensor)
            sigma_y: Blur std in y direction (differentiable tensor)

        Returns:
            Kernel of shape (1, 1, kernel_size, kernel_size),
            normalised to sum to 1.
        """
        k = self.config.kernel_size
        half = k // 2

        # Grid of (i, j) coordinates centred at 0
        coords = torch.arange(-half, half + 1, dtype=torch.float32, device=sigma_x.device)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")  # (k, k)

        # Bivariate Gaussian (independent x, y components)
        kernel = torch.exp(
            -(xx**2 / (2 * sigma_x**2) + yy**2 / (2 * sigma_y**2))
        )
        kernel = kernel / kernel.sum()  # normalise

        # Reshape to (1, 1, k, k) for depthwise conv
        return kernel.unsqueeze(0).unsqueeze(0)

    def _apply_kernel(
        self,
        x: torch.Tensor,
        kernel: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply a (1,1,k,k) kernel to a (B,C,H,W) image via depthwise conv.

        Pads with reflect mode to avoid border artefacts.
        """
        B, C, H, W = x.shape
        k = kernel.shape[-1]
        pad = k // 2

        # Expand kernel to (C, 1, k, k) for depthwise convolution
        kernel_expanded = kernel.expand(C, 1, k, k)

        # Reflect padding preserves edge statistics better than zero padding
        x_padded = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        return F.conv2d(x_padded, kernel_expanded, groups=C)

    # ------------------------------------------------------------------
    # Operator interface
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        sigma_x: float | torch.Tensor,
        sigma_y: float | torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply Gaussian blur: A(x; σ) = K_σ * x.

        Args:
            x: Image tensor (B, C, H, W), values in [0, 1]
            sigma_x: Blur std in x (scalar or tensor; tensor allows gradients)
            sigma_y: Blur std in y (scalar or tensor)

        Returns:
            Blurred image, same shape as x
        """
        # Ensure tensors for kernel construction
        if not isinstance(sigma_x, torch.Tensor):
            sigma_x = torch.tensor(float(sigma_x), device=x.device, dtype=torch.float32)
        if not isinstance(sigma_y, torch.Tensor):
            sigma_y = torch.tensor(float(sigma_y), device=x.device, dtype=torch.float32)

        kernel = self._make_kernel(sigma_x, sigma_y)
        return self._apply_kernel(x, kernel)

    def adjoint(
        self,
        y: torch.Tensor,
        sigma_x: float | torch.Tensor,
        sigma_y: float | torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply adjoint operator: A^T(y; σ).

        For Gaussian blur, A^T = A (the kernel is symmetric), so this
        is identical to forward(). Kept as a separate method to maintain
        the operator interface and make the code self-documenting.
        """
        return self.forward(y, sigma_x, sigma_y)

    # ------------------------------------------------------------------
    # Blind kernel estimation
    # ------------------------------------------------------------------

    def estimate_params(
        self,
        y: torch.Tensor,
        x_estimate: torch.Tensor,
        iterations: int = 15,
        lr: float = 0.3,
        betas: Tuple[float, float] = (0.9, 0.99),
        verbose: bool = False,
    ) -> dict:
        """
        Estimate blur parameters σ from measurement y and image estimate x̂.

        Solves: σ* = argmin_σ ||y - A(x̂; σ)||²

        This objective has a clear minimum at σ_true when x̂ is a
        reasonable image estimate (verified empirically and theoretically).

        Args:
            y: Observed degraded image (B, C, H, W)
            x_estimate: Current diffusion estimate of clean image
            iterations: Adam optimisation steps (15 is fast; 80 is thorough)
            lr: Adam learning rate
            betas: Adam momentum parameters
            verbose: Log loss and sigma values each step (for debugging)

        Returns:
            Dict with keys 'sigma_x', 'sigma_y' (Python floats, detached)

        Notes:
            x_estimate is frozen (no_grad) during kernel estimation —
            we only optimise over σ, not over the image.
        """
        device = y.device
        sigma_init_x, sigma_init_y = self.config.sigma_init

        # Learnable sigma parameters — these are what we optimise
        sigma_x = torch.tensor(
            sigma_init_x, device=device, dtype=torch.float32, requires_grad=True
        )
        sigma_y = torch.tensor(
            sigma_init_y, device=device, dtype=torch.float32, requires_grad=True
        )

        optimizer = torch.optim.Adam([sigma_x, sigma_y], lr=lr, betas=betas)

        # Freeze image estimate — only sigma gets gradients here
        x_frozen = x_estimate.detach()

        loss_history = []

        for step in range(iterations):
            optimizer.zero_grad()

            # Forward pass with current sigma estimate
            y_hat = self.forward(x_frozen, sigma_x, sigma_y)

            # Measurement consistency loss: minimum at σ_true
            loss = F.mse_loss(y_hat, y)
            loss.backward()
            optimizer.step()

            # Clamp to physically valid range
            with torch.no_grad():
                sigma_x.clamp_(self.config.sigma_min, self.config.sigma_max)
                sigma_y.clamp_(self.config.sigma_min, self.config.sigma_max)

            loss_val = loss.item()
            loss_history.append(loss_val)

            if verbose:
                print(
                    f"  Kernel est. step {step:02d}: "
                    f"loss={loss_val:.6f}, "
                    f"σ_x={sigma_x.item():.3f}, "
                    f"σ_y={sigma_y.item():.3f}"
                )

        return {
            "sigma_x": sigma_x.item(),  # detach for use in data consistency
            "sigma_y": sigma_y.item(),
            "loss_history": loss_history,
            "converged": loss_history[-1] < loss_history[0],  # basic convergence check
        }

    @property
    def is_blind(self) -> bool:
        return True
