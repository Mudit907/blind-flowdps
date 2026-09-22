"""
Abstract base class for degradation operators.

Every operator in this project implements this interface, which means
the solver, metrics, and evaluation pipeline are all operator-agnostic.
Adding a new degradation (motion blur, super-resolution, inpainting) is
just implementing this class — nothing else changes.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class OperatorConfig:
    """
    Base configuration for all operators.
    Subclasses extend this with operator-specific fields.
    """
    device: str = "cuda"
    channels: int = 3


class DegradationOperator(ABC, nn.Module):
    """
    Abstract base for all degradation operators A: X → Y.

    The operator models the forward degradation process:
        y = A(x; θ) + n
    where θ are the degradation parameters (known in non-blind setting,
    estimated in blind setting).

    Subclasses must implement:
        forward(x, **params)  — the forward degradation A(x; θ)
        adjoint(y, **params)  — the adjoint A^T(y; θ)

    Optional (for blind operators):
        estimate_params(y, x_estimate)  — online parameter estimation
    """

    def __init__(self, config: OperatorConfig):
        super().__init__()
        self.config = config
        self.device = config.device

    @abstractmethod
    def forward(self, x: torch.Tensor, **params) -> torch.Tensor:
        """
        Apply forward degradation: A(x; θ).

        Args:
            x: Clean image tensor, shape (B, C, H, W), values in [0, 1]
            **params: Operator-specific degradation parameters

        Returns:
            Degraded image tensor, same shape as x
        """
        ...

    @abstractmethod
    def adjoint(self, y: torch.Tensor, **params) -> torch.Tensor:
        """
        Apply adjoint operator: A^T(y; θ).

        For symmetric operators (e.g. Gaussian blur), A^T = A.
        For non-symmetric operators (e.g. super-resolution), must implement
        the true transpose.

        Args:
            y: Degraded image tensor, shape (B, C, H, W)
            **params: Same degradation parameters as forward()

        Returns:
            Tensor in image space, same shape as y
        """
        ...

    def data_consistency_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        **params,
    ) -> torch.Tensor:
        """
        Compute data consistency loss for the DC gradient step.

        Default formulation (works for most operators):
            L_dc = ||A^T(y; θ) - A^T(A(x; θ); θ)||²

        Override in subclasses if a different formulation is needed.

        Args:
            x: Current image estimate, shape (B, C, H, W)
            y: Observed degraded measurement, shape (B, C, H, W)
            **params: Degradation parameters

        Returns:
            Scalar loss tensor (differentiable w.r.t. x)
        """
        At_y = self.adjoint(y, **params)
        At_Ax = self.adjoint(self.forward(x, **params), **params)
        return torch.linalg.norm((At_y - At_Ax).view(1, -1))

    def estimate_params(
        self,
        y: torch.Tensor,
        x_estimate: torch.Tensor,
        **kwargs,
    ) -> dict:
        """
        Online parameter estimation (blind operators only).

        Default raises NotImplementedError — non-blind operators don't need this.
        Override in blind operator subclasses.

        Args:
            y: Observed degraded measurement
            x_estimate: Current image estimate from diffusion
            **kwargs: Estimation hyperparameters (lr, iterations, etc.)

        Returns:
            Dict of estimated parameters matching forward()'s **params signature
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement blind parameter estimation. "
            "If this is a blind operator, override estimate_params()."
        )

    @property
    def is_blind(self) -> bool:
        """Whether this operator supports blind parameter estimation."""
        try:
            self.estimate_params
            return True
        except AttributeError:
            return False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config})"
