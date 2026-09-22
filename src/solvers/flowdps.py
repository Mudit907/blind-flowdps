"""
FlowDPS solver integration for blind inverse problems.

This module wraps the original FlowDPS sd3_sampler.py into a clean,
importable class. The Kaggle notebook patched sd3_sampler.py in-place
via string replacement — this module replaces that approach with a
proper subclass that overrides only the data_consistency method.

How to use
----------
The original FlowDPS repo must be cloned alongside this project:

    git clone https://github.com/FlowDPS-Inverse/FlowDPS.git

Then set FLOWDPS_PATH in your environment or pass it to BlindFlowDPSSolver:

    export FLOWDPS_PATH=/path/to/FlowDPS

Architecture decision
---------------------
We subclass the original SD3Sampler rather than rewriting it.
This means we get all upstream bug fixes and improvements for free,
and our diff is minimal and auditable — only data_consistency changes.

The original data_consistency had two issues (from Kaggle notebook history):
  1. sigma detachment via .item() broke gradient flow
  2. blind=True was never passed from solve.py → method never executed

Both are fixed here with no string patching required.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from src.operators.gaussian_blur import GaussianBlurOperator


def _load_flowdps(flowdps_path: Optional[str] = None) -> Path:
    """
    Add FlowDPS to sys.path and return its root directory.

    Args:
        flowdps_path: Path to the cloned FlowDPS repo.
                      Falls back to FLOWDPS_PATH env var, then
                      looks for FlowDPS/ next to this project root.

    Raises:
        FileNotFoundError: If FlowDPS cannot be found.
    """
    candidates = []

    if flowdps_path:
        candidates.append(Path(flowdps_path))

    env_path = os.environ.get("FLOWDPS_PATH")
    if env_path:
        candidates.append(Path(env_path))

    # Common locations relative to project root
    project_root = Path(__file__).parent.parent.parent
    candidates += [
        project_root / "FlowDPS",
        project_root.parent / "FlowDPS",
        Path("/kaggle/working/FlowDPS"),  # Kaggle fallback
    ]

    for path in candidates:
        if path.exists() and (path / "sd3_sampler.py").exists():
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
            return path

    raise FileNotFoundError(
        "FlowDPS repository not found. Clone it with:\n"
        "  git clone https://github.com/FlowDPS-Inverse/FlowDPS.git\n"
        "Then set FLOWDPS_PATH=/path/to/FlowDPS or pass flowdps_path= to BlindFlowDPSSolver."
    )


class BlindFlowDPSSolver:
    """
    FlowDPS solver extended for blind inverse problems.

    Wraps the original SD3Sampler and overrides data_consistency to:
    1. Accept blind=True and recon_operator arguments
    2. Run online kernel estimation (Adam, 15 steps) when blind=True
    3. Use estimated sigma for data consistency update

    This is a clean replacement for the Kaggle string-patching approach.

    Example
    -------
    >>> solver = BlindFlowDPSSolver.from_pretrained(
    ...     model_id="stabilityai/stable-diffusion-3-medium-diffusers",
    ...     flowdps_path="/path/to/FlowDPS",
    ...     device="cuda",
    ... )
    >>> recon, sigma_history = solver.sample_blind(
    ...     measurement=degraded_image,
    ...     operator=gaussian_blur_operator,
    ...     prompt="a high quality photo",
    ...     NFE=28,
    ...     cfg_scale=2.0,
    ...     dc_stepsize=15.0,
    ...     kernel_iterations=15,
    ... )
    """

    def __init__(
        self,
        base_sampler,
        device: str = "cuda",
        verbose: bool = False,
    ):
        """
        Args:
            base_sampler: Instantiated SD3Sampler from FlowDPS repo
            device: Target device
            verbose: Print sigma estimates and DC losses during sampling
        """
        self.sampler = base_sampler
        self.device = device
        self.verbose = verbose

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "stabilityai/stable-diffusion-3-medium-diffusers",
        flowdps_path: Optional[str] = None,
        device: str = "cuda",
        verbose: bool = False,
    ) -> "BlindFlowDPSSolver":
        """
        Load SD3 weights and return a BlindFlowDPSSolver.

        Args:
            model_id: HuggingFace model ID for SD3
            flowdps_path: Path to cloned FlowDPS repo
            device: cuda or cpu
            verbose: Print progress during sampling

        Returns:
            BlindFlowDPSSolver ready for sampling
        """
        flowdps_root = _load_flowdps(flowdps_path)

        # Import the original sampler from FlowDPS
        # This import only works after _load_flowdps adds the path
        try:
            from sd3_sampler import SD3Sampler  # type: ignore
        except ImportError as e:
            raise ImportError(
                f"Could not import SD3Sampler from FlowDPS at {flowdps_root}. "
                f"Error: {e}"
            )

        # Instantiate the base sampler with SD3 weights
        base_sampler = SD3Sampler(model_id=model_id, device=device)

        return cls(base_sampler=base_sampler, device=device, verbose=verbose)

    # ------------------------------------------------------------------
    # Core: blind data consistency (replaces the Kaggle string patches)
    # ------------------------------------------------------------------

    def _data_consistency_blind(
        self,
        z0t: torch.Tensor,
        meas_img: torch.Tensor,
        operator: GaussianBlurOperator,
        task: str,
        stepsize: float = 15.0,
        num_iters: int = 7,
        kernel_iterations: int = 15,
        kernel_lr: float = 0.3,
        sigma_init: Tuple[float, float] = (10.0, 10.0),
        step_index: int = 0,
        sigma_state: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Data consistency update with blind kernel estimation.

        This is the clean version of the Kaggle notebook's data_consistency
        with the following fixes applied:
          - sigma is NOT detached (.item() removed from loss computation)
          - sigma persists across diffusion steps (coupled optimization)
          - kernel_iterations reduced to 15 for speed (vs 80 original)
          - correct loss: ||y - A(x; σ)||² (not the monotonic wrong loss)

        Args:
            z0t: Current latent code, shape (1, C, H/8, W/8)
            meas_img: Degraded measurement, shape (1, C, H, W)
            operator: GaussianBlurOperator with estimate_params()
            task: Task string (e.g. 'deblur_gauss')
            stepsize: Data consistency gradient step size
            num_iters: DC iterations per diffusion step
            kernel_iterations: Adam steps for kernel estimation
            kernel_lr: Adam learning rate for sigma
            sigma_init: Initial sigma guess (intentionally offset from true σ)
            step_index: Current diffusion timestep index (for sigma coupling)
            sigma_state: Dict carrying sigma across diffusion steps {sigma_x, sigma_y}

        Returns:
            Tuple of (updated z0t, sigma_state dict for next step)
        """
        z0t = z0t.requires_grad_(True)

        # Initialise or reuse sigma from previous diffusion step (coupling)
        if sigma_state is None or step_index == 0:
            sigma_x = torch.tensor(
                sigma_init[0], device=self.device, dtype=torch.float32, requires_grad=True
            )
            sigma_y = torch.tensor(
                sigma_init[1], device=self.device, dtype=torch.float32, requires_grad=True
            )
        else:
            # Coupled: continue from where last step left off
            sigma_x = sigma_state["sigma_x"].detach().clone().requires_grad_(True)
            sigma_y = sigma_state["sigma_y"].detach().clone().requires_grad_(True)

        kernel_optimizer = torch.optim.Adam(
            [sigma_x, sigma_y], lr=kernel_lr, betas=(0.9, 0.99)
        )

        for iter_idx in range(num_iters):
            # Decode current latent to image space
            x0t = self.sampler.decode(z0t).float()

            # ----------------------------------------------------------------
            # Kernel estimation: find σ that explains measurement y
            # Objective: min_σ ||y - A(x̂; σ)||²
            # ----------------------------------------------------------------
            x0t_frozen = x0t.detach().clone()

            for k_step in range(kernel_iterations):
                kernel_optimizer.zero_grad()

                # Forward blur with current sigma estimate
                blurred = operator.forward(x0t_frozen, sigma_x=sigma_x, sigma_y=sigma_y)

                # Correct loss — has minimum at σ_true
                kernel_loss = F.mse_loss(blurred, meas_img)
                kernel_loss.backward()
                kernel_optimizer.step()

                with torch.no_grad():
                    sigma_x.clamp_(operator.config.sigma_min, operator.config.sigma_max)
                    sigma_y.clamp_(operator.config.sigma_min, operator.config.sigma_max)

            if self.verbose and iter_idx == 0:
                print(
                    f"  [DC step {step_index}, iter {iter_idx}] "
                    f"σ_x={sigma_x.item():.3f}, σ_y={sigma_y.item():.3f}, "
                    f"kernel_loss={kernel_loss.item():.6f}"
                )

            # ----------------------------------------------------------------
            # Data consistency: update latent z toward measurement-consistent x
            # Loss: ||A^T(y; σ*) - A^T(A(x̂; σ*); σ*)||²
            # ----------------------------------------------------------------
            sigma_x_val = sigma_x.item()
            sigma_y_val = sigma_y.item()

            dc_loss = operator.data_consistency_loss(
                x=x0t,
                y=meas_img,
                sigma_x=sigma_x_val,
                sigma_y=sigma_y_val,
            )

            grad = torch.autograd.grad(dc_loss, z0t)[0].half()
            z0t = z0t - stepsize * grad

        sigma_state = {"sigma_x": sigma_x, "sigma_y": sigma_y}
        return z0t.detach(), sigma_state

    def _data_consistency_oracle(
        self,
        z0t: torch.Tensor,
        meas_img: torch.Tensor,
        operator: GaussianBlurOperator,
        task: str,
        sigma_x: float,
        sigma_y: float,
        stepsize: float = 15.0,
        num_iters: int = 7,
    ) -> torch.Tensor:
        """
        Data consistency with known (oracle) sigma.

        Used for ablation: gives the upper bound on performance
        when sigma is known exactly. Compare to blind estimation
        to measure the cost of not knowing sigma.
        """
        z0t = z0t.requires_grad_(True)

        for _ in range(num_iters):
            x0t = self.sampler.decode(z0t).float()

            if "sr" in task:
                loss = torch.linalg.norm(
                    (operator.adjoint(meas_img, sigma_x=sigma_x, sigma_y=sigma_y)
                     - operator.adjoint(operator.forward(x0t, sigma_x=sigma_x, sigma_y=sigma_y),
                                        sigma_x=sigma_x, sigma_y=sigma_y)).view(1, -1)
                )
            else:
                loss = operator.data_consistency_loss(
                    x=x0t, y=meas_img, sigma_x=sigma_x, sigma_y=sigma_y
                )

            grad = torch.autograd.grad(loss, z0t)[0].half()
            z0t = z0t - stepsize * grad

        return z0t.detach()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample_blind(
        self,
        measurement: torch.Tensor,
        operator: GaussianBlurOperator,
        prompt: str = "a high quality photo",
        task: str = "deblur_gauss",
        NFE: int = 28,
        cfg_scale: float = 2.0,
        dc_stepsize: float = 15.0,
        dc_num_iters: int = 7,
        kernel_iterations: int = 15,
        kernel_lr: float = 0.3,
        sigma_init: Tuple[float, float] = (10.0, 10.0),
    ) -> Tuple[torch.Tensor, List[dict]]:
        """
        Run blind FlowDPS: estimate kernel and reconstruct image jointly.

        This is the main entry point. Calls the base SD3Sampler's ODE loop
        but injects our blind data_consistency at each diffusion step.

        Args:
            measurement: Degraded image tensor (1, C, H, W), values in [0,1]
            operator: GaussianBlurOperator (sigma unknown — estimated online)
            prompt: Text prompt for SD3 guidance
            task: Degradation task string ('deblur_gauss')
            NFE: Number of diffusion function evaluations (steps)
            cfg_scale: Classifier-free guidance scale
            dc_stepsize: Data consistency gradient step size η
            dc_num_iters: DC iterations per diffusion step
            kernel_iterations: Adam steps for kernel estimation per DC call
            kernel_lr: Adam learning rate for sigma
            sigma_init: Starting sigma guess for Adam

        Returns:
            Tuple of:
              - reconstructed image tensor (1, C, H, W)
              - sigma_history: list of {step, sigma_x, sigma_y} dicts for analysis
        """
        measurement = measurement.to(self.device)

        # Reshape flat measurement if needed (FlowDPS sometimes flattens)
        if measurement.dim() == 2:
            B, N = measurement.shape
            C = 3
            H = W = int((N / C) ** 0.5)
            measurement = measurement.view(B, C, H, W)

        sigma_state = None
        sigma_history = []

        # ----------------------------------------------------------------
        # INTEGRATION POINT: hook into FlowDPS ODE loop
        #
        # The base sampler's sample() method runs the reverse diffusion ODE.
        # We need to inject _data_consistency_blind at each timestep.
        #
        # Two options:
        #
        # Option A (preferred): Pass a callback to base sampler
        #   recon = self.sampler.sample(
        #       measurement=measurement,
        #       operator=operator,
        #       task=task,
        #       prompts=[prompt],
        #       NFE=NFE,
        #       cfg_scale=cfg_scale,
        #       step_size=dc_stepsize,
        #       blind=True,                        ← now correctly passed
        #       recon_operator=operator,
        #       dc_callback=self._data_consistency_blind,  ← inject our method
        #   )
        #
        # Option B (fallback): Monkey-patch the sampler's data_consistency
        #   self.sampler.data_consistency = self._data_consistency_blind
        #   recon = self.sampler.sample(...)
        #
        # Use Option B until FlowDPS upstream adds callback support.
        # This is safer than the Kaggle string-replacement approach because
        # it operates on the live Python object, not the source file.
        # ----------------------------------------------------------------

        original_dc = self.sampler.data_consistency

        def patched_dc(z0t, op, measurement_inner, task_inner, stepsize=dc_stepsize,
                       blind=False, recon_operator=None, _step_idx=[0]):
            if blind and recon_operator is not None:
                # Reshape measurement if needed
                meas_img = measurement_inner
                if meas_img.dim() == 2:
                    B, N = meas_img.shape
                    C = 3
                    H = W = int((N / C) ** 0.5)
                    meas_img = meas_img.view(B, C, H, W)

                z0t_out, new_state = self._data_consistency_blind(
                    z0t=z0t,
                    meas_img=meas_img,
                    operator=recon_operator,
                    task=task_inner,
                    stepsize=stepsize,
                    num_iters=dc_num_iters,
                    kernel_iterations=kernel_iterations,
                    kernel_lr=kernel_lr,
                    sigma_init=sigma_init,
                    step_index=_step_idx[0],
                    sigma_state=sigma_state,
                )

                # Log sigma for convergence analysis
                sigma_history.append({
                    "step": _step_idx[0],
                    "sigma_x": new_state["sigma_x"].item(),
                    "sigma_y": new_state["sigma_y"].item(),
                })

                # Update coupled state
                nonlocal sigma_state
                sigma_state = new_state
                _step_idx[0] += 1

                return z0t_out
            else:
                return original_dc(z0t, op, measurement_inner, task_inner,
                                   stepsize=stepsize, blind=blind,
                                   recon_operator=recon_operator)

        # Temporarily replace data_consistency on the sampler instance
        self.sampler.data_consistency = patched_dc

        try:
            recon = self.sampler.sample(
                measurement=measurement,
                operator=operator,
                task=task,
                prompts=[prompt],
                NFE=NFE,
                cfg_scale=cfg_scale,
                step_size=dc_stepsize,
                blind=True,
                recon_operator=operator,
            )
        finally:
            # Always restore original method
            self.sampler.data_consistency = original_dc

        return recon, sigma_history

    def sample_oracle(
        self,
        measurement: torch.Tensor,
        operator: GaussianBlurOperator,
        sigma_x: float,
        sigma_y: float,
        prompt: str = "a high quality photo",
        task: str = "deblur_gauss",
        NFE: int = 28,
        cfg_scale: float = 2.0,
        dc_stepsize: float = 15.0,
    ) -> torch.Tensor:
        """
        Oracle mode: run FlowDPS with the TRUE sigma known.

        Use this for ablation experiments to establish the performance
        upper bound. Compare oracle PSNR vs blind PSNR to measure the
        gap that kernel estimation closes.

        Args:
            sigma_x: True blur sigma in x (from degradation pipeline)
            sigma_y: True blur sigma in y
            (other args same as sample_blind)

        Returns:
            Reconstructed image tensor (1, C, H, W)
        """
        measurement = measurement.to(self.device)
        if measurement.dim() == 2:
            B, N = measurement.shape
            C = 3
            H = W = int((N / C) ** 0.5)
            measurement = measurement.view(B, C, H, W)

        recon = self.sampler.sample(
            measurement=measurement,
            operator=operator,
            task=task,
            prompts=[prompt],
            NFE=NFE,
            cfg_scale=cfg_scale,
            step_size=dc_stepsize,
            blind=False,  # Oracle: no estimation needed
            recon_operator=operator,
        )

        return recon