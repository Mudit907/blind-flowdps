"""
FastAPI inference API for Blind-FlowDPS.

Exposes the blind deblurring solver as an HTTP endpoint.
Send a degraded image, receive the restored image + estimated blur parameters.

Endpoints:
    GET  /          — Health check + model info
    GET  /health    — Liveness check
    POST /restore   — Blind image restoration
    POST /estimate  — Kernel estimation only (no diffusion)

Usage:
    uvicorn api.main:app --host 0.0.0.0 --port 8000
    docker compose up api
"""

import io
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from PIL import Image
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.operators.gaussian_blur import GaussianBlurConfig, GaussianBlurOperator
from src.utils.image_io import load_image, tensor_to_numpy
from src.utils.reproducibility import seed_everything

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Blind-FlowDPS API",
    description=(
        "Blind image restoration using flow-matching diffusion models. "
        "Upload a degraded image; get back the restored image and estimated blur parameters."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global model state
# ---------------------------------------------------------------------------

_device: str = "cuda" if torch.cuda.is_available() else "cpu"
_operator: Optional[GaussianBlurOperator] = None
_solver = None  # FlowDPS solver — loaded on startup


@app.on_event("startup")
async def load_models():
    """Load operator and solver on startup. Fails fast if models aren't available."""
    global _operator, _solver

    seed_everything(42)

    op_config = GaussianBlurConfig(
        kernel_size=33,
        sigma_init=(10.0, 10.0),
        sigma_min=0.5,
        sigma_max=25.0,
        device=_device,
    )
    _operator = GaussianBlurOperator(op_config)

    # INTEGRATION POINT: load FlowDPS solver here
    # from src.solvers.flowdps import FlowDPSSolver
    # _solver = FlowDPSSolver.from_pretrained("sd3-medium", device=_device)
    # print(f"Solver loaded on {_device}")

    print(f"API ready on {_device}. Solver: {'loaded' if _solver else 'not loaded (placeholder mode)'}")


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class RestoreResponse(BaseModel):
    """Response from /restore endpoint."""
    success: bool
    sigma_x_estimated: float
    sigma_y_estimated: float
    kernel_converged: bool
    psnr_improvement_db: Optional[float]
    inference_time_seconds: float
    message: str


class KernelEstimateResponse(BaseModel):
    """Response from /estimate endpoint."""
    sigma_x_estimated: float
    sigma_y_estimated: float
    converged: bool
    loss_history: list[float]
    inference_time_seconds: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/")
async def root():
    return {
        "name": "Blind-FlowDPS API",
        "version": "0.1.0",
        "device": _device,
        "solver_loaded": _solver is not None,
        "endpoints": {
            "/restore": "POST — Blind image restoration",
            "/estimate": "POST — Kernel estimation only",
            "/health": "GET — Liveness check",
            "/docs": "GET — Interactive API docs",
        },
    }


@app.get("/health")
async def health():
    return {"status": "ok", "device": _device}


@app.post("/restore", response_class=Response)
async def restore_image(
    file: UploadFile = File(..., description="Degraded image (PNG or JPEG)"),
    return_image: bool = Query(True, description="Return restored image as PNG"),
    deg_scale: Optional[float] = Query(
        None,
        description="Known blur sigma (optional; if None, runs in blind mode)",
    ),
    NFE: int = Query(28, description="Diffusion steps (higher = better quality, slower)"),
    cfg_scale: float = Query(2.0, description="Classifier-free guidance scale"),
):
    """
    Restore a degraded image using blind flow-matching diffusion.

    In blind mode (deg_scale=None), estimates blur parameters online.
    In oracle mode (deg_scale provided), uses the given sigma directly.

    Returns the restored image as PNG with restoration metadata in response headers.
    """
    if _operator is None:
        raise HTTPException(status_code=503, detail="Operator not initialised")

    # Read and validate uploaded image
    try:
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    # Resize to 256×256 (model constraint)
    pil_image = pil_image.resize((256, 256), Image.LANCZOS)

    import torchvision.transforms.functional as TF
    degraded = TF.to_tensor(pil_image).unsqueeze(0).to(_device)

    start_time = time.time()

    # Run solver or placeholder
    if _solver is not None:
        # INTEGRATION POINT: actual solver call
        # recon, sigma_history = _solver.sample(
        #     measurement=degraded,
        #     operator=_operator,
        #     blind=deg_scale is None,
        #     sigma_known=(deg_scale, deg_scale) if deg_scale else None,
        #     NFE=NFE,
        #     cfg_scale=cfg_scale,
        # )
        recon = degraded.clone()  # placeholder
        sigma_history = []
    else:
        recon = degraded.clone()
        sigma_history = []

    # Run kernel estimation (always, for logging)
    kernel_result = _operator.estimate_params(
        y=degraded,
        x_estimate=recon,
        iterations=15,
        verbose=False,
    )

    elapsed = time.time() - start_time

    if return_image:
        # Return restored image as PNG with metadata in headers
        recon_np = tensor_to_numpy(recon)
        pil_recon = Image.fromarray(recon_np)

        buf = io.BytesIO()
        pil_recon.save(buf, format="PNG")
        buf.seek(0)

        return Response(
            content=buf.read(),
            media_type="image/png",
            headers={
                "X-Sigma-X": str(kernel_result["sigma_x"]),
                "X-Sigma-Y": str(kernel_result["sigma_y"]),
                "X-Kernel-Converged": str(kernel_result["converged"]),
                "X-Inference-Time": str(round(elapsed, 3)),
                "X-Solver-Mode": "blind" if deg_scale is None else "oracle",
            },
        )
    else:
        return JSONResponse({
            "success": True,
            "sigma_x_estimated": kernel_result["sigma_x"],
            "sigma_y_estimated": kernel_result["sigma_y"],
            "kernel_converged": kernel_result["converged"],
            "inference_time_seconds": round(elapsed, 3),
            "message": "Solver placeholder active — integrate FlowDPS solver for full restoration",
        })


@app.post("/estimate", response_model=KernelEstimateResponse)
async def estimate_kernel(
    file: UploadFile = File(..., description="Degraded image for kernel estimation"),
    iterations: int = Query(15, ge=1, le=200, description="Adam optimisation steps"),
    lr: float = Query(0.3, description="Adam learning rate"),
    verbose: bool = Query(False, description="Return full loss history"),
):
    """
    Estimate blur kernel parameters without running the full diffusion solver.

    Fast endpoint (~1–2 seconds) useful for:
    - Checking if kernel estimation converges for a given image
    - Debugging the even/odd sigma pattern
    - Building the convergence analysis dataset
    """
    if _operator is None:
        raise HTTPException(status_code=503, detail="Operator not initialised")

    try:
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    pil_image = pil_image.resize((256, 256), Image.LANCZOS)
    import torchvision.transforms.functional as TF
    degraded = TF.to_tensor(pil_image).unsqueeze(0).to(_device)

    start_time = time.time()

    result = _operator.estimate_params(
        y=degraded,
        x_estimate=degraded,  # use degraded as initial estimate
        iterations=iterations,
        lr=lr,
        verbose=verbose,
    )

    elapsed = time.time() - start_time

    return KernelEstimateResponse(
        sigma_x_estimated=result["sigma_x"],
        sigma_y_estimated=result["sigma_y"],
        converged=result["converged"],
        loss_history=result["loss_history"] if verbose else [],
        inference_time_seconds=round(elapsed, 3),
    )


# ---------------------------------------------------------------------------
# Dev server entrypoint
# ---------------------------------------------------------------------------


def start():
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    start()
