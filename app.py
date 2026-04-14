"""
DeepGuard AI — FastAPI Inference Server
========================================
Run:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

import io
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

from predict import DeepGuardPredictor
from utils import setup_logging

logger = setup_logging()

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_FILE_SIZE   = 10 * 1024 * 1024          # 10 MB
ALLOWED_TYPES   = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
MODEL_PATH      = "best_model.pth"

# ── Global state ──────────────────────────────────────────────────────────────
_predictor: DeepGuardPredictor | None = None
_startup_time: float = 0.0


# ── Lifespan (replaces deprecated on_event) ───────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _predictor, _startup_time
    t0 = time.perf_counter()
    try:
        logger.info("Loading DeepGuard model…")
        _predictor = DeepGuardPredictor(model_path=MODEL_PATH)
        _startup_time = time.perf_counter() - t0
        logger.info(f"Model ready in {_startup_time:.2f}s")
    except FileNotFoundError as e:
        logger.error(str(e))
        logger.warning("API will start but /predict will return 503 until a model is available.")
    except Exception as e:
        logger.error(f"Unexpected error loading model: {e}")

    yield  # ← app is running

    logger.info("Shutting down DeepGuard API…")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="DeepGuard AI",
    description=(
        "Deepfake face detection API powered by EfficientNet-B0. "
        "Upload a face image and receive a Real / Fake verdict with confidence scores."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# Allow all origins in development; restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _check_model() -> DeepGuardPredictor:
    if _predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Please train a model and restart the server.",
        )
    return _predictor


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health_endpoint():
    return {"status": "ok", "model_loaded": _predictor is not None}

@app.get("/")
def read_root():
    return {
        "service": "DeepGuard AI",
        "version": "2.0.0",
        "model_loaded": _predictor is not None,
        "startup_time_s": round(_startup_time, 3),
        "docs": "/docs",
    }


@app.post("/predict", tags=["Inference"])
async def predict(file: UploadFile = File(..., description="Face image (JPEG / PNG / WEBP)")):
    """
    Analyse an uploaded face image and return:
    - **label**: "Real" or "Fake"
    - **confidence**: probability of the predicted class (0–1)
    - **real_prob**: raw probability of Real
    - **fake_prob**: raw probability of Fake
    """
    predictor = _check_model()

    # ── Validate content-type ────────────────────────────────────────────
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{file.content_type}'. Accepted: {sorted(ALLOWED_TYPES)}",
        )

    # ── Read & size-check ────────────────────────────────────────────────
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large ({len(contents) / 1024 / 1024:.1f} MB). Max: {MAX_FILE_SIZE // 1024 // 1024} MB.",
        )

    # ── Decode image ─────────────────────────────────────────────────────
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not decode image. Ensure the file is a valid image.",
        )
    except Exception as e:
        logger.error(f"Image decode error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    # ── Run inference ────────────────────────────────────────────────────
    try:
        t0     = time.perf_counter()
        result = predictor.predict(image)
        result["inference_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
