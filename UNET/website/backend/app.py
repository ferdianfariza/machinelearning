from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from inference import FloodSegmenter


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "10"))
MAX_UPLOAD_BYTES = int(MAX_UPLOAD_MB * 1024 * 1024)

BASE_DIR = Path(__file__).resolve().parent
# DEFAULT_CHECKPOINT_PATH = (BASE_DIR.parent.parent / "model" / "best_overall_unet_effb0_4fold.pth").resolve()
DEFAULT_CHECKPOINT_PATH = (BASE_DIR.parent.parent / "model" / "best_overall_state_dict_unet_effb0_4fold.pth").resolve()
CHECKPOINT_PATH = Path(os.getenv("CHECKPOINT_PATH", str(DEFAULT_CHECKPOINT_PATH))).expanduser().resolve()
THRESHOLD_OVERRIDE = os.getenv("THRESHOLD")
THRESHOLD = float(THRESHOLD_OVERRIDE) if THRESHOLD_OVERRIDE is not None else None

app = FastAPI(title="Flood Segmentation API", version="0.0.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

segmenter: FloodSegmenter | None = None


@app.on_event("startup")
def _startup() -> None:
    global segmenter
    if not CHECKPOINT_PATH.exists():
        raise RuntimeError(f"Checkpoint not found at: {CHECKPOINT_PATH}")
    segmenter = FloodSegmenter(
        checkpoint_path=CHECKPOINT_PATH,
        threshold=THRESHOLD,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _validate_upload(file: UploadFile, payload: bytes) -> None:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Invalid file type. Use jpg/jpeg/png.")

    if file.content_type and file.content_type.lower() not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Invalid content type. Use image/jpeg or image/png.")

    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size is {MAX_UPLOAD_MB:.1f} MB.",
        )


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, float | int | str]:
    if segmenter is None:
        raise HTTPException(status_code=503, detail="Model not ready.")

    payload = await file.read()
    _validate_upload(file, payload)

    try:
        image = Image.open(BytesIO(payload))
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Could not decode image file.") from exc

    try:
        result = segmenter.predict(image)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    return {
        "mask_base64": result.mask_base64,
        "threshold": result.threshold,
        "inference_ms": result.inference_ms,
        "width": result.width,
        "height": result.height,
    }
