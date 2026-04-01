# Flood Segmentation Local MVP

This folder contains a runnable local MVP for single-image flood segmentation using your UNet4Fold checkpoint.

## Structure

- `backend/app.py`: FastAPI app and `/predict` endpoint.
- `backend/inference.py`: model build, checkpoint loading, preprocess, inference, postprocess.
- `backend/requirements.txt`: backend dependencies.
- `frontend/index.html`: single-page UI.
- `frontend/app.js`: upload + API call flow.
- `frontend/styles.css`: UI styling.

## Backend Run

From project root:

```powershell
cd UNET/website/backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Optional environment variables:

- `CHECKPOINT_PATH`: custom checkpoint location.
- `THRESHOLD`: override threshold (default from checkpoint config or 0.5).
- `MAX_UPLOAD_MB`: max upload size in MB (default 10).

## Frontend Run

In another terminal:

```powershell
cd UNET/website/frontend
python -m http.server 5500
```

Open:

- `http://127.0.0.1:5500`

## API Contract

`POST /predict` with `multipart/form-data` field `file` returns:

- `mask_base64` (binary PNG mask encoded to base64)
- `threshold`
- `inference_ms`
- `width`
- `height`
