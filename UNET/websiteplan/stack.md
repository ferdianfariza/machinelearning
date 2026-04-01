# Local Stack Plan (v0.0.1)

## Objective

Jalankan MVP secara lokal untuk flow berikut:

1. Upload 1 gambar.
2. Jalankan inferensi model UNet4Fold.
3. Tampilkan hasil mask hitam-putih.

## Recommended Stack (Local First)

- Backend API: FastAPI + Uvicorn
- Model runtime: PyTorch + segmentation-models-pytorch
- Image processing: OpenCV + NumPy + Pillow
- Frontend: HTML + vanilla JavaScript (single page)
- Transport: REST (`multipart/form-data` upload)

## Why This Stack

- Cepat dipasang dan ringan untuk v0.0.1.
- Mudah dipisah ke deployment cloud/HF Space nanti.
- Reusable dengan notebook training yang sudah ada.

## Suggested Project Structure

```text
websiteplan/
  backend/
    app.py
    inference.py
    requirements.txt
    weights/
      best_fold_1.pth
  frontend/
    index.html
    app.js
    styles.css
  stack.md
  v0.0.1.md
  v0.0.2.md
```

## Backend Responsibilities

- Load checkpoint UNet4Fold saat startup.
- Endpoint `POST /predict` menerima 1 gambar.
- Preprocess image sesuai training (resize + normalize).
- Inference + threshold 0.5.
- Return mask binary (0/255) sebagai base64 PNG.

## Frontend Responsibilities

- Input file tunggal (`jpg/jpeg/png`).
- Tombol `Run Inference`.
- Tampilkan 2 panel:
  - Original image
  - Predicted mask (black-white)
- Tampilkan pesan error jika request gagal.

## Local Development Commands

```powershell
# Backend
cd UNET/websiteplan/backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

```powershell
# Frontend (opsi paling sederhana)
cd UNET/websiteplan/frontend
python -m http.server 5500
```

## Minimal API Contract

### POST `/predict`

- Request:
  - `file`: image file
- Response JSON:
  - `mask_base64`: string
  - `threshold`: number
  - `inference_ms`: number
  - `width`: number
  - `height`: number

## Runtime Notes

- Device default: `cuda` jika tersedia, fallback ke `cpu`.
- Untuk test awal, resize tetap ke 256x256 agar konsisten dengan training config.
- Simpan threshold sebagai configurable variable di backend.

## Exit Criteria (Local Ready)

- Bisa upload 1 gambar dan dapat mask black-white.
- Output mask hanya bernilai 0 dan 255.
- Error handling jelas saat file invalid atau checkpoint tidak ditemukan.
