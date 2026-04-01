# Dokumentasi Teknis Notebook Flood Semantic Segmentation

Dokumen ini menjelaskan implementasi pada notebook [UNET/UNet4Fold.ipynb](UNET/UNet4Fold.ipynb) dengan fokus teknis agar mudah dipertanggungjawabkan saat presentasi metodologi.

## 1. Ruang Lingkup dan Tujuan

Tujuan utama proyek adalah semantic segmentation biner untuk memisahkan piksel banjir (`1`) dan non-banjir (`0`) dari citra input.

Spesifikasi eksperimen yang diimplementasikan:

- Ukuran input: 256x256.
- Mask biner dengan threshold 0.5.
- Model: `UNet(encoder=efficientnet-b0, pretrained=imagenet)`.
- Optimizer: Adam, learning rate `1e-4`.
- Loss: `BCELoss`.
- Validasi: 4-Fold Cross Validation.
- Metrik: Accuracy, Precision, Recall, F1-Score, IoU.

## 2. Konfigurasi Eksperimen

Konfigurasi dipusatkan pada dictionary `CONFIG` agar parameter training konsisten di semua cell:

- `SEED = 42`
- `IMAGE_SIZE = 256`
- `BATCH_SIZE = 8`
- `EPOCHS = 25`
- `LR = 1e-4`
- `N_SPLITS = 4`
- `THRESHOLD = 0.5`
- `DATASET_SLUG = "lihuayang111265/flood-semantic-segmentation-dataset"`

Device dipilih otomatis:

- GPU jika `torch.cuda.is_available()` bernilai true.
- CPU jika GPU tidak tersedia.

## 3. Spesifikasi Dataset dan Validasi Input

Notebook menggunakan dataset Kaggle dengan struktur direktori:

```text
dataset/
  train/
    images/
    labels/
  val/
    images/
    labels/
```

### 3.1 Discovery path dataset

Pipeline data melakukan:

1. Unduh dataset via `kagglehub.dataset_download`.
2. Cari base folder `dataset/` melalui fungsi `find_dataset_base_dir`.
3. Ambil pasangan split melalui `collect_pairs_from_split(base_dir, split_name)`.

### 3.2 Aturan pairing image-mask

Untuk setiap file di `images/`, pasangan mask dicari di `labels/` dengan nama file yang sama.

Contoh:

- `train/images/0001.jpg` dipasangkan dengan `train/labels/0001.jpg`.

File yang tidak punya pasangan label akan di-skip dan dihitung ke statistik `missing`.

### 3.3 Robust read dan quality gate

Pembacaan file menggunakan dua tahap:

1. OpenCV (`cv2.imread`).
2. Fallback PIL jika OpenCV gagal decode.

Jika image atau mask tidak bisa dibaca, sampel di-skip (`decode_fail`).

Hanya pasangan valid yang masuk ke `valid_samples`.

## 4. Data Pipeline di Dataset Class

Kelas `FloodDataset` menerapkan urutan transformasi berikut untuk setiap sampel:

1. Read image RGB dan mask grayscale.
2. Resize image dan mask ke 256x256.
3. Konversi mask ke float (`mask / 255.0`).
4. Binarisasi mask: `mask > 0.5`.
5. Ubah shape mask menjadi `(H, W, 1)`.
6. Terapkan augmentasi/normalisasi via Albumentations.
7. Konversi ke tensor dengan format channel-first `(C, H, W)`.

Output `__getitem__`:

- `image`: `torch.float32`, shape `(3, 256, 256)`.
- `mask`: `torch.float32`, shape `(1, 256, 256)`.

## 5. Augmentasi dan Normalisasi

### 5.1 Train transform

- `Resize(256,256)`
- `HorizontalFlip(p=0.5)`
- `Affine(scale, translation, rotation)`
- `RandomBrightnessContrast(p=0.5)`
- `Normalize(max_pixel_value=255.0)`
- `ToTensorV2(transpose_mask=True)`

### 5.2 Validation transform

- `Resize(256,256)`
- `Normalize(max_pixel_value=255.0)`
- `ToTensorV2(transpose_mask=True)`

Normalisasi memastikan nilai piksel input berada pada skala `[0,1]` saat masuk model.

## 6. Arsitektur Model dan Objective Function

Model dibangun dengan `segmentation_models_pytorch`:

- Arsitektur: U-Net.
- Encoder: EfficientNet-B0 pretrained ImageNet.
- Input channels: 3.
- Output classes: 1.
- Aktivasi akhir: sigmoid.

Loss function:

- `nn.BCELoss()` digunakan karena target segmentasi bersifat biner.

## 7. Training Loop dan Validasi

### 7.1 Prosedur per epoch

Fungsi `train_one_epoch`:

1. `model.train()`.
2. Forward pass.
3. Hitung `BCELoss`.
4. Backpropagation.
5. `optimizer.step()`.

Fungsi `validate_one_epoch`:

1. `model.eval()` dengan `torch.no_grad()`.
2. Hitung loss validasi.
3. Binarisasi prediksi dengan threshold 0.5.
4. Agregasi confusion components (`tp, tn, fp, fn`).
5. Hitung metrik akhir dari agregat global.

### 7.2 Rumus metrik

Metrik dihitung dari total `tp, tn, fp, fn`:

- Accuracy = $(tp + tn) / (tp + tn + fp + fn)$
- Precision = $tp / (tp + fp)$
- Recall = $tp / (tp + fn)$
- F1 = $2PR/(P+R)$
- IoU = $tp / (tp + fp + fn)$

Semua metrik memakai epsilon kecil untuk menghindari pembagian nol.

## 8. Skema 4-Fold Cross Validation

Skema CV memakai `KFold(n_splits=4, shuffle=True, random_state=SEED)`.

Alur tiap fold:

1. Bangun `train_loader` dan `val_loader` dari indeks fold.
2. Inisialisasi model baru dan optimizer baru.
3. Train 25 epoch.
4. Simpan checkpoint saat IoU validasi meningkat.
5. Simpan history epoch-level ke dataframe.

Output utama:

- `history_df`: seluruh riwayat train/val semua fold.
- `best_fold_df`: ringkasan performa terbaik per fold.

## 9. Checkpointing dan Pemilihan Model Terbaik

Setiap fold menyimpan file checkpoint:

- Path: `checkpoints_unet_effb0_4fold/best_fold_{k}.pth`

Isi checkpoint mencakup:

- `model_state_dict`
- `fold`, `epoch`
- `val_metrics`
- salinan `config`

Fold terbaik global dipilih dari nilai IoU tertinggi pada `best_fold_df`.

## 10. Evaluasi Akhir dan Visualisasi

Notebook menyediakan dua bentuk evaluasi:

1. Kuantitatif:

- tabel best per fold,
- ringkasan mean/std antar fold,
- evaluasi ulang pada fold terbaik.

2. Kualitatif:

- visualisasi triplet: Input, Ground Truth, Predicted Mask.

Visualisasi dipakai untuk memverifikasi bahwa segmentasi tidak hanya bagus secara angka, tetapi juga masuk akal secara spasial.

## 11. Kontrak Input-Output untuk Deployment API

Bagian inferensi notebook sudah dapat dijadikan fondasi API (misalnya FastAPI/Gradio):

- Input API: 1 citra RGB.
- Preprocess: resize 256x256 + normalisasi.
- Inferensi: model sigmoid output probabilitas piksel.
- Postprocess: threshold 0.5 menjadi mask biner.
- Output API: mask biner dan/atau overlay visual.

Catatan penting:

- Preprocessing di API harus identik dengan preprocessing training agar distribusi input konsisten.

## 12. Troubleshooting Teknis

Kasus: Tidak ada sampel valid.

- Pastikan struktur folder sesuai `dataset/train|val/images|labels`.
- Pastikan nama file image dan label cocok.
- Periksa output `missing` dan `decode_fail`.

Kasus: OOM GPU.

- Turunkan `BATCH_SIZE` (contoh: 8 ke 4).
- Aktifkan mixed precision jika ingin optimasi lanjutan.

Kasus: IoU stagnan.

- Coba kombinasi loss (BCE + Dice).
- Tuning augmentasi geometrik.
- Tambah epoch dan scheduler learning rate.

## 13. Lokasi File

- Notebook utama: [UNET/UNet4Fold.ipynb](UNET/UNet4Fold.ipynb)
- Dokumentasi teknis: [UNET/Unet4Fold.md](UNET/Unet4Fold.md)
