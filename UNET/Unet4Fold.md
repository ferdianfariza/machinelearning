# Dokumentasi Notebook Flood Semantic Segmentation

Dokumen ini menjelaskan isi notebook [UNET++/UNET++.ipynb](UNET++/UNET++.ipynb) dengan bahasa ringkas dan terstruktur, supaya mudah dipakai saat diminta menjelaskan proyek.

## 1. Tujuan Proyek

Proyek ini membangun model semantic segmentation untuk mendeteksi area banjir pada citra.

Target spesifikasi utama:

- Input image dan mask di-resize ke 256x256.
- Normalisasi image ke rentang [0, 1].
- Mask dibinarisasi dengan threshold 0.5.
- Format mask menjadi (H, W, 1) sebelum jadi tensor.
- Data augmentation menggunakan Albumentations.
- Validasi menggunakan 4-Fold Cross-Validation.
- Model: UNet + EfficientNet-B0 (pretrained ImageNet).
- Training: 25 epoch, batch size 8, Adam (lr 1e-4), BCELoss.
- Evaluasi: Accuracy, Precision, Recall, F1, IoU.

## 2. Struktur Dataset yang Dipakai

Notebook ini disiapkan untuk struktur folder tanpa metadata.csv:

```text
dataset/
  train/
    images/
    labels/
  val/
    images/
    labels/
```

Catatan:

- Loader juga mencoba beberapa variasi nama folder umum (misalnya image/label, images/masks).
- Jika dataset dibungkus satu folder tambahan, notebook akan mencoba retry otomatis ke folder di dalamnya.

## 3. Alur Besar Pipeline

1. Cek GPU dan install dependency.
2. Set konfigurasi eksperimen.
3. Download/baca dataset dari Kaggle.
4. Deteksi pasangan folder image-mask.
5. Validasi file image-mask yang benar-benar bisa dibaca.
6. Bangun dataset + data loader dengan augmentasi.
7. Build model UNet EfficientNet-B0.
8. Training dan evaluasi 4-fold.
9. Simpan checkpoint terbaik tiap fold.
10. Ringkas metrik, plot kurva, dan tampilkan visual prediksi.
11. Cetak checklist implementasi spesifikasi.

## 4. Penjelasan Per Cell (Urutan Eksekusi)

Cell 1:

- Judul dan daftar spesifikasi eksperimen.

Cell 2:

- Menjalankan `nvidia-smi` untuk cek GPU.

Cell 3:

- Install dependency utama: PyTorch, segmentation-models-pytorch, Albumentations, OpenCV, sklearn, pandas, matplotlib, kagglehub.

Cell 4:

- Import library.
- Definisikan `CONFIG` (seed, image size, batch size, epochs, lr, n_splits, threshold, workers, dataset slug).
- Inisialisasi device (`cuda` atau `cpu`).

Cell 5:

- Utility fungsi data:
- Login Kaggle opsional.
- Deteksi path dataset dari Kaggle input.
- Fungsi baca image/mask (OpenCV + fallback PIL).
- Pencarian pasangan folder image-mask.
- Koleksi sampel valid dan statistik missing/decode-fail.

Cell 6:

- Eksekusi loading dataset:
- Login Kaggle.
- Download atau ambil dataset dari path lokal Kaggle.
- Kumpulkan `valid_samples` dari folder split.
- Jika tidak ditemukan, coba nested root otomatis.
- Menampilkan ringkasan jumlah data valid.

Cell 7:

- Definisi class `FloodDataset`.
- Proses resize, normalisasi, threshold mask, reshape mask.
- Definisi transform train/val dengan Albumentations.
- Definisi `make_loaders`.

Cell 8:

- Definisi model `smp.Unet` dengan encoder `efficientnet-b0`.
- Set loss `nn.BCELoss()`.
- Definisi fungsi hitung confusion dan metrik.

Cell 9:

- Definisi `train_one_epoch` dan `validate_one_epoch`.

Cell 10:

- Loop utama 4-fold CV.
- Training 25 epoch per fold.
- Simpan checkpoint terbaik berdasarkan IoU validasi.
- Bangun `history_df` dan `best_fold_df`.

Cell 11:

- Tampilkan ringkasan best score per fold.
- Hitung mean/std metrik.
- Plot kurva loss dan IoU.

Cell 12:

- Ambil fold terbaik.
- Load model dari checkpoint terbaik.

Cell 13:

- Visualisasi triplet: Input, Ground Truth, Predicted Mask.

Cell 14:

- Evaluasi ulang metrik pada validation set fold terbaik.

Cell 15:

- Cetak checklist naratif implementasi spesifikasi jurnal.

Cell 16:

- Validasi status implementasi secara boolean dan kesimpulan otomatis.

## 5. Konsep Penting (Saat Menjelaskan)

Normalisasi:

- Image dinormalisasi ke [0, 1] melalui `A.Normalize(..., max_pixel_value=255.0)`.

Binarisasi mask:

- Nilai mask diubah menjadi 0/1 dengan threshold 0.5.

Cross-validation:

- Data dibagi 4 fold.
- Tiap fold bergantian menjadi validation set.
- Hasil akhir dilihat dari ringkasan performa semua fold.

Pemilihan model terbaik:

- Checkpoint per fold disimpan saat IoU validasi meningkat.
- Fold terbaik dipilih dari IoU tertinggi.

## 6. Cara Menjelaskan Proyek (Template Singkat)

Gunakan alur berikut saat presentasi:

1. Masalah:

- "Saya mengerjakan segmentasi banjir untuk memisahkan area banjir vs non-banjir pada citra."

2. Data:

- "Dataset disusun ke train/val dengan folder images dan labels, lalu saya validasi agar hanya pasangan file yang benar yang dipakai."

3. Metode:

- "Saya memakai UNet dengan backbone EfficientNet-B0 pretrained, karena balance antara akurasi dan efisiensi."

4. Training:

- "Konfigurasi utama: 256x256, batch 8, 25 epoch, Adam lr 1e-4, BCELoss, dengan 4-fold cross-validation."

5. Evaluasi:

- "Saya laporkan Accuracy, Precision, Recall, F1, dan IoU, lalu pilih model terbaik berdasarkan IoU validasi."

6. Hasil Visual:

- "Saya tampilkan input, ground truth, dan prediksi untuk melihat kualitas segmentasi secara kualitatif."

## 7. Output Penting yang Perlu Diperhatikan

Saat menjalankan notebook, fokus pada:

- `Valid samples`: memastikan data terbaca dengan baik.
- `best_fold_df`: performa terbaik setiap fold.
- `summary_df`: mean dan std performa antar fold.
- Plot loss/IoU: untuk melihat stabilitas training.
- Visualisasi prediksi: mengecek kualitas mask secara visual.

## 8. Troubleshooting Cepat

Kasus: FileNotFoundError saat load data.

- Cek apakah struktur folder mengikuti format train/images, train/labels, val/images, val/labels.
- Lihat output `Candidate folder pairs`.
- Pastikan nama file image-mask cocok (stem sama).

Kasus: Banyak data ke-skip.

- Cek `Missing file/label skipped` dan `Decode fail skipped`.
- Verifikasi file rusak atau ekstensi tidak umum.

Kasus: OOM GPU.

- Turunkan `BATCH_SIZE` (misalnya 8 ke 4).

Kasus: Performa rendah.

- Tambah epoch.
- Coba tuning augmentasi.
- Coba loss kombinasi (misalnya BCE + Dice) jika ingin eksperimen lanjutan.

## 9. Ringkasan 1 Menit

- Proyek ini melakukan segmentasi banjir dengan UNet + EfficientNet-B0.
- Data diproses ke 256x256, image dinormalisasi, mask dibinarisasi.
- Training memakai 4-fold CV agar evaluasi lebih robust.
- Metrik utama mencakup Accuracy, Precision, Recall, F1, dan IoU.
- Hasil dievaluasi secara kuantitatif (tabel metrik) dan kualitatif (visualisasi mask).

## 10. Lokasi File

- Notebook utama: [UNET++/UNET++.ipynb](UNET++/UNET++.ipynb)
- Dokumentasi ini: [docs.md](docs.md)
