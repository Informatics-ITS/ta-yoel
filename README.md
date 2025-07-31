**Nama Mahasiswa**: Yoel Mountanus Sitorus  
**NRP**: 5025211078  
**Judul TA**: Presentasi Data untuk Steganografi Citra dalam Domain Spasial  
**Dosen Pembimbing**: Prof. Tohari Ahmad, S.Kom., M.IT., Ph.D.

---

## 📺 Demo Aplikasi  
Embed video demo di bawah ini (ganti `VIDEO_ID` dengan ID video YouTube Anda):  
[![Demo Aplikasi](https://github.com/user-attachments/assets/64eeb04b-351c-49b6-ada8-c0e6c5826322)](https://youtu.be/4UD09hlPTUE)  
*Klik gambar di atas untuk menonton demo*

---

*Konten selanjutnya hanya merupakan contoh awalan yang baik. Anda dapat berimprovisasi bila diperlukan.*

## 📋 Deskripsi
Implementasi algoritma steganografi citra yang menggunakan paradigma frequency-to-index embedding dengan teknik LSB (Least Significant Bit) dan Reed-Solomon error correction. Metode ini mentransformasi data rahasia menjadi representasi frekuensi kemunculan elemen dalam basis-b tertentu, kemudian menyembunyikannya menggunakan dynamic indexing.

## 🛠 Panduan Instalasi & Menjalankan Software  

### Prasyarat  
- Python 3.11.5
- Dataset SIPI (USC-SIPI Image Database)

### Library yang Digunakan
- **Numpy**: Komputasi array multidimensi dan operasi matematis
- **ReedSolo**: Implementasi Reed-Solomon error correction 
- **OpenCV**: Pengolahan citra dan konversi ke grayscale
- **Random**: Generator angka acak untuk dynamic indexing

### Langkah-langkah  
1. **Clone Repository**  
   ```bash
   git clone https://github.com/Informatics-ITS/ta-zemetia.git
   cd steganography-frequency-index
   ```

2. **Instalasi Dependensi**
   ```bash
   pip install numpy opencv-python reedsolo
   ```

3. **Persiapan Data**
   - Download dataset SIPI dari [USC-SIPI Database](https://sipi.usc.edu/database/)
   - Letakkan citra di folder `images/`
   - Siapkan data rahasia di folder `payload/`

4. **Generate Data Rahasia (Opsional)**
   ```bash
   python generate.py 1MB -o test_data.bin --seed 42
   ```

5. **Jalankan Algoritma**
   - Buka `ta_yoel_sc_bigdata.ipynb` dengan Jupyter Notebook
   - Jalankan cell secara berurutan untuk proses embedding dan extraction

---

## 🔧 Struktur Algoritma

### Embedding Process
1. **Transformasi Data**: Konversi data rahasia ke frekuensi kemunculan elemen basis-b
2. **Dynamic Indexing**: Enkripsi frekuensi menggunakan random seed mapping
3. **LSB Modification**: Modifikasi LSB pada indeks yang berkorespondensi
4. **Reed-Solomon**: Generate parity key untuk error correction

### Extraction Process  
1. **Reed-Solomon Decoding**: Pulihkan LSB original menggunakan parity key
2. **Index Detection**: Identifikasi lokasi modifikasi melalui perbandingan LSB
3. **Reverse Mapping**: Kembalikan frekuensi original dari nilai terenkripsi
4. **Data Reconstruction**: Rekonstruksi data rahasia dari histogram frekuensi

### Key Components
- **K1**: Tabel indeks pengurutan data rahasia
- **K2**: Random seed untuk dynamic indexing  
- **K3**: Tabel indeks pengurutan frekuensi terenkripsi
- **K4**: Reed-Solomon parity key

---

## 📊 Hasil Eksperimen

### Performa (Data 100KB, Basis-10)
| Citra | Resolusi | PSNR (dB) | SSIM |
|-------|----------|-----------|------|
| Airplane | 1024x1024 | 98.337 | 1.0000 |
| Peppers | 512x512 | 76.777 | 0.9999 |
| Aerial | 256x256 | 86.296 | 1.0000 |

### Karakteristik Utama
- **Size-Independent**: Performa konsisten terlepas ukuran data (1KB-100KB)
- **Basis Scalable**: Mendukung basis 2^16 hingga 2^20 
- **High Quality**: PSNR 76-99 dB, SSIM mendekati 1.0

---

## 📚 Dokumentasi Tambahan
<div style="background-color: white">
  🔄 Flowchart Algoritma Embedding
<img style="background-color: white" width="2538" height="1698" alt="Tugas Akhir TA-Flowchart Embedding" src="https://github.com/user-attachments/assets/ccea2195-5ac2-4879-a816-52c84c8c8c38" />
  🔄 Flowchart Algoritma Embedding
<img style="background-color: white" width="3759" height="1296" alt="Tugas Akhir TA-Flowchart Extraction" src="https://github.com/user-attachments/assets/32b798c6-dea5-4f08-9dff-282d619171fa" />
</div>

---

## 📈 Evaluasi Metrik

- **PSNR**: Peak Signal-to-Noise Ratio (kualitas citra)
- **SSIM**: Structural Similarity Index (kesamaan struktural)  
- **BIR**: Base to Index Ratio (efisiensi kapasitas)

---

## ✅ Validasi

Pastikan proyek memenuhi kriteria berikut sebelum submit:
- Source code dapat di-build/run tanpa error
- Video demo jelas menampilkan fitur utama
- README lengkap dan terupdate
- Tidak ada data sensitif (password, API key) yang ter-expose

---

## ⁉️ Pertanyaan?

Hubungi:
- Penulis: [yoelsit@gmail.com]
- Pembimbing Utama: [tohari@if.its.ac.id]
