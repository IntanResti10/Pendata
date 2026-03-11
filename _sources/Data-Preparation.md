# PERTEMUAN 3 — DATA PREPARATION (EXTENSI)
## Studi Kasus: Iris + Data Campuran (Mixed-Type)

Dokumen ini melanjutkan materi Data Preparation (CRISP-DM) yang dibuat sebelumnya, lalu **ditambah**:
1) Identifikasi missing value (lebih lengkap)
2) Statistik deskriptif per **kelas**, **kolom**, dan **fitur**
3) Cara tarik data dari **MySQL/PostgreSQL** ke **Orange**
4) Penjelasan **fitur vs kelas** (contoh Iris)
5) Cara mengukur jarak (distance) & **distance matrix** di Orange
6) Imputasi missing value menggunakan **KNN** pada data numerik (Iris)
7) Imputasi missing value menggunakan **KNN** pada data campuran (DataCampuran)
8) Perhitungan manual langkah demi langkah + verifikasi Python
9) Cara menghitung jarak total menggunakan **Excel** (rumus lengkap)

## 1) Persiapan Lingkungan

Kita pakai Python untuk contoh analisis (missing value, statistik deskriptif, distance).

```python
%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
```

## 2) Memuat Dataset Iris

```python
df = pd.read_csv("IRIS.csv")
df.head()
```

## 3) Penjelasan: Fitur vs Kelas (Target)

Pada dataset Iris:
- **Fitur (features)** = kolom input/karakteristik bunga:
  - `sepal_length`, `sepal_width`, `petal_length`, `petal_width`
- **Kelas (class/label/target)** = kategori yang ingin diprediksi:
  - `species` (misal: *Iris-setosa*, *Iris-versicolor*, *Iris-virginica*)

✅ Jadi **sepal/petal itu fitur**, sedangkan **virginica/versicolor/setosa itu kelas**.

> Kalau kamu membuat kolom `species_encoded`, itu hanya versi numerik dari kelas.

## 4) Identifikasi Missing Value (Lebih Lengkap)

Selain `isnull().sum()`, kita juga cek:
- Persentase missing per kolom
- Baris mana yang punya missing

### 4.1 Jumlah missing per kolom

```python
missing_count = df.isnull().sum()
missing_count
```

### 4.2 Persentase missing per kolom

```python
missing_percent = (df.isnull().mean() * 100).round(2)
pd.DataFrame({'missing_count': missing_count, 'missing_%': missing_percent})
```

### 4.3 Baris yang memiliki missing (jika ada)

```python
rows_with_missing = df[df.isnull().any(axis=1)]
rows_with_missing.head()
```

## 5) Statistik Deskriptif

Target tugas: statistik deskriptif untuk:
1) **Setiap kolom/fitur** (overall)
2) **Setiap kelas** (group by `species`)
3) Ringkasan untuk data kategorikal (frekuensi kelas)

### 5.1 Statistik deskriptif overall (fitur numerik)

```python
numeric_cols = ['sepal_length','sepal_width','petal_length','petal_width']
df[numeric_cols].describe().T
```

### 5.2 Frekuensi tiap kelas

```python
df['species'].value_counts()
```

### 5.3 Statistik deskriptif per kelas (lengkap)

```python
df.groupby('species')[numeric_cols].describe()
```

### 5.4 Statistik per kelas (ringkas: mean, std, min, max)

```python
df.groupby('species')[numeric_cols].agg(['mean','std','min','max']).round(3)
```

## 6) Cara Menarik Data dari Database (MySQL/PostgreSQL) ke Orange

Orange bisa ambil data langsung dari DB melalui widget **SQL Table** (atau add-on yang mendukung koneksi DB).

### 6.1 Langkah Umum (Workflow Orange)

1. Buka **Orange**
2. Tambahkan widget: **SQL Table**
3. Pilih tipe DB: **MySQL** atau **PostgreSQL**
4. Isi koneksi: host, port, database, user, password
5. Pilih tabel atau tulis query SQL
6. Sambungkan ke widget lain: **Data Table**, **Select Columns**, **Impute**, **Normalize**, dll

### 6.2 Contoh Parameter Koneksi

**MySQL**
- Host: `localhost`
- Port: `3306`
- Database: `nama_db`
- User: `root` (atau user lain)

**PostgreSQL**
- Host: `localhost`
- Port: `5432`
- Database: `nama_db`
- User: `postgres` (atau user lain)

### 6.3 Contoh Query

```sql
SELECT sepal_length, sepal_width, petal_length, petal_width, species
FROM iris
WHERE sepal_length IS NOT NULL;
```

> Kalau widget **SQL Table** belum ada: buka **Options → Add-ons**, lalu install add-on yang mendukung database/SQL.

## 7) Cara Mengukur Jarak untuk Data Iris

Karena fitur Iris numerik, jarak yang umum dipakai:
- **Euclidean** (umum)
- **Manhattan**
- **Minkowski**

⚠️ Penting: lakukan **scaling** sebelum menghitung jarak.

### 7.1 Scaling

```python
X = df[numeric_cols].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled[:5]
```

### 7.2 Distance Matrix (Euclidean)

```python
D_euclid = pairwise_distances(X_scaled, metric='euclidean')
D_euclid[:5, :5]
```

## 8) Distance Matrix di Orange (Workflow)

1. **File** (atau **SQL Table**) → load data
2. **Select Columns** → pastikan fitur numerik masuk ke **Attributes**, `species` jadi **Class**
3. (Opsional) **Normalize** → Standardize/Normalize
4. **Distances** → pilih metric (Euclidean/Manhattan/Cosine)
5. Output ke **Distance Matrix** / **Heat Map** / **Hierarchical Clustering**

## 9) Checklist Output Pertemuan 3 (Untuk Laporan)

1) Bukti cek missing value (count + persen)
2) Statistik deskriptif: overall dan per kelas
3) Penjelasan fitur vs kelas (Iris)
4) Langkah tarik data DB ke Orange (MySQL/PostgreSQL)
5) Cara hitung jarak Iris + scaling
6) Distance matrix di Orange
7) Imputasi KNN pada data numerik (Iris) — perhitungan manual + kode
8) Imputasi KNN pada data campuran (DataCampuran) — perhitungan manual + kode

## 10) Konsep KNN untuk Mengisi Missing Value

### 10.1 Apa itu KNN Imputation?

KNN Imputation adalah metode untuk mengisi nilai yang hilang (missing value) dengan cara:

1. Mengukur **jarak** dari baris yang memiliki missing value ke semua baris lain
2. Memilih **k baris terdekat** (tetangga terdekat)
3. Mengisi missing value berdasarkan nilai dari tetangga terdekat tersebut

### 10.2 Aturan Pengisian

| Tipe Nilai yang Hilang | Cara Pengisian | Metode |
|---|---|---|
| **Numerik** | Rata-rata dari k tetangga terdekat | KNN Regression |
| **Kategorikal** | Modus (voting mayoritas) dari k tetangga | KNN Classification |

Pada pertemuan ini digunakan **KNN Regression** karena nilai yang diisi berupa angka.

### 10.3 Catatan Penting

- Jika satu kolom memiliki missing value, maka **kolom tersebut tidak diikutkan** dalam perhitungan jarak
- Contoh: jika kolom ke-4 kosong, maka jarak hanya dihitung dari kolom 1, 2, dan 3
- Setelah jarak dihitung, diambil **k data terdekat** (pada contoh ini k = 3)

## 11) Aturan Menghitung Jarak Berdasarkan Tipe Data

### 11.1 Jarak untuk Data Numerik — Euclidean Distance

Untuk data numerik digunakan **Euclidean Distance**:

$$d(i,j) = \sqrt{(x_1 - y_1)^2 + (x_2 - y_2)^2 + \cdots + (x_p - y_p)^2}$$

**Aturan:** Jika ada kolom yang missing, kolom tersebut **dilewati** dan tidak masuk perhitungan.

Contoh: Baris A = `[5.4, 3.9, 1.7, ?]` dan Baris B = `[5.1, 3.5, 1.4, 0.2]`

Karena kolom ke-4 kosong, jarak hanya dihitung dari 3 kolom pertama:

$$d = \sqrt{(5.4-5.1)^2 + (3.9-3.5)^2 + (1.7-1.4)^2} = \sqrt{0.09 + 0.16 + 0.09} = \sqrt{0.34} = 0.5831$$

### 11.2 Konversi Data Ordinal ke Numerik

Data ordinal memiliki urutan (ranking), tetapi tidak bisa langsung dimasukkan ke Euclidean. Harus diubah dulu menjadi numerik menggunakan rumus:

$$z = \frac{r - 1}{m - 1}$$

Keterangan:

- $r$ = urutan level ordinal (dimulai dari 1)
- $m$ = jumlah seluruh level

**Contoh:** Kolom `education` memiliki 3 level:

| Level | Urutan ($r$) | Hasil Konversi |
|---|---|---|
| primary | 1 | $(1-1)/(3-1) = 0$ |
| secondary | 2 | $(2-1)/(3-1) = 0.5$ |
| tertiary | 3 | $(3-1)/(3-1) = 1$ |

Setelah dikonversi, kolom ordinal **bisa digabungkan** dengan data numerik lain dan dihitung menggunakan Euclidean distance.

### 11.3 Jarak untuk Data Kategorikal

Untuk data kategorikal (nominal) digunakan rumus **ketidaksamaan**:

$$d_{kat} = \frac{P - M}{P}$$

Keterangan:

- $P$ = banyaknya kolom kategorikal yang dibandingkan
- $M$ = jumlah kolom yang **nilainya sama**

**Contoh:** Dibandingkan 2 kolom kategorikal (`Status Mahasiswa`, `Jenis Kelamin`):

| Kolom | Baris A | Baris B | Sama? |
|---|---|---|---|
| Status Mahasiswa | Ya | Ya | ✓ |
| Jenis Kelamin | P | L | ✗ |

Maka $P = 2$, $M = 1$ (satu kolom sama):

$$d_{kat} = \frac{2 - 1}{2} = \frac{1}{2} = 0.5$$

### 11.4 Jarak Total untuk Data Campuran

Untuk data campuran, langkah-langkahnya adalah:

1. **Hitung jarak numerik** → normalisasi Min-Max, lalu hitung Euclidean
2. **Hitung jarak kategorikal** → gunakan rumus $(P-M)/P$
3. **Jumlahkan keduanya** sebagai jarak total

$$d_{total} = d_{numerik} + d_{kat}$$

Inilah inti dari perhitungan jarak pada data campuran: **numerik dihitung Euclidean (setelah normalisasi), kategorikal dihitung sendiri → lalu keduanya dijumlahkan**.

## 12) Penerapan KNN Imputation pada Dataset Iris (Data Numerik)

Dataset Iris seluruhnya numerik (kecuali kolom `species`), sehingga perhitungan jarak menggunakan **Euclidean distance** saja.

### 12.1 Cek Missing Value Awal

Dataset Iris asli **tidak memiliki missing value**, sehingga perlu dibuat missing value buatan untuk simulasi.

### 12.2 Membuat Missing Value Buatan

Nilai yang dihilangkan:

- **Baris indeks 5** (baris ke-6)
- **Kolom `petal_width`**

Data asli baris 5:

| sepal_length | sepal_width | petal_length | petal_width | species |
|---|---|---|---|---|
| 5.4 | 3.9 | 1.7 | **0.4** | Iris-setosa |

Setelah dihilangkan:

| sepal_length | sepal_width | petal_length | petal_width | species |
|---|---|---|---|---|
| 5.4 | 3.9 | 1.7 | **?** | Iris-setosa |

Karena `petal_width` kosong, maka jarak dihitung hanya dari 3 kolom: `sepal_length`, `sepal_width`, `petal_length`.

### 12.3 Perhitungan Jarak Manual (Euclidean)

Berikut perhitungan jarak dari **baris 5** ke beberapa baris lain:

#### Baris 5 vs Baris 0

Baris 0: `[5.1, 3.5, 1.4]`

$$d(5,0) = \sqrt{(5.4-5.1)^2 + (3.9-3.5)^2 + (1.7-1.4)^2}$$

$$= \sqrt{0.3^2 + 0.4^2 + 0.3^2} = \sqrt{0.09 + 0.16 + 0.09} = \sqrt{0.34} = 0.5831$$

#### Baris 5 vs Baris 1

Baris 1: `[4.9, 3.0, 1.4]`

$$d(5,1) = \sqrt{(5.4-4.9)^2 + (3.9-3.0)^2 + (1.7-1.4)^2}$$

$$= \sqrt{0.5^2 + 0.9^2 + 0.3^2} = \sqrt{0.25 + 0.81 + 0.09} = \sqrt{1.15} = 1.0724$$

#### Baris 5 vs Baris 2

Baris 2: `[4.7, 3.2, 1.3]`

$$d(5,2) = \sqrt{(5.4-4.7)^2 + (3.9-3.2)^2 + (1.7-1.3)^2}$$

$$= \sqrt{0.7^2 + 0.7^2 + 0.4^2} = \sqrt{0.49 + 0.49 + 0.16} = \sqrt{1.14} = 1.0677$$

#### Baris 5 vs Baris 3

Baris 3: `[4.6, 3.1, 1.5]`

$$d(5,3) = \sqrt{(5.4-4.6)^2 + (3.9-3.1)^2 + (1.7-1.5)^2}$$

$$= \sqrt{0.8^2 + 0.8^2 + 0.2^2} = \sqrt{0.64 + 0.64 + 0.04} = \sqrt{1.32} = 1.1489$$

#### Baris 5 vs Baris 4

Baris 4: `[5.0, 3.6, 1.4]`

$$d(5,4) = \sqrt{(5.4-5.0)^2 + (3.9-3.6)^2 + (1.7-1.4)^2}$$

$$= \sqrt{0.4^2 + 0.3^2 + 0.3^2} = \sqrt{0.16 + 0.09 + 0.09} = \sqrt{0.34} = 0.5831$$

Perhitungan yang sama dilakukan untuk **seluruh 149 baris lainnya**.

### 12.4 Menentukan 3 Tetangga Terdekat (k = 3)

Setelah seluruh jarak dihitung dan diurutkan dari terkecil, diperoleh **3 tetangga terdekat**:

| Peringkat | Baris | Jarak | Data (SL, SW, PL) | petal_width |
|---|---|---|---|---|
| 1 | Baris 10 | 0.2828 | [5.4, 3.7, 1.5] | 0.2 |
| 2 | Baris 48 | 0.3000 | [5.3, 3.7, 1.5] | 0.2 |
| 3 | Baris 18 | 0.3162 | [5.7, 3.8, 1.7] | 0.3 |

**Verifikasi manual baris 5 vs baris 10:**

Baris 10: `[5.4, 3.7, 1.5]`

$$d(5,10) = \sqrt{(5.4-5.4)^2 + (3.9-3.7)^2 + (1.7-1.5)^2} = \sqrt{0 + 0.04 + 0.04} = \sqrt{0.08} = 0.2828$$

### 12.5 Imputasi Missing Value (KNN Regression)

Karena nilai yang diisi berupa numerik, digunakan **rata-rata** dari `petal_width` ketiga tetangga:

$$\hat{x} = \frac{0.2 + 0.2 + 0.3}{3} = \frac{0.7}{3} = 0.2333$$

**Hasil:** Missing value `petal_width` pada baris 5 diisi dengan **0.2333**

Nilai asli sebelum dihilangkan: **0.4**

### 12.6 Kode Python — Imputasi KNN pada Iris

```python
import pandas as pd
import numpy as np

# Load data
iris = pd.read_csv("IRIS.csv")

# Simpan nilai asli sebelum dihilangkan
target_idx = 5
col_missing = 'petal_width'
nilai_asli = iris.loc[target_idx, col_missing]

# Buat missing value buatan
iris_knn = iris.copy()
iris_knn.loc[target_idx, col_missing] = np.nan

# Fitur yang dipakai untuk menghitung jarak (tanpa kolom yang missing)
fitur = ['sepal_length', 'sepal_width', 'petal_length']

# Hitung jarak ke semua baris lain
hasil_jarak = []
for i in range(len(iris_knn)):
    if i == target_idx:
        continue
    d = np.sqrt(((iris_knn.loc[target_idx, fitur] - iris_knn.loc[i, fitur]) ** 2).sum())
    hasil_jarak.append((i, d, iris.loc[i, col_missing]))

# Urutkan berdasarkan jarak terkecil
hasil_jarak = sorted(hasil_jarak, key=lambda x: x[1])

# Ambil 3 tetangga terdekat
k = 3
tetangga = hasil_jarak[:k]

# Imputasi dengan rata-rata (KNN Regression)
imputasi = np.mean([x[2] for x in tetangga])

print("Nilai asli petal_width:", nilai_asli)
print("\n3 tetangga terdekat:")
for t in tetangga:
    print(f"  Baris {t[0]}: jarak = {t[1]:.4f}, petal_width = {t[2]}")
print(f"\nHasil imputasi KNN (k=3): {imputasi:.4f}")
```

**Output yang diharapkan:**

```
Nilai asli petal_width: 0.4

3 tetangga terdekat:
  Baris 10: jarak = 0.2828, petal_width = 0.2
  Baris 48: jarak = 0.3000, petal_width = 0.2
  Baris 18: jarak = 0.3162, petal_width = 0.3

Hasil imputasi KNN (k=3): 0.2333
```

## 13) Penerapan KNN Imputation pada DataCampuran (Data Campuran)

Dataset DataCampuran merupakan **data campuran** karena memiliki kolom numerik dan kategorikal. Oleh karena itu, jarak **tidak bisa langsung** dihitung dengan Euclidean untuk semua kolom — setiap tipe data harus diperlakukan sesuai jenisnya.

### 13.1 Dataset

Dataset `DataCampuran.csv` berisi 6 data mahasiswa:

| No. | Nama | Umur | Tinggi | Status Mahasiswa | Jenis Kelamin |
|---|---|---|---|---|---|
| 1 | Andi | 19 | 165 | Ya | L |
| 2 | Budi | 21 | 170 | Ya | L |
| 3 | Citra | 20 | 158 | Ya | P |
| 4 | Dika | 22 | 172 | Tidak | L |
| 5 | Nana | 19 | 160 | Ya | P |
| 6 | Rudi | 23 | 175 | Tidak | L |

### 13.2 Menentukan Tipe Variabel

Kolom-kolom yang digunakan dalam perhitungan jarak:

| Kolom | Tipe Data | Keterangan |
|---|---|---|
| `Umur` | **Numerik** | Usia mahasiswa |
| `Tinggi` | **Numerik (target)** | Tinggi badan — **kolom yang akan diimputasi** |
| `Status Mahasiswa` | **Kategorikal** | Ya / Tidak |
| `Jenis Kelamin` | **Kategorikal** | L / P |

Karena `Tinggi` berupa numerik, maka pengisian missing value menggunakan **KNN Regression**.

### 13.3 Membuat Missing Value Buatan

Dataset DataCampuran tidak memiliki missing value, sehingga dibuat satu missing value buatan:

- **Baris indeks 2** (baris ke-3, Citra)
- **Kolom `Tinggi`**

Data asli baris 2 (Citra):

| Nama | Umur | Tinggi | Status Mahasiswa | Jenis Kelamin |
|---|---|---|---|---|
| Citra | 20 | **158** | Ya | P |

Setelah dihilangkan:

| Nama | Umur | Tinggi | Status Mahasiswa | Jenis Kelamin |
|---|---|---|---|---|
| Citra | 20 | **?** | Ya | P |

Karena `Tinggi` kosong, maka perhitungan jarak menggunakan kolom: `Umur`, `Status Mahasiswa`, `Jenis Kelamin`.

### 13.4 Langkah 1 — Normalisasi Data Numerik (Min-Max)

Kolom `Umur` dinormalisasi agar skalanya setara:

$$x' = \frac{x - x_{min}}{x_{max} - x_{min}}$$

Dari dataset: `Umur_min = 19`, `Umur_max = 23`.

Citra: `Umur = 20`

$$Umur'_{Citra} = \frac{20 - 19}{23 - 19} = \frac{1}{4} = 0.25$$

### 13.5 Langkah 2 — Perhitungan Jarak Manual

Sekarang kita hitung jarak Citra ke setiap baris lain, dengan menggabungkan:

- **Jarak numerik** (Euclidean) dari kolom `Umur` (setelah normalisasi)
- **Jarak kategorikal** dari kolom `Status Mahasiswa`, `Jenis Kelamin`

#### Citra vs Andi

Andi: `Umur=19, Status=Ya, JK=L`

**Numerik:**

$$Umur'_{Andi} = \frac{19-19}{4} = 0$$

$$d_{num} = \sqrt{(0.25 - 0)^2} = 0.25$$

**Kategorikal:**

| Kolom | Citra | Andi | Sama? |
|---|---|---|---|
| Status Mahasiswa | Ya | Ya | ✓ |
| Jenis Kelamin | P | L | ✗ |

$P = 2$, $M = 1$:

$$d_{kat} = \frac{2-1}{2} = 0.5$$

**Jarak Total:**

$$d_{total} = 0.25 + 0.5 = 0.75$$

#### Citra vs Budi

Budi: `Umur=21, Status=Ya, JK=L`

**Numerik:**

$$Umur'_{Budi} = \frac{21-19}{4} = 0.5$$

$$d_{num} = \sqrt{(0.25 - 0.5)^2} = 0.25$$

**Kategorikal:** Status: Ya=Ya ✓, JK: P≠L ✗ → $d_{kat} = 0.5$

**Jarak Total:**

$$d_{total} = 0.25 + 0.5 = 0.75$$

#### Citra vs Dika

Dika: `Umur=22, Status=Tidak, JK=L`

**Numerik:**

$$Umur'_{Dika} = \frac{22-19}{4} = 0.75$$

$$d_{num} = \sqrt{(0.25 - 0.75)^2} = 0.5$$

**Kategorikal:**

| Kolom | Citra | Dika | Sama? |
|---|---|---|---|
| Status Mahasiswa | Ya | Tidak | ✗ |
| Jenis Kelamin | P | L | ✗ |

$P = 2$, $M = 0$:

$$d_{kat} = \frac{2-0}{2} = 1.0$$

**Jarak Total:**

$$d_{total} = 0.5 + 1.0 = 1.5$$

#### Citra vs Nana

Nana: `Umur=19, Status=Ya, JK=P`

**Numerik:**

$$Umur'_{Nana} = \frac{19-19}{4} = 0$$

$$d_{num} = \sqrt{(0.25 - 0)^2} = 0.25$$

**Kategorikal:**

| Kolom | Citra | Nana | Sama? |
|---|---|---|---|
| Status Mahasiswa | Ya | Ya | ✓ |
| Jenis Kelamin | P | P | ✓ |

$P = 2$, $M = 2$:

$$d_{kat} = \frac{2-2}{2} = 0$$

**Jarak Total:**

$$d_{total} = 0.25 + 0 = 0.25$$

#### Citra vs Rudi

Rudi: `Umur=23, Status=Tidak, JK=L`

**Numerik:**

$$Umur'_{Rudi} = \frac{23-19}{4} = 1.0$$

$$d_{num} = \sqrt{(0.25 - 1.0)^2} = 0.75$$

**Kategorikal:** Status: Ya≠Tidak ✗, JK: P≠L ✗ → $P=2, M=0$: $d_{kat} = 1.0$

**Jarak Total:**

$$d_{total} = 0.75 + 1.0 = 1.75$$

### 13.6 Menentukan 3 Tetangga Terdekat (k = 3)

Setelah seluruh jarak dihitung dan diurutkan:

| Peringkat | Nama | $d_{num}$ | $d_{kat}$ | $d_{total}$ | Tinggi |
|---|---|---|---|---|---|
| 1 | Nana | 0.25 | 0 | **0.25** | 160 |
| 2 | Andi | 0.25 | 0.5 | **0.75** | 165 |
| 3 | Budi | 0.25 | 0.5 | **0.75** | 170 |

**Perhatikan:**

- Nana memiliki jarak terkecil karena kategorikal semua sama (sesama perempuan, sesama mahasiswa aktif)
- Andi dan Budi memiliki jarak sama karena perbedaan hanya di jenis kelamin

Ini menunjukkan pentingnya **memperhitungkan kedua tipe jarak** pada data campuran.

### 13.7 Imputasi Missing Value (KNN Regression)

$$\hat{Tinggi} = \frac{160 + 165 + 170}{3} = \frac{495}{3} = 165.0$$

**Hasil:** Missing value `Tinggi` pada baris 2 (Citra) diisi dengan **165.0**

Nilai asli sebelum dihilangkan: **158**

### 13.8 Kode Python — Imputasi KNN pada Data Campuran DataCampuran

```python
import pandas as pd
import numpy as np
import io

csv_data = """No.;Nama;Umur;Tinggi;Status Mahasiswa;Jenis Kelamin
1;Andi;19;165;Ya;L
2;Budi;21;170;Ya;L
3;Citra;20;158;Ya;P
4;Dika;22;172;Tidak;L
5;Nana;19;160;Ya;P
6;Rudi;23;175;Tidak;L"""

df = pd.read_csv(io.StringIO(csv_data), sep=';')

# Simpan nilai asli
target_idx = 2  # Citra
col_missing = 'Tinggi'
nilai_asli = df.loc[target_idx, col_missing]

# Buat missing value buatan
df_knn = df.copy()
df_knn.loc[target_idx, col_missing] = np.nan

# === NORMALISASI NUMERIK (Min-Max) ===
umur_min = df_knn['Umur'].min()
umur_max = df_knn['Umur'].max()

def umur_norm(val):
    return (val - umur_min) / (umur_max - umur_min)

# === PERHITUNGAN JARAK ===
cat_cols = ['Status Mahasiswa', 'Jenis Kelamin']
hasil_jarak = []

for i in range(len(df_knn)):
    if i == target_idx:
        continue

    # Jarak numerik (Euclidean pada Umur ternormalisasi)
    d_num = np.sqrt(
        (umur_norm(df_knn.loc[target_idx, 'Umur']) - umur_norm(df_knn.loc[i, 'Umur']))**2
    )

    # Jarak kategorikal
    P = len(cat_cols)
    M = sum(df_knn.loc[target_idx, c] == df_knn.loc[i, c] for c in cat_cols)
    d_cat = (P - M) / P

    # Jarak total
    d_total = d_num + d_cat

    tinggi = df_knn.loc[i, col_missing]
    if pd.isna(tinggi):
        continue
    hasil_jarak.append((i, d_total, d_num, d_cat, tinggi, df.loc[i, 'Nama']))

# Urutkan berdasarkan jarak terkecil
hasil_jarak = sorted(hasil_jarak, key=lambda x: x[1])

# Ambil 3 tetangga terdekat
k = 3
tetangga = hasil_jarak[:k]

# Imputasi dengan rata-rata (KNN Regression)
imputasi = np.mean([x[4] for x in tetangga])

print("Nilai asli Tinggi:", nilai_asli)
print("\n3 tetangga terdekat:")
for t in tetangga:
    r = df_knn.loc[t[0]]
    print(f"  {t[5]}: d_total={t[1]:.4f} (d_num={t[2]:.4f}, d_kat={t[3]:.4f}), Tinggi={t[4]}")
    print(f"    Umur={r['Umur']}, Status={r['Status Mahasiswa']}, JK={r['Jenis Kelamin']}")
print(f"\nHasil imputasi KNN (k=3): {imputasi:.2f}")
```

**Output yang diharapkan:**

```
Nilai asli Tinggi: 158

3 tetangga terdekat:
  Nana: d_total=0.2500 (d_num=0.2500, d_kat=0.0000), Tinggi=160
    Umur=19, Status=Ya, JK=P
  Andi: d_total=0.7500 (d_num=0.2500, d_kat=0.5000), Tinggi=165
    Umur=19, Status=Ya, JK=L
  Budi: d_total=0.7500 (d_num=0.2500, d_kat=0.5000), Tinggi=170
    Umur=21, Status=Ya, JK=L

Hasil imputasi KNN (k=3): 165.00
```

## 14) Ringkasan Langkah KNN Imputation

### 14.1 Untuk Data Numerik (Iris)

```
1. Cek missing value → tidak ada → buat missing value buatan
2. Tentukan baris target dan kolom yang kosong
3. Hitung jarak Euclidean ke semua baris lain (tanpa kolom yang kosong)
4. Urutkan jarak dari terkecil
5. Ambil k=3 tetangga terdekat
6. Isi missing value = rata-rata nilai kolom target dari 3 tetangga
```

### 14.2 Untuk Data Campuran (DataCampuran)

```
1. Cek missing value → tidak ada → buat missing value buatan
2. Tentukan tipe setiap kolom: numerik atau kategorikal
3. Normalisasi kolom numerik dengan Min-Max Scaling
4. Hitung jarak numerik menggunakan Euclidean
5. Hitung jarak kategorikal menggunakan (P-M)/P
6. Jumlahkan: d_total = d_num + d_kat
7. Urutkan jarak dari terkecil
8. Ambil k=3 tetangga terdekat
9. Isi missing value = rata-rata nilai target dari 3 tetangga (KNN Regression)
```

## 15) Cara Menghitung Jarak Total dengan Excel

Selain menggunakan Python, perhitungan jarak KNN juga dapat dilakukan menggunakan **Microsoft Excel**. Berikut panduan langkah demi langkah untuk **kedua dataset** (Iris dan DataCampuran).

### 15.1 Perhitungan Jarak Euclidean — Dataset Iris (di Excel)

#### Langkah 1: Siapkan Data

Buat tabel di Excel dengan kolom:

| | A | B | C | D | E |
|---|---|---|---|---|---|
| **1** | **Baris** | **sepal_length** | **sepal_width** | **petal_length** | **petal_width** |
| **2** | 0 | 5.1 | 3.5 | 1.4 | 0.2 |
| **3** | 1 | 4.9 | 3.0 | 1.4 | 0.2 |
| **4** | ... | ... | ... | ... | ... |
| **7** | 5 (target) | 5.4 | 3.9 | 1.7 | **NaN** |

#### Langkah 2: Hitung Jarak Euclidean (tanpa kolom petal_width)

Di kolom baru (misal **F**), masukkan rumus:

```
=SQRT((B$7-B2)^2 + (C$7-C2)^2 + (D$7-D2)^2)
```

**Penjelasan rumus:**
- `B$7` = `sepal_length` baris target (5.4) — tanda `$` mengunci baris 7
- `SQRT(...)` = akar kuadrat untuk Euclidean distance

#### Langkah 3: Copy rumus ke bawah

Drag rumus dari F2 ke bawah sampai semua baris. Excel akan otomatis menghitung jarak ke setiap baris.

#### Langkah 4: Urutkan dan Ambil 3 Terkecil

1. Pilih seluruh data termasuk kolom jarak
2. Klik **Data → Sort** → Sort by kolom F (Jarak) → **Smallest to Largest**
3. Ambil **3 baris teratas** — itulah 3 tetangga terdekat

#### Langkah 5: Hitung Rata-Rata (Imputasi)

```
=AVERAGE(E_tetangga1, E_tetangga2, E_tetangga3)
```

**Hasil yang diharapkan:** `0.2333`

### 15.2 Perhitungan Jarak Campuran — DataCampuran (di Excel)

#### Langkah 1: Siapkan Data

Buat tabel dengan kolom yang relevan:

| | A | B | C | D | E |
|---|---|---|---|---|---|
| **1** | **Nama** | **Umur** | **Tinggi** | **Status Mahasiswa** | **Jenis Kelamin** |
| **2** | Andi | 19 | 165 | Ya | L |
| **3** | Budi | 21 | 170 | Ya | L |
| **4** | Citra (target) | 20 | **?** | Ya | P |
| **5** | Dika | 22 | 172 | Tidak | L |
| **6** | Nana | 19 | 160 | Ya | P |
| **7** | Rudi | 23 | 175 | Tidak | L |

#### Langkah 2: Normalisasi Umur (Min-Max)

Buat kolom bantu **F** untuk normalisasi `Umur`:

```
=(B2 - MIN(B$2:B$7)) / (MAX(B$2:B$7) - MIN(B$2:B$7))
```

#### Langkah 3: Hitung Jarak Numerik (Euclidean)

Buat kolom **G** untuk $d_{num}$:

```
=SQRT((F$4-F2)^2)
```

Di mana `F$4` = Umur ternormalisasi baris target (Citra).

#### Langkah 4: Hitung Jarak Kategorikal

Buat 2 kolom bantu untuk pencocokan kategorikal:

**Kolom H** (Status sama?):
```
=IF(D2=D$4, 1, 0)
```

**Kolom I** (JK sama?):
```
=IF(E2=E$4, 1, 0)
```

Lalu buat kolom **J** untuk $d_{kat}$:

```
=(2 - (H2+I2)) / 2
```

#### Langkah 5: Hitung Jarak Total

Buat kolom **K** untuk $d_{total}$:

```
=G2 + J2
```

#### Langkah 6: Urutkan dan Ambil 3 Terkecil

1. **Data → Sort** by kolom K → **Smallest to Largest**
2. Ambil **3 baris teratas** sebagai tetangga terdekat

#### Langkah 7: Hitung Rata-Rata Tinggi (Imputasi)

```
=AVERAGE(C_tetangga1, C_tetangga2, C_tetangga3)
```

**Hasil yang diharapkan:** `165.0`

### 15.3 Ringkasan Rumus Excel

| Komponen | Rumus Excel | Penjelasan |
|---|---|---|
| **Euclidean Distance** (Iris) | `=SQRT((B$7-B2)^2+(C$7-C2)^2+(D$7-D2)^2)` | Jarak 3 kolom numerik |
| **Normalisasi Min-Max** | `=(B2-MIN(B$2:B$7))/(MAX(B$2:B$7)-MIN(B$2:B$7))` | Skala 0–1 |
| **Jarak Numerik** | `=SQRT((F$4-F2)^2)` | Euclidean pada kolom numerik ternormalisasi |
| **Pencocokan Kategorikal** | `=IF(D2=D$4,1,0)` | 1 jika sama, 0 jika beda |
| **Jarak Kategorikal** | `=(2-(H2+I2))/2` | $(P-M)/P$ |
| **Jarak Total** | `=G2+J2` | $d_{num} + d_{kat}$ |
| **Imputasi (rata-rata)** | `=AVERAGE(...)` | Rata-rata k tetangga terdekat |

### 15.4 Tips Penting di Excel

1. **Gunakan `$` (absolute reference)** pada sel baris target agar rumus tidak bergeser saat di-copy ke bawah.
2. **Jangan ikutkan baris target** dalam perhitungan jarak (jarak ke diri sendiri = 0, tidak bermakna).
3. Gunakan **Conditional Formatting** untuk mewarnai 3 jarak terkecil secara otomatis:
   - Pilih kolom jarak → **Home → Conditional Formatting → Top/Bottom Rules → Top 10 Items** → ubah ke **3** → pilih warna hijau.
4. Untuk dataset besar, gunakan fungsi **SMALL()** untuk menemukan jarak terkecil tanpa perlu sort:
   - `=SMALL(K2:K7, 1)` → jarak terkecil ke-1
   - `=SMALL(K2:K7, 2)` → jarak terkecil ke-2
   - `=SMALL(K2:K7, 3)` → jarak terkecil ke-3

## 16) Kesimpulan

1. **Dataset Iris** termasuk data numerik, sehingga missing value dapat diimputasi langsung menggunakan **Euclidean distance** → hasilnya: `petal_width = 0.2333`

2. **Dataset DataCampuran** merupakan data campuran, sehingga:
   - Data numerik (`Umur`) dinormalisasi menggunakan Min-Max Scaling
   - Data kategorikal (`Status Mahasiswa`, `Jenis Kelamin`) dihitung menggunakan rasio ketidaksamaan $(P-M)/P$
   - Jarak total = jarak numerik + jarak kategorikal
   - Hasilnya: `Tinggi = 165.0`

3. Jika ada missing value pada satu kolom, maka **kolom tersebut tidak diikutkan** dalam perhitungan jarak

4. Setelah semua jarak dihitung, dipilih **k = 3 tetangga terdekat**

5. Karena nilai yang diisi berupa numerik, maka digunakan **KNN Regression** (rata-rata dari tetangga terdekat)

6. Perhitungan KNN Imputation dapat dilakukan dengan **3 cara**: perhitungan manual (tulisan), **kode Python**, atau **Microsoft Excel** — ketiganya menghasilkan hasil yang sama

### Hasil Imputasi

| Dataset | Kolom | Nilai Asli | Hasil Imputasi |
|---|---|---|---|
| Iris | petal_width | 0.4 | **0.2333** |
| DataCampuran | Tinggi | 158 | **165.0** |
