# MISSING VALUES

Missing Values adalah yang hilang atau kosong dalam dataset.
Contoh Data :
| No | IPK | PO      | JML |
| -- | --- | ------- | --- |
| 1  | 2   | 2000000 | 2   |
| 2  | 3   | 3000000 | 3   |
| 3  | 4   | 2000000 | 2   |
| 4  | 2   | 2000000 | 3   |
| 5  | 3   | 3000000 | 2   |
| 6  | 4   | 4000000 | 3   |
| 7  | 2   | 3000000 | ?   |

Nilai JML pada data ke-7 hilang, sehingga harus diprediksi menggunakan WKNN (Weighted K-Nearest Neighbor).
**Penyebab Missing Values**
- Kesalahan saat pengumpulan data
- Data tidak diinput oleh responden atau sistem
- Kerusakan atau kehilangan data
- Kesalahan sat proses penyimpanan atau pengolahan data.
**Cara Mengatasi Missing Values**
Beberapa metode yang dapat digunakan untuk mengatasi missing values:
- menghapus data yang memiliki nilai kosong.
- Mengisi dengan nilai Mean, Median, atau Modus.
- Menggunakan metode KNN atau WKNN(Weight K-Nearest Neighbor)
- Menggunakan metode imputasi lainnya.

Pada data tersebut menghitung menggunakan metode WKNN

**Langkah WKNN manual:**
### 1. Normalisasi Data MIN-MAX
Rumus Normalisasi Data: ![image](https://hackmd.io/_uploads/B1BobmEcZx.png)
Keterangan:
- x = nilai data
- xmin = nilai terkecil
- xmax = nilai terbesar
- x′ = hasil normalisasi

### 2.Menentukan Nilai Min-Max
**Nilai IPK:** 
- MIN = 2
- MAX = 4
**Nilai PO :**
- MIN = 2000000
- MAX = 4000000
**Nilai JML:**
- MIN = 2
- MAX = 3
**Contoh Perhitungan**
**Normalisasi IPK** 
Data 1 (IPK = 2)
![image](https://hackmd.io/_uploads/BkjRYVE9Zg.png)
**Normalisasi PO**
Data 1 (PO = 2000000)
![image](https://hackmd.io/_uploads/S1kzcEN5be.png)
**Normalisasi JML**
JML = 2 
![image](https://hackmd.io/_uploads/rJCXcNN5Wx.png)
JML = 3
![image](https://hackmd.io/_uploads/rk-r9V4qbe.png)

**Data setelah di normalisasi**
| Data | IPK | PO  | JML |
| ---- | --- | --- | --- |
| 1    | 0   | 0   | 0   |
| 2    | 0.5 | 0.5 | 1   |
| 3    | 1   | 0   | 0   |
| 4    | 0   | 0   | 1   |
| 5    | 0.5 | 0.5 | 0   |
| 6    | 1   | 1   | 1   |
| 7    | 0   | 0.5 | ?   |

### 3. Menghitung JaraK Euclidean
Jarak Euclidean adalah metode untuk menghitung jarak antara dua titik dalam ruang multidimensi. Dalam data mining atau machine learning, jarak Euclidean digunakan untuk:
- mengukur kemiripan antar data
- menentukan tetangga terdekat pada algoritma KNN
Semakin kecil jaraknya, maka data tersebut semakin mirip.
Rumus Euclidean : ![image](https://hackmd.io/_uploads/By2tXDHc-x.png)
Keterangan : 
- d = jarak antara dua data
- x₁ , y₁ = nilai atribut data pertama
- x₂ , y₂ = nilai atribut data kedua

Target data yang dicari adalah data missing JML pada baris ke-7 setelah di normalisasi dan akan dibandingkan dengan data 1 sampai data 6 setelah di normalisasi.
Data yang dibandingkan adalah IPK data i , PO data i dengan IPK data target , PO data target
**Contoh perhitungan:**
**1. Jarak data 1 ke data 7**
Data 1 = (IPK = 0, PO = 0)
Data 7 = (IPK = 0, PO = 0.5)
![image](https://hackmd.io/_uploads/SJ5WQPB9-g.png)
**2. Jarak data 2 ke data 7**
Data 2 = (IPK= 0.5, PO = 0.5)
Data 7 = (IPK = 0, PO = 0.5)
![image](https://hackmd.io/_uploads/BJtImvHcWe.png)
**3. Jarak data 3 ke data 7**
Data 3 = (IPK= 1, PO = 0)
Data 7 = (IPK = 0, PO = 0.5)
![image](https://hackmd.io/_uploads/Synr4vH9Wl.png)
**4. Jarak data 4 ke data 7**
Data 4 = (IPK= 0, PO = 0)
Data 7 = (IPK = 0, PO = 0.5)
![image](https://hackmd.io/_uploads/SyliUEPHqZe.png)
**5. Jarak data 5 ke data 7**
Data 5 = (IPK= 0.5, PO = 0.5)
Data 7 = (IPK = 0, PO = 0.5)
![image](https://hackmd.io/_uploads/SJkKVwS9-g.png)
**6. Jarak data 6 ke data 7**
Data 6 = (IPK= 1, PO = 1)
Data 7 = (IPK = 0, PO = 0.5)
![image](https://hackmd.io/_uploads/Hyu9Ewr9Zx.png)

**Hasil semua jarak :**
| Data | Jarak |
| ---- | ----- |
| 1    | 0.5   |
| 2    | 0.5   |
| 3    | 1.118 |
| 4    | 0.5   |
| 5    | 0.5   |
| 6    | 1.118 |

### 4. Menghitung Bobot WKNN
Rumus : ![image](https://hackmd.io/_uploads/r1cZzKrqZe.png)
Keterangan 
- wi = bobot data ke-i
- di = jarak Euclidean data ke-i terhadap data target

Contoh perhitungan bobot
**Data 1** : ![image](https://hackmd.io/_uploads/SylKdXqSqZe.png)
**Data 2** : ![image](https://hackmd.io/_uploads/HyStQ9r5-x.png)
**Data 3** : ![image](https://hackmd.io/_uploads/SJA5XqHqbg.png)
**Data 4** : ![image](https://hackmd.io/_uploads/HJb2QcScZx.png)
**Data 5** : ![image](https://hackmd.io/_uploads/H1Wp7cH5-x.png)
**Data 6** : ![image](https://hackmd.io/_uploads/SkwAQcr9Wl.png)
**Hasil Perhitungan Bobot WKNN**
| Data | Jarak | Bobot | JML |
| ---- | ----- | ----- | --- |
| 1    | 0.5   | 2     | 0   |
| 2    | 0.5   | 2     | 1   |
| 3    | 1.118 | 0.894 | 0   |
| 4    | 0.5   | 2     | 1   |
| 5    | 0.5   | 2     | 0   |
| 6    | 1.118 | 0.894 | 1   |

### 5. Pilih K Tetangga Terdekat
Nilai K adalah jumlah tetangga yang akan digunakan.
Misal K =3 
| Data | Jarak | Bobot | JML |
| ---- | ----- | ----- | --- |
| 1    | 0.5   | 2     | 2   |
| 2    | 0.5   | 2     | 3   |
| 4    | 0.5   | 2     | 3   |

Dari ketiga ini disebut K tetangga terdekat.

### 6. Hitung WKNN 
Rumus : ![image](https://hackmd.io/_uploads/Skuwk9S5be.png)
Perhitungan: ![image](https://hackmd.io/_uploads/SyuKJ9BcWe.png)
Jumlah Bobot: ![image](https://hackmd.io/_uploads/BkPqJcr9-l.png)
Hasil akhir: ![image](https://hackmd.io/_uploads/rk4ijJ9B9Ze.png)
Karena nilai JML berupa bilangan bulat, maka hasil akhir dibulatkan menjadi 3
Hasil imputasi missing value JML pada data ke-7 adalah 3

**Tabel Akhir**
| Data | IPK | PO      | JML   |
| ---- | --- | ------- | ----- |
| 1    | 2   | 2000000 | 2     |
| 2    | 3   | 3000000 | 3     |
| 3    | 4   | 2000000 | 2     |
| 4    | 2   | 2000000 | 3     |
| 5    | 3   | 3000000 | 2     |
| 6    | 4   | 4000000 | 3     |
| 7    | 2   | 3000000 | **3** |

**Tabel Data setalah di normalisasi**
| Data | IPK | PO  | JML   |
| ---- | --- | --- | ----- |
| 1    | 0   | 0   | 0     |
| 2    | 0.5 | 0.5 | 1     |
| 3    | 1   | 0   | 0     |
| 4    | 0   | 0   | 1     |
| 5    | 0.5 | 0.5 | 0     |
| 6    | 1   | 1   | 1     |
| 7    | 0   | 0.5 | **1** |

Nilai JML pada data ke-7 diisi dari hasil WKNN = 3. Jika dinormalisasi : ![image](https://hackmd.io/_uploads/SyjY_5H5bx.png)
maka nilai normalisasi JML adalah 1

## NORMALISASI DATA
Normalisasi data adalah proses mengubah skala nilai pada dataset agar berada pada rentang tertentu tanpa mengubah hubungan antar data.
Tujuannya adalah : 
- Menyamakan skala data
- Mempercepat proses perhitungan algoritma
- Menghindari dominasi nilai yang besar
Normalisasi sering digunakan pada algoritma seperti KNN, clustering, dan neural network.

#### Macam - Macam Normalisasi Data
##### 1. Min Max Normalization
Metode ini mengubah nilai data ke rentang 0 sampai 1.
Rumus : ![image](https://hackmd.io/_uploads/HJ-22cH5-e.png)
Contoh : 
**Data Awal** 
| Data | IPK | PO      | JML |
| ---- | --- | ------- | --- |
| 1    | 2   | 2000000 | 2   |
| 2    | 3   | 3000000 | 3   |
| 3    | 4   | 2000000 | 2   |
| 4    | 2   | 2000000 | 3   |
| 5    | 3   | 3000000 | 2   |
| 6    | 4   | 4000000 | 3   |
| 7    | 2   | 3000000 | 3   |

**Menentukan Min - Max**
- IPK (Min = 2, Max = 4)
- PO (Min = 2000000, Max = 4000000)
- JML (Min = 2, Max = 3)
**Menghitung IPK**
**Data 1** : (2−2)/(4−2)=0
**Data 2** : (3−2)/(4−2)=0.5
**Data 3**: (4−2)/(4−2)=1
**Menghitung PO**
**Data 1 :** (2000000−2000000)/(4000000−2000000)=0
**Data 2 :**(3000000−2000000)/(4000000−2000000)=0.5
**Data 6 :** (4000000−2000000)/(4000000−2000000)=1
**Menghitung JML**
Data 1 : (2−2)/(3−2)=0
Data 2 : (3−2)/(3−2)=1

**Hasil Normalisasi Min-Max**
| Data | IPK | PO  | JML |
| ---- | --- | --- | --- |
| 1    | 0   | 0   | 0   |
| 2    | 0.5 | 0.5 | 1   |
| 3    | 1   | 0   | 0   |
| 4    | 0   | 0   | 1   |
| 5    | 0.5 | 0.5 | 0   |
| 6    | 1   | 1   | 1   |
| 7    | 0   | 0.5 | 1   |

##### 2. Z-Score Normalization
Metode ii menyesuaikan data berdasarkan mean dan standar deviasi
Rumus: ![image](https://hackmd.io/_uploads/HJPduQL9We.png)
Keterangan : 
- x = nilai data
- μ = mean (rata-rata)
- σ = standar deviasi

**1. Menghitung Mean (Rata-rata)**
Rumus : ![image](https://hackmd.io/_uploads/SyC8d78qWe.png)
Menjumlahkan semua data lalu bagi jumlah data
μ = 2+3+4+2+3+4+2/7
μ = 20/7
μ = 2.857
Jadi mean adalah **2.857**
**2. Mengurangi setiap data dengan mean (x - mean)**
| Data | Perhitungan | Hasil  |
| ---- | ----------- | ------ |
| 2    | 2 − 2.857   | -0.857 |
| 3    | 3 − 2.857   | 0.143  |
| 4    | 4 − 2.857   | 1.143  |
| 2    | 2 − 2.857   | -0.857 |
| 3    | 3 − 2.857   | 0.143  |
| 4    | 4 − 2.857   | 1.143  |
| 2    | 2 − 2.857   | -0.857 |

**3. Kuadratkan Hasilnya**
Hitung (x − μ)²
| Data | x-μ    | (x-μ)² |
| ---- | ------ | ------ |
| 2    | -0.857 | 0.735  |
| 3    | 0.143  | 0.020  |
| 4    | 1.143  | 1.306  |
| 2    | -0.857 | 0.735  |
| 3    | 0.143  | 0.020  |
| 4    | 1.143  | 1.306  |
| 2    | -0.857 | 0.735  |

Jumlahkan semua hasilnya
0.735 + 0.020 + 1.306 + 0.735 + 0.020 + 1.306 + 0.735 = **4.857**

**4. Menghitung Standar Deviasi**
Hasil jumlah dibagi dengan 7 dan di akarkan
![image](https://hackmd.io/_uploads/Skgah7I5Ze.png)
Jadi standar deviasi adalah 0.833
**Menghitung Z-Score**
Rumus : ![image](https://hackmd.io/_uploads/S1dQa7IqZx.png)
**Data 1 (IPK = 2)** 
z = 2−2.857/0.833 
z = -1.03
**Data 2 (IPK = 3)** 
z = 3−2.857/0.833 
z = 0.17
**Data 3 (IPK = 4)**
z = 4−2.857/0.833 
z = 1.37

**Hasil Akhir**
| Data | IPK    | Z-Score|
| ---- | ------ | ------ |
| 1    | 2      | -1.03  |
| 2    | 3      | 0.17   |
| 3    | 4      | 1.37   |
| 4    | 2      | -1.03  |
| 5    | 3      | 0.17   |
| 6    | 4      | 1.37   |
| 7    | 2      | -1.03  |

##### 3. Decimal Scalling
Metode ini membagi nilai dengan 10ⁿ hingga berada pada rentang kecil.
Rumus: ![image](https://hackmd.io/_uploads/SyUV2MLcWe.png)
Keterangan : 
- x = nilai data
- j = jumlah digit dari nilai terbesar pada data
**Menentukan data yang akan dinormalisasi**
Contoh menggunakan data PO : 2000000, 3000000, 2000000, 2000000, 3000000, 4000000, 3000000
Nilai terbesar dari data tersebut adalah **4000000**
Menentukan jumlah digit. pada data nilai terbesar, 4000000 memiliki 7 digit.
**jadi, jumlah j = 7**
Selanjutnya menghitung jumlah 10 dipangkatkan dengan 7 (10^j) adalah 10^7=10000000
Lalu membagi setiap data dengan 10^j 
Data 1 : 2000000/10000000 = 0.2
Data 2 : 3000000/10000000 = 0.3
Data 3 : 2000000/10000000 = 0.2
Data 4 : 2000000/10000000 = 0.2
Data 5 : 3000000/10000000 = 0.3
Data 6 : 4000000/10000000 = 0.4
Data 7 : 3000000/10000000 = 0.3

**Hasil Normalisasi Decimal Scaling**
| Data | PO      | Hasil Normalisasi |
| ---- | ------- | ----------------- |
| 1    | 2000000 | 0.2               |
| 2    | 3000000 | 0.3               |
| 3    | 2000000 | 0.2               |
| 4    | 2000000 | 0.2               |
| 5    | 3000000 | 0.3               |
| 6    | 4000000 | 0.4               |
| 7    | 3000000 | 0.3               |


##### 4. Normalisasi Menggunakan Pyhton (Sklearn)
``````import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Dataset
data = {
    'IPK': [2,3,4,2,3,4,2],
    'PO': [2000000,3000000,2000000,2000000,3000000,4000000,3000000],
    'JML': [2,3,2,3,2,3,3]
}

df = pd.DataFrame(data)

# Min-Max Normalization
minmax_scaler = MinMaxScaler()
minmax = minmax_scaler.fit_transform(df)

# Z-Score Normalization
zscore_scaler = StandardScaler()
zscore = zscore_scaler.fit_transform(df)

print("Min-Max Normalization")
print(minmax)

print("\nZ-Score Normalization")
print(zscore)
``````
##### 5. Membuat Fungsi Normalisasi Sendiri
``````import pandas as pd
import numpy as np

# Dataset
data = {
    'IPK': [2,3,4,2,3,4,2],
    'PO': [2000000,3000000,2000000,2000000,3000000,4000000,3000000],
    'JML': [2,3,2,3,2,3,3]
}

df = pd.DataFrame(data)

# Fungsi Min-Max
def minmax_normalisasi(data):
    return (data - data.min()) / (data.max() - data.min())

# Fungsi Z-Score
def zscore_normalisasi(data):
    mean = data.mean()
    std = data.std()
    return (data - mean) / std

# Fungsi Decimal Scaling
def decimal_scaling(data):
    max_val = data.abs().max()
    j = len(str(int(max_val)))
    return data / (10**j)

# Hasil Normalisasi
minmax = df.apply(minmax_normalisasi)
zscore = df.apply(zscore_normalisasi)
decimal = df.apply(decimal_scaling)

print("Data Asli")
print(df)

print("\nMin-Max Normalization")
print(minmax)

print("\nZ-Score Normalization")
print(zscore)

print("\nDecimal Scaling")
print(decimal)
``````
