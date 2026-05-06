# Implementasi Klasifikasi Play Tennis menggunakan KNIME

Data yang digunakan adalah data gain ratio.xlsx
| Day | Outlook  | Temp | Humidity | Wind  | PlayTennis |
|-----|----------|------|----------|-------|------------|
| D1  | Sunny    | Hot  | High     | False | No         |
| D2  | Sunny    | Hot  | High     | True  | No         |
| D3  | Overcast | Hot  | High     | False | Yes        |
| D4  | Rain     | Mild | High     | False | Yes        |
| D5  | Rain     | Cool | Normal   | False | Yes        |
| D6  | Rain     | Cool | Normal   | True  | No         |
| D7  | Overcast | Cool | Normal   | True  | Yes        |
| D8  | Sunny    | Mild | High     | False | No         |
| D9  | Sunny    | Cool | Normal   | False | Yes        |
| D10 | Rain     | Mild | Normal   | False | Yes        |
| D11 | Sunny    | Mild | Normal   | True  | Yes        |
| D12 | Overcast | Mild | High     | True  | Yes        |
| D13 | Overcast | Hot  | Normal   | False | Yes        |
| D14 | Rain     | Mild | High     | True  | No         |

ada beberapa atribut: 
- Outlook: Sunny, Overcast, Rain.
- Temperature: Hot, Mild, Cool.
- Humidity: High, Normal.
- Wind: True, False.

target : 
- PlayTennis: Yes, No.

1. Information Gain: Mengukur efektivitas suatu atribut dalam mengklasifikasikan data.  
2. Split Info: Nilai yang digunakan untuk menghambat atribut yang memiliki banyak nilai unik agar tidak mendominasi hasil.  
3. Gain Ratio: Hasil bagi antara Information Gain dan Split Info. Atribut dengan Gain Ratio tertinggi akan dipilih sebagai simpul (node) pemisah.

#### Implementasi KNIME
![gain1](images/gain1.png)

- Excel Reader: Digunakan untuk membaca file dataset. Pastikan baris nama kolom sudah diatur dengan benar agar terbaca sebagai header.
- Table Partitioner: Membagi data menjadi data latih (training) dan data uji (testing) untuk validasi model.
- Color Manager: Memberikan label warna pada kolom target PlayTennis. Hal ini memudahkan identifikasi visual pada pohon akar (misal: coklat untuk 'yes' dan hijau untuk 'no').
- Decision Tree Learner:
    - Class Column: PlayTennis.
    - Quality Measure: Wajib pilih Gain Ratio.  
    - Pruning: Dimatikan agar pohon tampil secara detail (pohon akar lengkap).
- Decision Tree Predictor: Menerapkan model yang sudah dilatih ke data uji.
- Scorer: Menghitung tingkat akurasi model untuk melihat seberapa tepat prediksi yang dihasilkan.

#### Hasil dan Visualisasi Decition Tree
![gain2](images/gain2.png)
- Akar Utama (Root Node): Atribut Humidity (teridentifikasi sebagai empty_E) terpilih sebagai pemisah utama karena memiliki Gain Ratio tertinggi.
- Distribusi Data: Dari total 12 data yang masuk ke model, 7 data (58.3%) menghasilkan keputusan Yes dan 4 data (33.3%) menghasilkan keputusan No.
- Logika Percabangan:
    - Jika Humidity = Normal: Mayoritas data (83.3%) mengarah pada keputusan Yes.  
    - Jika Humidity = High: Mayoritas data (60%) mengarah pada keputusan No.

#### Evaluasi Model Scorer
![gain3](images/gain3.png)
- Correct Classified: 1 (Satu data berhasil ditebak dengan benar)
- Wrong Classified: 1 (Satu data gagal ditebak dengan benar).
- Accuracy: 50% (Model memiliki tingkat ketepatan sebesar 50% pada data uji).
- Error: 50% (Tingkat kesalahan prediksi model adalah 50%).

Tabel ini menunjukkan perbandingan antara nilai asli (Actual) dengan hasil prediksi (Prediction):
- True Negative: Terdapat 1 data "No" yang diprediksi dengan benar sebagai "No".
- False Negative: Terdapat 1 data "Yes" yang salah diprediksi menjadi "No".
