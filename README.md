## Domain Proyek

Proyek ini mengangkat tema untuk memprediksi harga komoditas pangan yang ada di pasar Kramat Kota Cirebon. tema ini diangkat karena masalah kenaikan harga yang sangat fluktuatif untuk komoditas pangan yang ada sehingga pihak terkait dapat mempersiapkan langkah yang lebih awal untuk mencegah kenaikan harga yang tidak terkendali. proyek ini melakukan pendekatan prediksi menggunakan deep learning dan kombinasi algoritma LSTM-GRU untuk membuat model prediksi, kombinasi tersebut dipilih setelah melakukan beberapa tinjauan pada jurnal. pada penelitian ini dataset yang digunakan ada 5 data harga komoditas pangan dengan tingkat persentase kenaikan tertinggi, namun pada dokumentasi ini, saya hanya menggunakan 1 buah komoditas saja (bawang merah) untuk mensimplifikasi dokumentasi.

**Referensi**
- [Prediksi Harga Komoditas Pangan Menggunakan Algoritma Long Short-Term Memory (LSTM)](https://ejurnal.seminar-id.com/index.php/bits/article/view/2229)
- [Prediksi Harga Pangan Di Pasar Tradisional Kota Surabaya Dengan Metode LSTM](https://journal.untar.ac.id/index.php/jiksi/article/view/26012)
- [Stock Price Prediction by Normalizing LSTM and GRU Models](https://sifisheriessciences.com/journal/index.php/journal/article/view/1875)
- [Applying Hybrid Lstm-Gru Model Based on Heterogeneous Data Sources for Traffic Speed Prediction in Urban Areas](https://www.mdpi.com/1424-8220/22/9/3348)

## Business Understanding

Berdasarkan data harga yang diperoleh dari proses scrapping website Pusat Informasi Harga Pangan Strategis (PIHPS) Nasional untuk Pasar Kramat Kota Cirebon dari tanggal 3 Mei 2021 - 3 Mei 2024 (3 tahun terakhir), dari 14 komoditas pangan yang diambil terdapat 5 komoditas yang mengalami persentase kenaikan tertinggi lebih dari 100%. berikut datanya : 

| Komoditas               | Harga Minimum (Rp) | Harga Maksimum (Rp) | Persentase Kenaikan (%) | Kenaikan Harga (Rp) |
|--------------------------|--------------------|---------------------|------------------------|---------------------|
| Cabai Merah Besar        | 15.800             | 110.000             | 596,20                 | 94.200              |
| Cabai Merah Keriting     | 17.000             | 105.000             | 517,65                 | 88.000              |
| Cabai Rawit Hijau        | 17.000             | 100.000             | 488,24                 | 83.000              |
| Cabai Rawit Merah        | 24.500             | 120.000             | 389,80                 | 95.500              |
| Bawang Merah             | 19.000             | 70.000              | 268,42                 | 51.000              |
| Bawang Putih             | 28.000             | 51.000              | 82,14                  | 23.000              |
| Telur Ayam Ras           | 18.500             | 33.000              | 78,38                  | 14.500              |
| Minyak Goreng Kemasan    | 14.000             | 24.000              | 71,43                  | 10.000              |
| Ayam Ras Segar           | 24.500             | 40.000              | 63,27                  | 15.500              |
| Beras Bawah              | 9.500              | 14.500              | 52,63                  | 5.000               |
| Beras Medium             | 11.000             | 16.000              | 45,45                  | 5.000               |
| Gula Pasir Lokal         | 13.000             | 18.500              | 42,31                  | 5.500               |
| Beras Super              | 12.000             | 16.500              | 37,50                  | 4.500               |
| Daging Sapi              | 130.000            | 160.000             | 23,08                  | 30.000              |

**Catatan**

- Sumber data dari website [PIHPS Nasional](https://www.bi.go.id/hargapangan) dari tanggal 3 Mei 2021 - 3 Mei 2024
- Harga minimum dan maksimum dalam Rupiah (Rp).
- Persentase kenaikan dihitung dari harga minimum ke harga maksimum.
- Kenaikan harga adalah selisih antara harga maksimum dan harga minimum.


Dapat dilihat bahwa di antara komoditas-komoditas tersebut, cabai merah besar mengalami kenaikan tertinggi sebesar 596,2%, diikuti cabai merah keriting (517,6%), cabai rawit hijau (488,2%), cabai rawit merah (389,8%), dan bawang merah (268,4%). 

Berbagai upaya telah dilakukan oleh pihak terkait untuk mengendalikan fluktuasi harga, salah satunya melalui inspeksi mendadak (sidak) pasar. Sidak pasar merupakan kegiatan pemeriksaan kondisi pasar secara langsung dan tanpa pemberitahuan sebelumnya oleh pejabat pemerintah, biasanya dari Dinas Perdagangan atau instansi terkait lainnya. Kegiatan ini bertujuan untuk memastikan stabilitas harga komoditas pangan.

Namun, dalam praktiknya, sidak pasar seringkali dilakukan hanya pada waktu-waktu tertentu atau setelah adanya laporan mengenai lonjakan harga yang tidak wajar. Hal ini menyebabkan keterlambatan dalam pengambilan tindakan lanjutan, seperti pengeluaran cadangan pangan, sehingga harga komoditas pangan terus meningkat. Sehingga salah satu cara untuk mengatasi hal tersebut adalah dengan melakukan prediksi harga.

### Problem Statements

Poin Permasalahan :
- Kenaikan harga komoditas pangan yang sangat fluktuatif dan tidak dapat diprediksi.
- Keterlambatan penanganan tindak lanjut dari pihak terkait atas kenaikan harga komoditas pangan tertentu, sehingga harga melonjak secara tiba tiba dan tidak terkendali.

### Goals
Tujuan dilakukan nya penelitian :
- Untuk mencegah lonjakan harga komoditas dengan melakukan antisipasi dini terhadap potensi kenaikan harga berdasarkan hasil prediksi, sehingga dapat mempersiapkan langkah-langkah pengendalian yang diperlukan oleh pihak terkait.

### Solution
- Melakukan pelatihan model untuk memprediksi harga komoditas pangan, pendekatan algoritma yang digunakan adalah algoritma LSTM-GRU.

## Data Understanding

Dataset ini mengadung 6 buah kolom, satu berisi tanggal dan sisanya berisi komoditas harga pangan dengan persentase kenaikan diatas 100% dalam 3 tahun terakhir (3 Mei 2021 - 3 Mei 2024). namun karena peramalan ini menggunakan data univariate (data tunggal), jadi setiap nilai di setiap kolom tidak memiliki saling berkaitan. dataset ini diambil dari website [PIHPS Nasional](https://www.bi.go.id/hargapangan)

kolom : 
- date : Tanggal diambilnya harga
- cabai_merah_besar : harga cabai merah besar
- cabai_merah_keriting : harga cabai merah keriting
- cabai_rawit_hijau : harga cabai rawit hijau
- cabai_rawit_merah : harga cabai rawit merah
- bawang_merah : harga bawang merah

## Data Preparation

Data preparation meliputi : 
- Interpolasi dataset
- Pemecahahan dataset (80 % train & 20 % test)
- Normalisasi data
- Pembentukan Sliding Window

Penjelasan : 
- Interpolasi dataset diperlukan agar tidak adanya data kosong, interpolasi dilakukan dengan teknik interpolasi linear yang mengisi nilai nilai kosong pada dataset (sudah dilakukan sebelumnya, tidak dari kode colab)
- Pemecahan dataset menjadi 80% train dan 20% test diperlukan agar model bisa belajar dengan baik.
- Normalisasi data dilakukan agar bentuk data menjadi seragam (rentang 0 - 1) dan proses komputasi lebih ringan.
- Dalam konteks prediksi menggunakan deep learning, teknik sliding window membantu memecah data berurutan menjadi segmen-segmen yang lebih kecil sehingga memungkinkan model untuk mempelajari pola-pola lokal dalam data dan membuat prediksi berdasarkan data historis dalam jendela tersebut. jika dari konteks model yang kita buat, saya ingin melakukan prediksi harga berdasarkan data 30 hari terakhir.

## Modeling
Pada tahap ini, modelling dilakukan dengan tensorflow dan menggunakan Algoritma LSTM-GRU
```bash
# Create LSTM + GRU Model
def define_model():
    input1 = Input(shape=(window_size,1))
    x = LSTM(units = set_neurons, return_sequences=True)(input1)
    x = GRU(units = set_neurons)(x)  # Setting return_sequences=False to get 2D output
    dnn_output = Dense(1)(x)

    model = Model(inputs=input1, outputs=[dnn_output])
    model.compile(loss='mean_squared_error', optimizer=Adam())
    model.summary()

    return model
```
Pada file google colab ini, dilakukan pelatihan model tersebut menggunakan hyperparameter:
- epoch : 100
- neurons : 16
- batch_size : 32
- optimizer: Adam

gambaran visual bagaimana model terhubung dapat dilihat pada gambar berikut : 
![image](https://github.com/user-attachments/assets/4aa3fd6d-b8ba-4738-a6dc-1439a5ab86d1)

Model ini bekerja dengan cara menerima 30 data input pada layer pertama yang membentuk node pada input layer. Setelah itu, input diteruskan ke hidden layer pertama, yaitu LSTM dengan jumlah 16 neuron. Pada proses ini, LSTM mengembalikan output dengan return_sequences=True, yang berarti setiap timestep menghasilkan output dan seluruh urutan akan diteruskan ke layer berikutnya. Output dari LSTM memiliki bentuk (batch_size, 30 timesteps, 16 unit).

Selanjutnya, output ini diproses oleh hidden layer kedua, yaitu GRU yang juga memiliki 16 neuron. GRU ini mengembalikan output dengan return_sequences=False, sehingga hanya output dari timestep terakhir yang digunakan. Hasil dari GRU berbentuk (batch_size, 16 unit).

Nilai ini kemudian diteruskan ke Dense layer tunggal untuk menghasilkan nilai akhir sebagai output prediksi dari model.

## Evaluation
Evaluasi disini dilakukan dengan menggunakan metriks RMSE, MSE, MAPE, Accuracy untuk melihat performa model.

**Detail Pengertian Metriks yang digunakan**

<img src="https://github.com/user-attachments/assets/fc29ae29-f792-46c8-a1d0-ca457e1c7e73" height=100 width=200>

MSE = Mean Squared Error atau menghitung rata rata kuadrat error dari nilai prediksi dan nilai aktual

![image](https://github.com/user-attachments/assets/2d00b4d1-1f68-4928-95da-fa15a7c73a02)

RMSE = Root Mean Squared Error atau menghitung akar dari rata rata kuadrat error dari nilai prediksi dan nilai aktual 

![image](https://github.com/user-attachments/assets/6d03e7e7-7cef-48d6-933d-a52d2637dac3)

MAPE = Mean average percentage error atau menghitung rata rata nilai error kemudian dikalikan 100%

Accuracy = 1 - MAPE

**Hasil Evaluasi**

hasil evaluasi menunjukan performa yang cukup baik dengan nilai sebagai berikut (data masih dalam bentuk normalisasi) : 

```
Test Loss (MSE): 0.0011948698665946722
Test RMSE: 0.034566878222976216
Test MAPE: 0.04133664902587545
Test Accuracy: 0.9586633509741246
```

hasil evaluasi menunjukan performa yang cukup baik dengan nilai sebagai berikut (data sudah didenormalisasi agar tercermin hasil yang lebih nyata) : 

```
Test MSE on denormalized data: 3107819.0593
Test RMSE on denormalized data: 1762.900
Test MAPE on denormalized data: 1.807%
Test Accuracy on denormalized data: 98.1%
```

kedua hasil evaluasi prediksi tersebut baik masih dalam bentuk normalisasi atau denormalisasi menghasilkan hasil evaluasi yang baik.
