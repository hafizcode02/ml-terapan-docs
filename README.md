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
Harga komoditas pangan di Pasar Kramat Kota Cirebon mengalami fluktuasi yang sangat tinggi dalam tiga tahun terakhir. Kondisi ini menyulitkan instansi pemerintah dan stakeholder terkait, seperti Dinas Perdagangan dan pelaku distribusi pangan, dalam mengantisipasi lonjakan harga secara tepat waktu.

Dua permasalahan utama yang ingin diselesaikan dalam proyek ini adalah:
- Bagaimana cara mengatasi kenaikan harga komoditas pangan yang sangat fluktuatif dan sulit diprediksi?
- Bagaimana cara mempercepat pengambilan tindakan dari pihak terkait untuk mengendalikan harga pangan sebelum terjadi lonjakan yang ekstrem?

### Goals
Penelitian ini bertujuan untuk:

- Mengembangkan model prediksi harga komoditas pangan menggunakan pendekatan deep learning berbasis kombinasi LSTM dan GRU. Model ini bertujuan untuk memberikan estimasi harga secara akurat berdasarkan data historis.
- Menyediakan alat bantu prediksi yang dapat digunakan oleh pemerintah daerah, khususnya Dinas Perdagangan, sebagai bagian dari sistem peringatan dini. Dengan prediksi harga yang akurat, pihak terkait dapat merespons lebih cepat, misalnya melalui intervensi pasar atau distribusi cadangan pangan, sehingga harga tetap terkendali.
- Memberikan insight berbasis data yang dapat dimanfaatkan oleh peneliti dan pengambil kebijakan untuk perencanaan jangka menengah dan panjang terkait pengelolaan distribusi pangan.

## Data Understanding

Dataset ini mengadung 6 buah kolom, satu berisi tanggal dan sisanya berisi komoditas harga pangan dengan persentase kenaikan diatas 100% dalam 3 tahun terakhir (3 Mei 2021 - 3 Mei 2024). namun karena peramalan ini menggunakan data univariate (data tunggal), jadi setiap nilai di setiap kolom tidak memiliki saling berkaitan. dataset ini diambil dari website [PIHPS Nasional](https://www.bi.go.id/hargapangan)

jumlah data : 1096 baris data & 6 kolom.

kondisi data : sebelumnya terdapat data kosong pada hari sabtu dan minggu, serta beberapa hari yang kosong juga. namun sebelum masuk ke pelatihan model ini, data yang kosong tersebut telah dilakukan proses interpolasi dengan teknik **interpolasi linear** di excel.

kolom : 
- date : Tanggal diambilnya harga
- cabai_merah_besar : harga cabai merah besar
- cabai_merah_keriting : harga cabai merah keriting
- cabai_rawit_hijau : harga cabai rawit hijau
- cabai_rawit_merah : harga cabai rawit merah
- bawang_merah : harga bawang merah

## Data Preparation



Pada tahap ini, dilakukan serangkaian proses untuk menyiapkan data sebelum dimasukkan ke dalam model. Langkah-langkah yang dilakukan sebagai berikut:

1. Pemilihan Data Komoditas Data yang digunakan merupakan harga komoditas tertentu yang dipilih dan ditetapkan pada variable ```commodity_selected``` sebelumnya.
   
2. Normalisasi Data Sebelum dilakukan pemisahan data menjadi training dan testing, seluruh data terlebih dahulu digunakan untuk melakukan fitting pada MinMaxScaler. Hal ini bertujuan untuk mendapatkan skala minimum dan maksimum dari keseluruhan dataset, yang kemudian akan digunakan untuk mentransformasi data training dan data testing. Normalisasi ini membantu mempercepat proses pelatihan model dan meningkatkan konvergensi.
    ```
    scaler = MinMaxScaler()
    scaler.fit(df[comodity_list[comodity_selected]].values.reshape(-1,1))
    ```
    Scaler ini kemudian disimpan menggunakan joblib untuk keperluan pemodelan selanjutnya.

3. Penentuan Ukuran Data Testing,
   Proporsi data testing ditentukan menggunakan variabel set_test_size (misalnya 0.2 untuk 20%), kemudian dikonversi ke jumlah data aktual:
    ```
        test_size = int(len(df) * set_test_size)
    ```
    
4. Pemisahan Data Training, Normalisasi, dan Pembuatan Dataset Berbasis Window
   - Pemisahan Data Training (diambil dari awal hingga sebelum data testing)
   - Normalisasi Data Training
   - Pembulatan nilai normalisasi ke maksimal 6 angka dibelakang koma, agar proses komputasi menjadi lebih ringan
   - Membuat Dataset Berbasis Window dengan metode windowed sequence yaitu dengan memanfaatkan 30 data terakhir (window size) sebagai input untuk memprediksi 1 data berikutnya.
    ```
    # Panjang data Loopback x hari
    window_size = set_window_size
    
    # Persiapkan data Training 80% dari data, dan normalisasikan
    train_data = df[comodity_list[comodity_selected]][:-test_size]
    train_data = scaler.transform(train_data.values.reshape(-1,1))
    
    # Membulatkan hingga 6 angka di belakang koma
    train_data = np.round(train_data, 6)
    
    # Siapkan Variable untuk menampung data Train, disesuaikan dengan data loopback yaitu 30
    X_train = []
    y_train = []
    
    for i in range(window_size, len(train_data)):
        X_train.append(train_data[i-30:i, 0])
        y_train.append(train_data[i, 0])
    ```

5. Pemisahan Data Testing, Normalisasi, dan Pembuatan Dataset Berbasis Window
   - Data testing diambil dari 30 data sebelum awal data test hingga ke akhir (agar mencakup window size saat transformasi berurutan nanti).
   - Normalisasi Data Testing
   - Pembulatan nilai normalisasi ke maksimal 6 angka dibelakang koma, agar proses komputasi menjadi lebih ringan
   - Membuat Dataset Berbasis Window dengan metode windowed sequence yaitu dengan memanfaatkan 30 data terakhir (window size) sebagai input untuk memprediksi 1 data berikutnya.
    ```
    # Persiapkan data test
    test_data = df[comodity_list[comodity_selected]][-test_size-30:]
    test_data = scaler.transform(test_data.values.reshape(-1,1))
    
    # Membulatkan hingga 6 angka di belakang koma
    test_data = np.round(test_data, 6)
    
    # Siapkan Variable untuk menampung data Test, disesuaikan dengan data loopback yaitu 30
    X_test = []
    y_test = []
    
    for i in range(window_size, len(test_data)):
        X_test.append(test_data[i-30:i, 0])
        y_test.append(test_data[i, 0])
    ```

6. Konversi ke Format yang Diterima oleh TensorFlow
   - Data input dan target yang semula berupa list dikonversi ke dalam format numpy array, dan disesuaikan bentuk dimensinya agar dapat digunakan sebagai input dalam pelatihan model Deep Learning.
    ```
    X_train = np.array(X_train)
    X_test  = np.array(X_test)
    y_train = np.array(y_train)
    y_test  = np.array(y_test)
    
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test  = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    y_train = np.reshape(y_train, (-1,1))
    y_test  = np.reshape(y_test, (-1,1))
    ```

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

Model yang dikembangkan dievaluasi menggunakan metrik MSE, RMSE, MAPE, dan Accuracy, baik dalam bentuk data ternormalisasi maupun sudah didenormalisasi.

hasil evaluasi menunjukan performa yang cukup baik dengan nilai sebagai berikut (data masih dalam bentuk normalisasi) : 

```
Test Loss (MSE): 0.0015037448611110449
Test RMSE: 0.038778112321401444
Test MAPE: 0.06523878888823192
Test Accuracy: 0.934761211111768
```

hasil evaluasi menunjukan performa yang cukup baik dengan nilai sebagai berikut (data sudah didenormalisasi agar tercermin hasil yang lebih nyata) : 

```
Test MSE on denormalized data: 3911326.8858
Test RMSE on denormalized data: 1977.7074
Test MAPE on denormalized data: 2.701%
Test Accuracy on denormalized data: 97.29%
```


**Kesimpulan**

Model prediksi harga komoditas berbasis LSTM-GRU berhasil dibangun dan menunjukkan performa yang sangat baik, dengan tingkat akurasi tinggi, ditandai oleh nilai MAPE sebesar 2.701% setelah denormalisasi. Hasil ini menunjukkan bahwa model mampu memprediksi harga dengan kesalahan yang sangat kecil.

Model ini tidak hanya akurat, tetapi juga praktis dan siap digunakan sebagai alat bantu dalam mendukung pengambilan keputusan berbasis data (data-driven decision making). Dengan adanya prediksi yang presisi, pemerintah dan dinas terkait dapat melakukan intervensi lebih dini untuk menjaga stabilitas harga, mengawasi distribusi, mengendalikan pasokan, dan mengantisipasi fluktuasi harga pangan di pasar. namun perlu dan patut diingat, pemantauan terhadap performa model pun harus dilakukan secara berkala agar model senantiasa dalam performa terbaiknya.
