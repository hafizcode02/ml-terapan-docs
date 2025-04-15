# Laporan Proyek Machine Learning - Hafiz Caniago

## Domain Proyek

Proyek ini mengangkat tema untuk memprediksi harga komoditas pangan yang ada di pasar Kramat Kota Cirebon. tema ini diangkat karena masalah kenaikan harga yang sangat fluktuatif untuk komoditas pangan yang ada sehingga pihak terkait dapat mempersiapkan langkah yang lebih awal untuk mencegah kenaikan harga yang tidak terkendali. proyek ini melakukan pendekatan prediksi menggunakan deep learning dan kombinasi algoritma LSTM-GRU untuk membuat model prediksi, kombinasi tersebut dipilih setelah melakukan beberapa tinjauan pada jurnal. pada penelitian ini dataset yang digunakan ada 5 data harga komoditas pangan dengan tingkat persentase kenaikan tertinggi, namun pada dokumentasi ini, saya hanya menggunakan 1 buah komoditas saja untuk mensimplifikasi dokumentasi.

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


Dapat dilihat bahwa di antara komoditas-komoditas tersebut, cabai merah besar mengalami kenaikan tertinggi sebesar 596,2%, diikuti cabai merah keriting (517,6%), cabai rawit hijsau (488,2%), cabai rawit merah (389,8%), dan bawang merah (268,4%). 

Berbagai upaya telah dilakukan oleh pihak terkait untuk mengendalikan fluktuasi harga, salah satunya melalui inspeksi mendadak (sidak) pasar. Sidak pasar merupakan kegiatan pemeriksaan kondisi pasar secara langsung dan tanpa pemberitahuan sebelumnya oleh pejabat pemerintah, biasanya dari Dinas Perdagangan atau instansi terkait lainnya. Kegiatan ini bertujuan untuk memastikan stabilitas harga komoditas pangan.

Namun, dalam praktiknya, sidak pasar seringkali dilakukan hanya pada waktu-waktu tertentu atau setelah adanya laporan mengenai lonjakan harga yang tidak wajar. Hal ini menyebabkan keterlambatan dalam pengambilan tindakan lanjutan, seperti pengeluaran cadangan pangan, sehingga harga komoditas pangan terus meningkat.

### Problem Statements

Poin Permasalahan :
- Kenaikan harga komoditas pangan yang sangat fluktuatif dan tidak dapat diprediksi.
- Keterlambatan penanganan tindak lanjut dari pihak terkait atas kenaikan harga komoditas pangan tertentu, sehingga harga melonjak secara tiba tiba dan tidak terkendali.

### Goals
Tujuan dilakukan nya penelitian :
- Untuk mencegah lonjakan harga komoditas dengan melakukan antisipasi dini terhadap potensi kenaikan harga berdasarkan hasil prediksi, sehingga dapat mempersiapkan langkah-langkah pengendalian yang diperlukan oleh pihak terkait.

### Solution
- Melakukan pelatihan model untuk memprediksi harga komoditas pangan, pendekatan algoritma yang digunakan adalah algoritma LSTM-GRU. hyperparameter tuning juga dilakuan

Semua poin di atas harus diuraikan dengan jelas. Anda bebas menuliskan berapa pernyataan masalah dan juga goals yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**:
- Menambahkan bagian “Solution Statement” yang menguraikan cara untuk meraih goals. Bagian ini dibuat dengan ketentuan sebagai berikut: 

    ### Solution statements
    - Mengajukan 2 atau lebih solution statement. Misalnya, menggunakan dua atau lebih algoritma untuk mencapai solusi yang diinginkan atau melakukan improvement pada baseline model dengan hyperparameter tuning.
    - Solusi yang diberikan harus dapat terukur dengan metrik evaluasi.

## Data Understanding
Paragraf awal bagian ini menjelaskan informasi mengenai data yang Anda gunakan dalam proyek. Sertakan juga sumber atau tautan untuk mengunduh dataset. Contoh: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Restaurant+%26+consumer+data).

Selanjutnya uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:  

### Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:
- accepts : merupakan jenis pembayaran yang diterima pada restoran tertentu.
- cuisine : merupakan jenis masakan yang disajikan pada restoran.
- dst

**Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data atau exploratory data analysis.

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.
