# Laporan Proyek Machine Learning

### Nama : Rehan

### Nim :

### Kelas : Malam B

## Domain Proyek

Zaman sekarang dimana teknologi dan informasi sudah berkembang, kebutuhan akan gadget pintar sangat meningkat pesat sehingga banyak produsen gadget
yang membuat berbagai macam gadget pintar. Dengan banyaknya varian dan jenis-jenis gadget pintar juga dengan harga yang sangat bervariasi, maka konsumen
perlu lebih selektif dalam memilih gadget pintar yang akan dibelinya, khususnya dibagian harga. Apakah dengan harga yang telah dipatok sudah sesuai dengan kebutuhan
yang diperlukan atau tidak.

## Business Understanding

Pada bagian ini, kamu perlu menjelaskan proses klarifikasi masalah.

Bagian laporan ini mencakup:

### Problem Statements

- Banyaknya varian gadget pintar sehingga membuat konsumen kebingungan untuk memilih.
- Konsumen kerap membeli gadget pintar yang terlalu mahal dan tidak sesuai kebutuhannya.

### Goals

- Solusi untuk konsumen agar ketika ingin membeli gadget pintar, dapat memperkirakan harganya terlebih dahulu, sesuai dengan kebutuhan yang ingin dipenuhi.

Semua poin di atas harus diuraikan dengan jelas. Anda bebas menuliskan berapa pernyataan masalah dan juga goals yang diinginkan.

### Solution statements

- Membangun suatu sistem yang dapat mempelajari suatu data (Machine Learning) terkait jenis dan harga untuk melakukan estimasi harga gadget pintar
- Sistem berjalan dengan menggunakan metode Regresi Linear yang dinilai cocok untuk melakukan estimasi.

## Data Understanding

Dataset yang digunakan berasal dari situs Kaggle yang berisi data harga dan spesifikasi gadget pintar. Dataset ini mengandung 407 entries dan 8 columns<br>

Contoh: [Mobile Phone Price](https://www.kaggle.com/datasets/rkiattisak/mobile-phone-price).

### Variabel-variabel yang terdapat pada Dataset adalah sebagai berikut:

- Brand = Adalah perusahaan yang mem-produksi handphone tersebut
- Model = Nama unik dari suatu seri produksi handphone
- Storage = Ukuran penyimpanan/memory internal dari handphone tersebut
- RAM = Berfungsi sebagai tempat penyimpanan df sementara dan hanya bekerja saat perangkat tersebut hidup
- Screen Size = Ukuran layar dari handphone tersebut, dijelaskan dalam ukuran inches
- Camera = Resolusi dan jumlah kamera dari handphone tersebut, dijelaskan dalam ukuran MegaPixel
- Battery Capacity = Ukuran/kapasitas batterai dari handphone tersebut, dijelaskan dalam ukuran mAh
- Price = Harga dari perangkat handphone tersebut

## Data Preparation

Pertama-tama mari import semua library yang dibutuhkan,

```bash
import pandas as pd
import numpy as np
import matplotlib.pypot as plt
import seaborn as sns
```

Setelah itu kita akan men-definsikan dataset menggunakan fungsi pada library pandas

```bash
df = pd.read_csv('Mobile phone price.csv')
```

Lalu kita akan melihat informasi mengenai dataset dengan syntax seperti dibawah:

```bash
df.info()
```

Dengan hasil sebagai berikut:
![df.info](dfinfo.jpg) <br>

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

## Deployment

pada bagian ini anda memberikan link project yang diupload melalui streamlit share. boleh ditambahkan screen shoot halaman webnya.

**---Ini adalah bagian akhir laporan---**

_Catatan:_

- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.
