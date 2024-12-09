# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

## Business Understanding

Jaya Jaya Maju merupakan salah satu perusahaan multinasional yang telah berdiri sejak tahun 2000. Ia memiliki lebih dari 1000 karyawan yang tersebar di seluruh penjuru negeri. 

Walaupun telah menjadi menjadi perusahaan yang cukup besar, Jaya Jaya Maju masih cukup kesulitan dalam mengelola karyawan. Hal ini berimbas tingginya attrition rate (rasio jumlah karyawan yang keluar dengan total karyawan keseluruhan) hingga lebih dari 10%.

Untuk mencegah hal ini semakin parah, manajer departemen HR ingin meminta bantuan Data Scientist mengidentifikasi berbagai faktor yang mempengaruhi tingginya attrition rate tersebut. Selain itu, ia juga meminta Data Scientist untuk membuat business dashboard untuk membantunya memonitori berbagai faktor tersebut.

### Permasalahan Bisnis

Terdapat beberapa hal yang bisa diambil dari latar belakang diatas khususnya bagian permasalahan utama adalah tingginya attrition rate atau rasio jumlah karyawan yang keluar dibanding dengan total keseluruhan yang ada. Melalui permasalahan utama tersebut dapat dibagi ke dalam beberapa bagian kecil untuk nantinya di analisis dalam menemukan jawaban atau kesimpulan yang mampu memberikan solusi dari permasalahan HR ini. Penyebab attrition rate bisa didukung oleh beberapa kemungkinan diantaranya

Main Problem : Tingginya Attrition Rate

- Pendapatan karyawan yang tidak sesuai dengan kompetensi
- Lingkungan masing-masing departemen yang berbeda
- Penerapan pelatihan karyawan yang memberikan dampak attrition
- Lokasi tinggal dengan tempat kerja yang aksesnya cukup jauh
- Jam kerja maupun waktu lembur yang tidak cocok
- Fasilitas yang tidak memadai

Sebelum ke tahapan pemahaman data HR perlu disusun beberapa problem statement untuk menentukan permasalahan dengan lebih jelas. Pertanyaan tersebut disusun untuk dapat menjawab solusi melalui insight data.

#### Problem Statement

Berikut beberapa pertanyaan yang disusun untuk dijadikan acuan dalam menggali insight data nantinya untuk mencari jawaban serta solusi yang dibutuhkan : 
- Bagaimana perbandingan karyawan attrition dengan non attrition?
- Bagaimana persebaran data karyawan Attrition secara usia?
- Bagaimana persebaran data karyawan Attrition pada tiap departemen?
- Bagaimana pengaruh MonthlyIncome terhadap Attrition?
- Bagaimana dampak TrainingTimesLastYear terhadap Attrition?
- Bagaimana pengaruh DistanceFromHome terhadap Attrition?
- Bagaimana hubungan StandardHours, HourlyRate dan Overtime terhadap Attrition?
- Bagaimana dampak EnvironmentSatisfaction terhadap Attrition? 
- Bagaimana hubungan antar parameter atau variabel di data HR?

#### Goals

Tujuan dari pembuatan proyek ini adalah sebagai berikut
- Mendapatkan insight melalui hubungan attrition dengan seluruh paramater atau variabel yang ada pada data HR.
- Membuat dashboard untuk dapat dianalisis secara visual disesuaikan dengan kebutuhan HR berdasarkan attririon.
- Menemukan jawaban serta memberikan solusi kepada HR terkait permasalahan attrition yang tinggi.
- Membuat sistem berbasis machine learning untuk menklasifikasikan karyawan yang memiliki potensi attrition tinggia dengan karyawan tidak. 

### Cakupan Proyek

Beberapa tahapan yang dilakukan pada proyek ini adalah sebagai berikut

1. Pengumpulan Data dan Pemahaman Data
   
   Tahapan ini terdiri dari beberapa bagian seperti import library yang dibutuhkan untuk proyek, import dataset karyawan sesuai ketentuan, memahami isi data melalui info() & describe(), mengecek nilai NaN atau NULL pada data, mengecek duplikat data serta mengecek data unik berupa kategori pada masing-masing kolom. Perlu dilakukan untuk melihat karakteristik dari tiap kolom bahkan termasuk kolom yang bertipe numerik untuk melihat nilai mean, min, max, standar deviasi dan kuartalnya.

2. Data Cleaning & Preparation

   Tahapan data preparation terbagi menjadi dua bagian yaitu data preparation untuk analisis dan data preparation untuk modeling. Hal ini dilakukan karena keduanya punya behaviour yang berbeda. 

   a. Data Preparation for Analysis

   Analisis digunakan untuk melihat pola kolom-kolom yang sekiranya berpengaruh dengan Attrition sehingga data yang digunakan merupakan data real sesuai yang ada di dataset. Pada analisis tahapan data preparation berfokus pada menghilangkan nilai NaN atau NULL dan penyesuaian nama kategori pada kolom.

   b. Data Preparation for Modeling

   Data prep untuk modeling dilakukan sebagai tahapan menyesuaikan dataset dengan environment model algoritma machine learning dalam membaca data. Beberapa aturan yang umum sebelum membuat model machine learning adalam membuat data yang dipunya mudah dibaca oleh sistem.

3. Exploratory Data Analysis berdasarkan problem statement yang disusun

   Menggunakan data preparation untuk analisis, data yang sebelumnya sudah diolah serta diseusaikan akan dibuat ke dalam bentuk visual tertentu misal bar chart, boxplot maupun histogram. Fokus dari EDA ini adalam membentuk visual yang menghubungkan data Attrition dengan variabel tertentu yang sesuai dengan pertanyan di awal problem statement. Hal tersebut dilakukan agar lingkup dari proyek tidak melebar kemana-mana dalam pembahasannya.

4. Pengembangan Sistem Klasifikasi Karyawan Attrition dan Non-Attrition berbasis Machine Learning

   Tahapan modeling menggunakan data preparation untuk modeling. Ada beberapa algoritma yang akan digunakan yaitu 
   - Algoritma K-Nearest Neighbor
   - Algoritma SVM
   - Algoritma Decision Tree
   - Algoritma Random Forest
   - Algoritma Boosting Algorithm
   
   Dari kelima algoritma diatas akan dipilih satu dengan nilai akurasi yang paling tinggi melalui tahao evaluasi. Kemudian hasil model dikonversi ke dalam bentuk .pkl yang akan digunakan pada sistem klasifikasi karyawan Attrition dan Non-Attrition.

5. Membuat Dashboard Analysis Karyawan Attrition menggunakan Metabase yang terintegrasi dengan Supabase

   Dashaboard analisis akan dibuat di metabase melalui docker. Dataset yang digunakan merupakan data hasil olahan pada bagian Data Preparation for Analysis. Dataset diupload ke supabase kemudian dikoneksikan ke metabase.


### Persiapan

Sumber data : [Dataset Karyawan Jaya Jaya Maju](https://github.com/dicodingacademy/dicoding_dataset/tree/main/employee)

Setup environment - notebook :

```
python -m venv main-ds
main-ds\Scripts\activate
pip install -r requirements.txt

Notebook dikerjakan di Google Collaboratory
a. numpy==1.26.4
b. pandas==2.2.2
c. matplotlib==3.8.0
d. seaborn==0.13.2
```

Setup environment - shell/terminal :

```
mkdir Proyek_Analisis_Solusi_Human_Resources
cd Proyek_Analisis_Solusi_Human_Resources
pipenv install
pipenv shell
pip install -r requirements.txt
```

#### Run Aplikasi Sistem Klasifikasi Karyawan Attrion

Sistem klasifikasi saat ini bisa dijalankan melalui Google Collaboratory pada cell dengan menjalankan kode program yang berada di file `prediction.py`. Atau bisa menjalankan di lokal dengan update kode upload dan kode model di sesuaikan

Kode Upload
```
# Membaca file CSV yang diupload
data_pred = pd.read_csv("/dataset/prediksi_employee.csv")
```

Kode Model
```
# Memuat model XGBoost yang telah dilatih sebelumnya
model_pred = pickle.load(open('/model/xgboost_model.pkl', 'rb'))
```

Jalankan di terminal
```bash
python prediction.py
```

## Business Dashboard

Dashboard yang dibuat merupakan hasil analisis dari EDA based on Question. Setidaknya beberapa variabel utama yang dibandingkan dengan variabel Attrition untuk menganalisis masing-masing persebaran datanya. Beberapa variabel tersebut diantaranya Age, MonthlyIncome, DistanceFromHome, OverTime, Education dan JobLevel. Terdapat filter untuk beberapa data yang disajikan secara total yaitu Filter Attrition untuk menyaring beberapa informasi berdasarkan karyawan Attrition dan Non-Attrition. Dashboard disajika seringkas mungkin agar mudah dibaca oleh tim HR.

![dashboard](https://raw.githubusercontent.com/Abito21/Proyek_Analisis_Solusi_Human_Resources/main/assets/dashboard.png)

## Conclusion

Melalui hasil analisis di tahapan EDA based on questions dapat disimpulkan beberapa alasan utama tinggianya Attrition karyawan diantaranya terlihat pada 5 variabel diantaranya  Age, MonthlyIncome, TrainingTimeLastYear, DistanceFromHome dan OverTime. 
- Karyawan Attrition yang kebanyakan didominasi oleh usia muda dari rentang 11 hingga 40 tahun dengan distribusi terbanyak jika dijumlahkan yaitu 136 karyawan. Untuk usia 41 hingga 60 tahun berjumlah 43 karyawan yang mana usia diatas 40 ke atas memiliki kemungkina yang disebabkan masa pensiun atau produktivitas yang sudah menurun.
- Tingkat Attrition tinggi bisa disebabkan gaji bulanan yang kemungkinan tidak kompetitif atau dibawah standar baik tidak sesuai departemen atau profesinya. Memang gaji merupakan rahasia kontrak masing-masing karyawan namun ada kemungkinan pola beban kerja yang dianggap tidak sesuai dengan gaji atau adanya sharing informasi antar karyawan mengenai gaji sehingga menurunnya keinginan karyawan berada di perusahaan. Atau kemungkinan lain terdapat tawaran yang lebih tinggi diperusahaan lain.
- Karyawan Attrition kebanyakan didominasi oleh karyawan yang tidak mengikuti pelatihan atau hanya mengikuti sekali dalam setahun terakhir. Presentasenya lebih banyak ketimbang yang mengikuti pelatihan setahun terakhir 2 hingga 6 kali. Hal ini membuktikan bahwa pelatihan kepada karyawan memberikan ruang untuk mengimprove kemampuan serta meningkatkan kinerja karyawan sejalan dengan keinginan karyawan bekerja diperusahaan dengan nyaman.
- Karyawan Attrition juga banyak di dominasi dengan data dimana DistanceFromHome atau jarak tinggal dengan kantor yang cukup jauh. Jarak yang jauh bisa diidentifikasi beberapa hal terkait energi karyawan untuk ke kantor dan pulang, kemudian transportasi yang kurang memadai, keadaan jalur menuju kantor yang tidak mendukung seperti adanya kemacetan-keramaian dan bisa banyak hal lain memungkinkan tingkat jenuh karyawan meningkat.
- Terakhir karyawan Attrition secara presentase di dominasi oleh karyawan yang menjalankan OverTime atau lembur. Waktu lembur yang barangkali jadwalnya tidak selalu pasti kemudian kemungkinan banyak OverTime yang diambil oleh karyawan bisa menjadi penyebab karyawan Attrition.

Selain dari variabel yang sudah dianalisis melalui korelasi variabel Attrition dengan variabel lain ditemukan beberapa korelasi yang kuat pada variabel Education, JobLevel dan TotalWorkingYears. 

Melalui model machine learning yang dibuat dari kelima algoritma terdapat algoritma terbaik dengan nilai akurasi tinggi yaitu XGBoost. Model machine learning dibuat untuk pada mengklasifikasikan karyawan yang memiliki potensi Attrition dan Non-Attrition sehingga bisa diambil tindakan preventif awal agar Attrition dapat dikurangi. Akurasi yang didapatkan cukup stabil menggunakan XGBoost dari training dan testing sehingga sistem prediksi mampu memberikan nilai yang mendekati klasifikasi yang baik.

### Rekomendasi Action Items (Optional)

Terdapat beberapa hal yang dapat dilakukan oleh tim HR agar dapat meminimalisir terjadinya Attrition yang tinggi diantaranya

- Melalui pelatihan untuk mengajak atau memberikan pemberitahuan kepada karyawan pentingnya pelatihan di kalangan karyawan. Bukan menuntut namun memberikan fasilitas pelatihan yang baik di setiap departemen agar karyawan tertarik mengikuti. Selain itu mengadakan pelatihan yang penting maupun sertifikasi kepada karyawan untuk meningkatkan kinerja di sisi lain sebagai bentuk penghargaan karyawan terlibat dalam perusahaan.
- Memberikan fasilitas berupa shuttle kepada karyawan apabila memungkinkan dan terdapat budget untuk hal itu. Karyawan yang memiliki lokasi tinggal jauh atau sulit di akses oleh kendaraan umum dapat difasilitasi perusahaan agar mudah untuk mencapai kantor.
- Mengatur jadwal Overtime atau lembur dengan lebih baik minimal 3-7 hari sebelum lembur dilaksanakan tiap divisi perlu mengkomunikasikan Overtime baik berupa alasan perlunya Overtime sehingga perlu mendapatkan persetujuan terlebih dahulu di tingkat manajemen. Sehingga jawal lembur lebih bisa dikontrol dengan baik selain itu karyawan yang lembur sebisa mungkin bukan tertuju pada karyawan itu itu saja namun sebisa mungkin di rolling atau karyawan yang memiliki waktu luang jadi perlu ditanyakan kepada karyawan yang bersangkutan apakah di waktu Overtime yang ditentukan terdapat kegiatan atau tidak.