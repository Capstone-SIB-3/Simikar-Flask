import numpy as np
import pandas as pd
from sklearn.utils import resample 
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

jobs = pd.read_csv('./model/techloker.csv', encoding = 'ISO-8859-1')

jobs.drop('link',axis=1,inplace= True)
jobs.drop('nama_perusahaan',axis=1,inplace= True)
jobs.drop('lokasi_perusahaan',axis=1,inplace= True)

# Membersihkan missing value dengan fungsi dropna()
jobs_clean = jobs.dropna()

# Mengurutkan kemampuan berdasarkan kemampuan kemudian memasukkannya ke dalam variabel fix_jobs
fix_jobs = jobs_clean.sort_values('kemampuan', ascending=True)

# Membuang data duplikat pada variabel preparation
preparation_jobs = fix_jobs.drop_duplicates('kemampuan')

# Mengonversi data series ‘jenis_pekerjaan’ menjadi dalam bentuk list
jenis_jobs = preparation_jobs['jenis_pekerjaan'].tolist()

# Mengonversi data series ‘kemampuan’ menjadi dalam bentuk list
kemampuan_jobs = preparation_jobs['kemampuan'].tolist()

# Membuat dictionary untuk data ‘jenis_pekerjaan’, ‘kemampuan’
jobs_new = pd.DataFrame({
    'Jenis_Pekerjaan': jenis_jobs,
    'Kemampuan': kemampuan_jobs
})

jobs_recomend = jobs_new

from sklearn.feature_extraction.text import TfidfVectorizer

# Inisialisasi TfidfVectorizer
tf = TfidfVectorizer()

# Melakukan perhitungan idf pada data kemampuan 
tf.fit(jobs_recomend['Kemampuan'])


# Mapping array dari fitur index integer  ke fitur nama 
tf.get_feature_names_out()

# Melakukan fit lalu ditransformasikan ke bentuk matriks 
tfidf_matrix = tf.fit_transform(jobs_recomend['Kemampuan'])

# Mengubah vektor tf-idf dalam bentuk matriks dengan fungsi todense()
tfidf_matrix.todense()

# Membuat dataframe untuk melihat tf-idf matrix
# Kolom diisi dengan jenis get_names
# Baris diisi dengan nama Kemampuan

pd.DataFrame(
    tfidf_matrix.todense(), 
    columns=tf.get_feature_names_out(),
    index=jobs_recomend.Jenis_Pekerjaan
).sample(10, axis=1).sample(10, axis=0)

from sklearn.metrics.pairwise import cosine_similarity

# Menghitung cosine similarity pada matrix tf-idf
cosine_sim = cosine_similarity(tfidf_matrix)

# Membuat dataframe dari variabel cosine_sim dengan baris dan kolom berupa movie
cosine_sim_df = pd.DataFrame(cosine_sim, index=jobs_recomend['Kemampuan'], columns=jobs_recomend['Jenis_Pekerjaan'])
def jobs_recommendations(Kemampuan, similarity_data=cosine_sim_df, items=jobs_recomend[['Jenis_Pekerjaan', 'Kemampuan']], k=10):
    # Mengambil data dengan menggunakan argpartition untuk melakukan partisi secara tidak langsung sepanjang sumbu yang diberikan    
    index = similarity_data.loc[Kemampuan].to_numpy().argpartition(
        range(-1,-k,-1))
    # Mengambil data dengan similarity terbesar dari index yang ada
    closest = similarity_data.columns[index[-1:-(k+2):-1]]
    # Drop kemampuan agar kemampuan yang dicari tidak muncul dalam daftar rekomendasi
    closest = closest.drop(Kemampuan, errors='ignore')
    return pd.DataFrame(closest).merge(items).head(k)

jobs_recommendations('Management, Microsoft, Communication Skills')

rekomendasi = pd.DataFrame(jobs_recommendations('Laravel, PHP, Vue.js'))
print(rekomendasi['Jenis_Pekerjaan'][0])