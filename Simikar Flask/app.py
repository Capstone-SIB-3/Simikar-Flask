from sklearn.feature_extraction.text import TfidfVectorizer
import flask
import difflib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = flask.Flask(__name__, template_folder='templates')

import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd

jobs = pd.read_csv('./model/techloker.csv', encoding='ISO-8859-1')
jobs.drop('link', axis=1, inplace=True)
jobs.drop('nama_perusahaan', axis=1, inplace=True)
jobs.drop('lokasi_perusahaan', axis=1, inplace=True)
jobs_clean = jobs.dropna()
jobs_clean.head(10)
fix_jobs = jobs_clean.sort_values('kemampuan', ascending=True)

preparation_jobs = fix_jobs.drop_duplicates('kemampuan')
jenis_jobs = preparation_jobs['jenis_pekerjaan'].tolist()
kemampuan_jobs = preparation_jobs['kemampuan'].tolist()

jobs_new = pd.DataFrame({
    'Jenis_Pekerjaan': jenis_jobs,
    'Kemampuan': kemampuan_jobs
})
jobs_recomend = jobs_new

tf = TfidfVectorizer()
tf.fit(jobs_recomend['Kemampuan'])
tf.get_feature_names()
tfidf_matrix = tf.fit_transform(jobs_recomend['Kemampuan'])
tfidf_matrix.todense()

pd.DataFrame(
    tfidf_matrix.todense(),
    columns=tf.get_feature_names(),
    index=jobs_recomend.Jenis_Pekerjaan
).sample(10, axis=1).sample(10, axis=0)

cosine_sim = cosine_similarity(tfidf_matrix)
cosine_sim_df = pd.DataFrame(
    cosine_sim, index=jobs_recomend['Kemampuan'], columns=jobs_recomend['Jenis_Pekerjaan'])
cosine_sim_df.sample(10, axis=1).sample(10, axis=0)


def jobs_recommendations(Kemampuan, similarity_data=cosine_sim_df, items=jobs_recomend[['Jenis_Pekerjaan', 'Kemampuan']], k=10):
    # Mengambil data dengan menggunakan argpartition untuk melakukan partisi secara tidak langsung sepanjang sumbu yang diberikan
    index = similarity_data.loc[Kemampuan].to_numpy().argpartition(
        range(-1, -k, -1))
    # Mengambil data dengan similarity terbesar dari index yang ada
    closest = similarity_data.columns[index[-1:-(k+2):-1]]
    # Drop kemampuan agar kemampuan yang dicari tidak muncul dalam daftar rekomendasi
    closest = closest.drop(Kemampuan, errors='ignore')
    return pd.DataFrame(closest).merge(items).head(k)

@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('index.html'))
    if flask.request.method == 'POST':
        m_name = flask.request.form['kemampuan']
        m_name = m_name.title()
        result_final = jobs_recommendations(m_name)
        return result_final['Jenis_Pekerjaan'][0]
            
if __name__ == '__main__':
    app.run()