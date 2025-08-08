import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
import nltk
import joblib

nltk_data_path = 'nltk_data'
nltk.data.path.append(nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path)


df = pd.read_csv('noticias.csv')
print(df.columns)
titulos_noticias = df['Noticia'].dropna().tolist()

spanish_stop_words = stopwords.words('spanish')

vectorizaciones = TfidfVectorizer(stop_words=spanish_stop_words)
x = vectorizaciones.fit_transform(titulos_noticias)

modelo = KMeans(n_clusters=15, random_state=1234, n_init=10)
modelo.fit(x)

joblib.dump(modelo, 'modelo_entrenado.pkl')

print(f"clusteres: {modelo.labels_}")

for i, texto in enumerate(titulos_noticias):
    print(f"{texto} -> cluster {modelo.labels_[i]}")