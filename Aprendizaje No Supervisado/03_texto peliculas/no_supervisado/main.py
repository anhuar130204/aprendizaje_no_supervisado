import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import re

# Descargar recursos de NLTK
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')  # Recurso adicional necesario para español

# Cargar datos (sin etiquetas de sentimiento)
try:
    df = pd.read_csv("peliculas.csv", encoding="utf-8")
except FileNotFoundError:
    print("Error: No se encontró el archivo 'peliculas.csv'")
    exit()
    
df['comentario'] = df['comentario'].astype(str)

# Preprocesamiento de texto
def preprocesar_texto(texto):
    # Eliminar caracteres especiales y convertir a minúsculas
    texto = re.sub(r'[^\w\s]', '', texto.lower())
    # Tokenizar y eliminar stopwords
    palabras = word_tokenize(texto)  # Eliminamos el parámetro language='spanish'
    stop_words = set(stopwords.words('spanish'))
    palabras = [palabra for palabra in palabras if palabra not in stop_words and len(palabra) > 2]
    return ' '.join(palabras)

df['comentario_limpio'] = df['comentario'].apply(preprocesar_texto)

# Vectorización con TF-IDF
vectorizer = TfidfVectorizer(max_features=2000)
X = vectorizer.fit_transform(df['comentario_limpio'])

# Método del codo para encontrar el número óptimo de clusters
inercia = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inercia.append(kmeans.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(range(1, 10), inercia, marker='o')
plt.title('Método del Codo para K-Means')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inercia')
plt.show()

# Aplicar K-Means con el k óptimo (ej: k=3 para positivo/neutral/negativo)
k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(X)
df['cluster'] = clusters

# Visualización con PCA (reducción a 2D)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X.toarray())

plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['cluster'], palette='viridis', s=100)
plt.title('Agrupamiento de Comentarios (K-Means)')
plt.xlabel('Componente PCA 1')
plt.ylabel('Componente PCA 2')
plt.legend(title='Cluster')
plt.show()

# Analizar los clusters
for i in range(k):
    print(f"\nCluster {i}:")
    muestra = df[df['cluster'] == i]['comentario'].sample(3, random_state=42)
    for comentario in muestra:
        print(f" - {comentario}")

# Interpretación manual (asignar etiquetas según las palabras más frecuentes)
from collections import Counter
for i in range(k):
    palabras = ' '.join(df[df['cluster'] == i]['comentario_limpio']).split()
    frecuencias = Counter(palabras).most_common(10)
    print(f"\nPalabras más frecuentes en Cluster {i}:")
    for palabra, frecuencia in frecuencias:
        print(f"{palabra}: {frecuencia}")
