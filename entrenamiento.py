import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 1. Importar el dataset con los datos de entrenamiento 
df_datos_clientes = pd.read_csv("clientes_entrenamiento.csv")

# 2. Convertir el dataframe a array de numpy
X = df_datos_clientes.values

# 3. Entrenar el modelo
modelo = KMeans(n_clusters=3, random_state=1234, n_init=10)
modelo.fit(X)

# 4. Análisis del modelo
df_datos_clientes['cluster'] = modelo.labels_
analisis = df_datos_clientes.groupby('cluster').mean()
print(analisis)

# 5. Exportar el modelo
joblib.dump(modelo, "modelo_segmentacion_clientes.pkl")

# 6. Graficar Clusters
centroides = modelo.cluster_centers_
etiquetas = modelo.labels_

# Separar los datos por clúster
cluster0 = X[etiquetas == 0]
cluster1 = X[etiquetas == 1]
cluster2 = X[etiquetas == 2]

# Colocar los puntos de cada clúster
plt.scatter(cluster0[:, 0], cluster0[:, 1], c="red", label='cluster0')
plt.scatter(cluster1[:, 0], cluster1[:, 1], c="blue", label='cluster1')
plt.scatter(cluster2[:, 0], cluster2[:, 1], c="green", label='cluster2')

# Colocar los centroides
plt.scatter(centroides[:, 0], centroides[:, 1], c="black", marker='X', s=200, label='Centroides')

# Colocar títulos y etiquetas
plt.title('Segmentación de clientes')
plt.xlabel('Gasto total')
plt.ylabel('Vistas')
plt.legend()
plt.grid(True)

# Guardar la gráfica
plt.savefig('graficas/clusters.png')
plt.show()