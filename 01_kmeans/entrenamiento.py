import numpy as np 
import pandas as pd 
import joblib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 1. Importar el Dataset con los datos de entrenamiento
df_datos_clientes = pd.read_csv("clientes_entrenamiento.csv")
#print(df_datos_clientes.info())
#print(df_datos_clientes.head())

# 2. Convertir el Dataframe a un Array de Numpy
x = df_datos_clientes.values
#print(x)

# 3. Entrenar el modelo
modelo = KMeans(n_clusters=3, random_state=1234, n_init=10)
modelo.fit(x)

# 4. Analisis del modelo
df_datos_clientes["cluster"] = modelo.labels_
analisis = df_datos_clientes.groupby("cluster").mean()
print(analisis)

# 5. Exportar el modelo

joblib.dump(modelo, "modelo_segmentacion_clientes,pkl")

# 6. Graficar los clusters

centroides = modelo.cluster_centers_
etiquetas = modelo.labels_

cluster0 = x[etiquetas == 0]
cluster1 = x[etiquetas == 1]
cluster2 = x[etiquetas == 2]

# Colocar los puntos de cada cluster
plt.scatter(cluster0[:,0],cluster0[:,1], c='red', label='Clientes Temporada 2')
plt.scatter(cluster1[:,0],cluster1[:,1], c='blue', label='Clientes VIP 1')
plt.scatter(cluster2[:,0],cluster2[:,1], c='green', label='Clientes Ofertas 0')

# Colocar los centroides de cada cluster
plt.scatter(centroides[:,0],centroides[:,1], c='black', label='Centroides')

# Colocar titulos y etiquetas
plt.title('Segmentacion de clientes')
plt.xlabel('Gasto total')
plt.ylabel('Visitas')
plt.legend()
plt.grid(True)

plt.savefig('graficas/clusters.png')