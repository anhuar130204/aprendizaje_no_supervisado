# Análisis Supervisado y No Supervisado

Este repositorio contiene ejemplos prácticos de aprendizaje supervisado y no supervisado utilizando Python y librerías populares como scikit-learn, pandas y matplotlib.

## Estructura del repositorio

```
Aprendizaje No Supervisado/
  01_kmeans/
    agrupacion.py
    clientes_entrenamiento.csv
    entrenamiento.py
    modelo_segmentacion_clientes.pkl
    requirements.txt
    graficas/
      clusters.png
  02_texto/
    noticias.csv
    noticias_entrnamiento.py
    modelo_entrenado.pkl
clasificacion/
    train.py
```

## Requisitos

Instala las dependencias necesarias ejecutando:

```sh
pip install -r "Aprendizaje No Supervisado/01_kmeans/requirements.txt"
```
Para el análisis de texto, también necesitas:
```sh
pip install pandas nltk scikit-learn joblib
```

## Aprendizaje No Supervisado

### Segmentación de clientes (KMeans)

En la carpeta [`Aprendizaje No Supervisado/01_kmeans`](Aprendizaje%20No%20Supervisado/01_kmeans):

- [`entrenamiento.py`](Aprendizaje%20No%20Supervisado/01_kmeans/entrenamiento.py): Entrena un modelo KMeans para segmentar clientes y guarda el modelo y una gráfica de los clusters.
- [`agrupacion.py`](Aprendizaje%20No%20Supervisado/01_kmeans/agrupacion.py): Usa el modelo entrenado para predecir el cluster de nuevos datos.
- [`clientes_entrenamiento.csv`](Aprendizaje%20No%20Supervisado/01_kmeans/clientes_entrenamiento.csv): Datos de entrenamiento.

#### Ejecución

Para entrenar el modelo y generar la gráfica:

```sh
cd "Aprendizaje No Supervisado/01_kmeans"
python entrenamiento.py
```

Para predecir el cluster de nuevos clientes:

```sh
python agrupacion.py
```

### Agrupamiento de noticias por texto

En la carpeta [`Aprendizaje No Supervisado/02_texto`](Aprendizaje%20No%20Supervisado/02_texto):

- [`noticias.csv`](Aprendizaje%20No%20Supervisado/02_texto/noticias.csv): Archivo con noticias (solo columna `Noticia`).
- [`noticias_entrnamiento.py`](Aprendizaje%20No%20Supervisado/02_texto/noticias_entrnamiento.py): Agrupa noticias por similitud de texto usando KMeans y muestra los resultados ordenados por cluster.

#### Ejecución

```sh
cd "Aprendizaje No Supervisado/02_texto"
python noticias_entrnamiento.py
```

## Aprendizaje Supervisado

En la carpeta [`clasificacion`](ejemplos/clasificacion):

- [`train.py`](ejemplos/clasificacion/train.py): Ejemplo de clasificación usando regresión logística.

### Ejecución

```sh
cd clasificacion
python train.py
```

## Notas

- Los modelos y gráficos generados se guardan automáticamente en sus respectivas carpetas.
- Asegúrate de tener Python 3.7+ instalado.