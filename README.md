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
  03_texto peliculas/
    no_supervisado/
      main.py
      peliculas.csv
    modelo_entrenado.pkl
    modelo_peliculas.pkl
    peliculas.csv
    peliculas_entrenamiento.py
    vectorizador_sentimientos.pkl
clasificacion/
    train.py
redes/
    train.py
```

## Requisitos

Instala las dependencias necesarias ejecutando:

```sh
pip install -r requirements.txt"
```
Para el análisis de texto, también necesitas:
```sh
pip install pandas nltk scikit-learn joblib
```

## Aprendizaje No Supervisado
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
#### Ejecución

```sh
cd "Aprendizaje No Supervisado/02_texto"
python noticias_entrnamiento.py
```

## Aprendizaje Supervisado

En la carpeta [`clasificacion`](/clasificacion):

- [`train.py`](/clasificacion/train.py):
## Aprendizaje de sentimientos peliculas supervisado
En la carpeta [`Apremdizaje no supervisado`](/Aprendizaje No Supervisado/03_texto peliculas):
- [`peliculas_entrenamiento.py`]_(/Aprendizaje No Supervisado/03_texto peliculas/peliculas_entrenamiento.py):
## Aprendizaje de sentimientos peliculas no supervisado
En la carpeta [`Apremdizaje no supervisado`](/Aprendizaje No Supervisado/03_texto peliculas/no_supervisado):
- [`main.py`]_(/Aprendizaje No Supervisado/03_texto peliculas/main.py):



## Notas
- Asegúrate de tener Python 3.7+ instalado.
