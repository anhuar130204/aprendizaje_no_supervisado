import 


modelo = joblib.load

datos_prueba = np.array([
    [50,3],
    [600,4]
])

clusters = modelo.predict(datos_prueba)
pritnt(clusters)