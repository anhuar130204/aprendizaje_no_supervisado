import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump, load

data = {
    "tiempo_minutos": [5, 8, 12, 2, 20, 15, 3, 12, 45, 10],
    "paginas_visitadas": [3, 5, 7, 1, 10, 8, 2, 6, 8, 3],
    "compro": [0, 0, 1, 0, 1, 1, 0, 0, 1, 0]
}

df = pd.DataFrame(data)

x = df[["tiempo_minutos", "paginas_visitadas"]]
y = df["compro"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

model = LogisticRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("Predicciones:", y_pred)
print("Entrada de prueba:", x_test.values)

# Corrección aquí:
print(f"Precisión del modelo: {accuracy_score(y_test, y_pred):.2%}")
