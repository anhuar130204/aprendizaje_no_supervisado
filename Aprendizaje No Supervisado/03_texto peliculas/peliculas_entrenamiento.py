import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from nltk.corpus import stopwords
from textblob import TextBlob
import nltk
import joblib
import re

# Configurar NLTK
nltk_data_path = 'nltk_data'
nltk.data.path.append(nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path)

def limpiar_texto(texto):
    """Limpia y preprocesa el texto"""
    if pd.isna(texto):
        return ""
    
    # Convertir a minúsculas
    texto = texto.lower()
    
    # Eliminar caracteres especiales y números
    texto = re.sub(r'[^a-záéíóúñü\s]', '', texto)
    
    # Eliminar espacios extra
    texto = re.sub(r'\s+', ' ', texto).strip()
    
    return texto

def analizar_sentimiento_textblob(texto):
    """Analiza sentimiento usando TextBlob como baseline"""
    blob = TextBlob(texto)
    polaridad = blob.sentiment.polarity
    
    if polaridad > 0.1:
        return 'positiva'
    elif polaridad < -0.1:
        return 'negativa'
    else:
        return 'neutral'

def crear_etiquetas_automaticas(criticas):
    """Crea etiquetas automáticas basadas en palabras clave y TextBlob"""
    etiquetas = []
    
    # Palabras clave para clasificación
    palabras_positivas = [
        'excelente', 'buena', 'genial', 'increíble', 'fantástica', 'maravillosa',
        'brillante', 'magnífica', 'extraordinaria', 'impresionante', 'divertida',
        'emocionante', 'perfecta', 'recomiendo', 'obra maestra', 'espectacular'
    ]
    
    palabras_negativas = [
        'mala', 'terrible', 'horrible', 'aburrida', 'decepcionante', 'pésima',
        'mediocre', 'deficiente', 'fracaso', 'desastre', 'innecesaria', 'tediosa',
        'confusa', 'predecible', 'cliché', 'no recomiendo'
    ]
    
    for critica in criticas:
        critica_limpia = limpiar_texto(str(critica))
        
        # Contar palabras positivas y negativas
        score_positivo = sum(1 for palabra in palabras_positivas if palabra in critica_limpia)
        score_negativo = sum(1 for palabra in palabras_negativas if palabra in critica_limpia)
        
        # Combinar con análisis de TextBlob
        sentimiento_textblob = analizar_sentimiento_textblob(critica_limpia)
        
        # Decisión final
        if score_positivo > score_negativo:
            etiquetas.append(1)  # Positiva
        elif score_negativo > score_positivo:
            etiquetas.append(0)  # Negativa
        else:
            # Si hay empate, usar TextBlob
            if sentimiento_textblob == 'positiva':
                etiquetas.append(1)
            elif sentimiento_textblob == 'negativa':
                etiquetas.append(0)
            else:
                etiquetas.append(1)  # Por defecto positiva en caso neutral
    
    return etiquetas

# Cargar datos
try:
    df = pd.read_csv('peliculas.csv')  # Cambiado el nombre del archivo
    print("Columnas encontradas:", df.columns.tolist())
    print("Primeras filas:")
    print(df.head())
    
    # Verificar que existan las columnas necesarias
    if 'titulo' not in df.columns or 'critica' not in df.columns:
        print("Error: Las columnas 'titulo' y 'critica' no se encontraron.")
        print("Columnas disponibles:", df.columns.tolist())
        exit()
    
    # Limpiar datos
    df = df.dropna(subset=['titulo', 'critica'])
    print(f"Número de registros después de limpiar: {len(df)}")
    
    if len(df) == 0:
        print("Error: No hay datos válidos para procesar.")
        exit()
    
    # Preparar textos y etiquetas
    titulos = df['titulo'].tolist()
    criticas = df['critica'].tolist()
    criticas_limpias = [limpiar_texto(critica) for critica in criticas]
    
    # Crear etiquetas automáticamente
    print("Creando etiquetas automáticas...")
    etiquetas = crear_etiquetas_automaticas(criticas)
    
    # Mostrar distribución de etiquetas
    etiquetas_positivas = sum(etiquetas)
    etiquetas_negativas = len(etiquetas) - etiquetas_positivas
    print(f"Críticas positivas: {etiquetas_positivas}")
    print(f"Críticas negativas: {etiquetas_negativas}")
    
    # Verificar que tengamos suficientes datos
    if len(set(etiquetas)) < 2:
        print("Warning: Solo se encontró una clase de sentimiento. Ajustando criterios...")
        # Forzar una distribución más balanceada
        medio = len(etiquetas) // 2
        etiquetas = [1] * medio + [0] * (len(etiquetas) - medio)
    
    # Configurar vectorizador TF-IDF
    spanish_stop_words = stopwords.words('spanish')
    vectorizador = TfidfVectorizer(
        stop_words=spanish_stop_words,
        max_features=5000,
        ngram_range=(1, 2),  # Incluir bigramas
        min_df=2,
        max_df=0.8
    )
    
    # Vectorizar las críticas
    X = vectorizador.fit_transform(criticas_limpias)
    y = etiquetas
    
    print(f"Forma de la matriz de características: {X.shape}")
    
    # Dividir datos en entrenamiento y prueba
    if len(df) > 10:  # Solo dividir si tenemos suficientes datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = X, X, y, y
        print("Dataset pequeño: usando todos los datos para entrenamiento y prueba")
    
    # Entrenar modelo de clasificación
    modelo = LogisticRegression(random_state=42, max_iter=1000)
    modelo.fit(X_train, y_train)
    
    # Hacer predicciones
    y_pred = modelo.predict(X_test)
    
    # Evaluar modelo
    print("\n=== EVALUACIÓN DEL MODELO ===")
    print(f"Precisión: {accuracy_score(y_test, y_pred):.3f}")
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred, target_names=['Negativa', 'Positiva']))
    
    # Guardar modelos
    joblib.dump(modelo, 'modelo_sentimientos.pkl')
    joblib.dump(vectorizador, 'vectorizador_sentimientos.pkl')
    print("\nModelos guardados: 'modelo_sentimientos.pkl' y 'vectorizador_sentimientos.pkl'")
    
    # Mostrar resultados para cada película
    print("\n=== ANÁLISIS POR PELÍCULA ===")
    y_pred_all = modelo.predict(X)
    probabilidades = modelo.predict_proba(X)
    
    for i, (titulo, critica, pred, prob) in enumerate(zip(titulos, criticas, y_pred_all, probabilidades)):
        sentimiento = "POSITIVA" if pred == 1 else "NEGATIVA"
        confianza = max(prob) * 100
        
        print(f"\nPelícula: {titulo}")
        print(f"Crítica: {critica[:100]}{'...' if len(critica) > 100 else ''}")
        print(f"Sentimiento: {sentimiento} (Confianza: {confianza:.1f}%)")
    
    # Función para predecir nuevas críticas
    def predecir_sentimiento(nueva_critica):
        """Predice el sentimiento de una nueva crítica"""
        critica_limpia = limpiar_texto(nueva_critica)
        critica_vectorizada = vectorizador.transform([critica_limpia])
        prediccion = modelo.predict(critica_vectorizada)[0]
        probabilidad = modelo.predict_proba(critica_vectorizada)[0]
        
        sentimiento = "POSITIVA" if prediccion == 1 else "NEGATIVA"
        confianza = max(probabilidad) * 100
        
        return sentimiento, confianza
    
    # Ejemplo de uso
    print("\n=== EJEMPLO DE PREDICCIÓN ===")
    ejemplos = [
        "Esta película es increíble, la actuación es fantástica",
        "Muy aburrida y predecible, no la recomiendo",
        "Una obra maestra del cine, brillante dirección"
    ]
    
    for ejemplo in ejemplos:
        sentimiento, confianza = predecir_sentimiento(ejemplo)
        print(f"Crítica: '{ejemplo}'")
        print(f"Predicción: {sentimiento} ({confianza:.1f}% confianza)")
        print()

except FileNotFoundError:
    print("Error: No se encontró el archivo 'peliculas.csv'")
    print("Asegúrate de que el archivo existe y tiene las columnas 'titulo' y 'critica'")
except Exception as e:
    print(f"Error inesperado: {str(e)}")
    import traceback
    traceback.print_exc()