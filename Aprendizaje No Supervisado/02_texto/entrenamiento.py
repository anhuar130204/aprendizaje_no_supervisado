
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
import nltk
import joblib

nltk_data_path = 'nltk_data'
nltk.data.path.append(nltk_data_path)

nltk.download('stopwords', download_dir=nltk_data_path)

titulos_noticias = [
    "Avances en inteligencia artificial impulsan nuevas aplicaciones médicas",
    "Economías emergentes lideran crecimiento global en el segundo trimestre",
    "Investigadores desarrollan vacuna experimental contra el Alzheimer",
    "El mercado tecnológico supera expectativas en Wall Street",
    "Nuevos estudios vinculan sueño de calidad con mejor salud cardiovascular",
    "Gigantes tecnológicos apuestan por la computación cuántica",
    "La inflación en América Latina muestra señales de desaceleración",
    "Tecnología portátil mejora el monitoreo de pacientes con enfermedades crónicas",
    "El FMI ajusta al alza sus proyecciones para la economía global",
    "Dispositivos inteligentes permiten diagnóstico precoz de enfermedades respiratorias",
    "La industria de los semiconductores recibe inversiones millonarias",
    "Nuevas terapias genéticas prometen tratamientos personalizados más eficaces",
    "China y EE.UU. negocian medidas para estabilizar el comercio bilateral",
    "Estudio relaciona el uso excesivo de redes sociales con trastornos de ansiedad",
    "Europa invierte en energías limpias para revitalizar su economía",
    "Científicos desarrollan biosensores que detectan infecciones en tiempo real",
    "Mercado de criptomonedas se recupera tras semanas de volatilidad",
    "La telemedicina se consolida como opción preferida en zonas rurales",
    "Empresas fintech redefinen el acceso a servicios financieros",
    "Nuevas plataformas de educación digital incorporan inteligencia artificial"
]

spanis_stop_words = stopwords.words('spanish')

vectorizaciones = TfidfVectorizer(stop_words=spanis_stop_words)
x = vectorizaciones.fit_transform(titulos_noticias)

modelo = KMeans(n_clusters=4, random_state=1234, n_init=10)
modelo.fit(x)

joblib.dump(modelo, 'modelo_entrenado.pkl')
import joblib

#print(x)
print(f"clusteres: {modelo.labels_}")


for i, texto in enumerate(titulos_noticias):
    print(f"{texto} -> cluster {modelo.labels_[i]}")
    