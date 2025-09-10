import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import unicodedata
import re

st.title("TF-IDF (con stopwords en español y preprocesado)")

st.write("Cada línea = un documento (frase, párrafo, artículo...). Este demo normaliza, quita acentos y remueve stopwords comunes en español.")

text_input = st.text_area(
    "Escribe tus documentos (uno por línea):",
    "El perro ladra fuerte.\nEl gato maúlla en la noche.\nEl perro y el gato juegan juntos."
)

# Lista de stopwords en español (ejemplo razonable, puedes ampliar)
SPANISH_STOPWORDS = {
    "de","la","que","el","en","y","a","los","del","se","las","por","un","para","con",
    "no","una","su","al","lo","como","más","pero","sus","le","ya","o","este","sí",
    "porque","esta","entre","cuando","muy","sin","sobre","también","me","hasta","hay",
    "donde","quien","desde","todo","nos","durante","todos","uno","les","ni","contra",
    "otros","ese","eso","ante","ellos","e","esto","mí","antes","algunos","qué","unos",
    "yo","otro","otras","otra","él","tanto","esa","estos","mucho","quienes","nada",
    "muchos","cual","poco","ella","estar","estas","algunas","algo","nosotros","mi",
    "mis","tú","te","ti","tu","tus","ellas","nosotras","vosotros","vosotras","os",
    "mío","mía","míos","mías","tuyo","tuya","tuyos","tuyas","suyo","suya","suyos",
    "suyas","nuestro","nuestra","nuestros","nuestras","vuestro","vuestra","vuestros",
    "vuestras","estoy","estás","está","estamos","estáis","están"
}

def remove_accents(text: str) -> str:
    """Quita acentos y caracteres diacríticos"""
    nk = unicodedata.normalize("NFKD", text)
    return "".join([c for c in nk if not unicodedata.combining(c)])

def preprocess(text: str) -> str:
    text = text.lower()
    text = remove_accents(text)
    # quitar puntuación (reemplaza por espacio para no juntar palabras)
    text = re.sub(r'[^\w\s]', ' ', text)
    # colapsar espacios
    text = re.sub(r'\s+', ' ', text).strip()
    return text

if st.button("Calcular TF-IDF"):
    documentos = [d.strip() for d in text_input.split("\n") if d.strip()]
    if len(documentos) == 0:
        st.warning("Por favor ingresa al menos un documento (una línea).")
    else:
        # Vectorizador con preprocesado y stopwords en español
        vectorizer = TfidfVectorizer(
            preprocessor=preprocess,
            stop_words=SPANISH_STOPWORDS,     # usamos la lista definida arriba
            token_pattern=r'(?u)\b\w\w+\b',  # tokens de al menos 2 caracteres
            max_df=0.85,                     # ignora términos muy frecuentes
            min_df=1                         # puedes subirlo a 2 si tienes muchos docs
        )

        X = vectorizer.fit_transform(documentos)
        df_tfidf = pd.DataFrame(
            X.toarray(),
            columns=vectorizer.get_feature_names_out(),
            index=[f"Doc {i+1}" for i in range(len(documentos))]
        )

        st.write("### Matriz TF-IDF (valores redondeados)")
        st.dataframe(df_tfidf.round(3))

        top_k = st.slider("Top k palabras por documento (gráfico)", 1, 10, 5)
        st.write("### Palabras más importantes por documento")
        for i, doc in enumerate(documentos):
            scores = df_tfidf.iloc[i]
            top_terms = scores.sort_values(ascending=False).head(top_k)
            st.write(f"**Doc {i+1}:** {doc}")
            st.bar_chart(top_terms)

