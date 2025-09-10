import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

st.title("Demo de TF-IDF en NLP")

st.write("Cada línea que escribas se tratará como un **documento** (puede ser una frase, párrafo o texto más largo).")

# Entrada de documentos
text_input = st.text_area("Escribe tus documentos aquí:", 
                          "El perro ladra fuerte.\n"
                          "El gato maúlla en la noche.\n"
                          "El perro y el gato juegan juntos.")

# Botón para procesar
if st.button("Calcular TF-IDF"):
    documentos = [doc.strip() for doc in text_input.split("\n") if doc.strip()]

    if len(documentos) > 1:
        # Crear vectorizador TF-IDF
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(documentos)

        # Convertir a DataFrame
        df_tfidf = pd.DataFrame(
            X.toarray(), 
            columns=vectorizer.get_feature_names_out(), 
            index=[f"Doc {i+1}" for i in range(len(documentos))]
        )

        st.write("### Matriz TF-IDF")
        st.dataframe(df_tfidf.style.format(precision=2))

        # Mostrar gráficos de las palabras más relevantes
        st.write("### Palabras más importantes por documento")
        for i, doc in enumerate(documentos):
            scores = df_tfidf.iloc[i]
            top_terms = scores.sort_values(ascending=False).head(5)  # Top 5 palabras
            st.write(f"**Doc {i+1}:** {doc}")
            st.bar_chart(top_terms)

    else:
        st.warning("Por favor, ingresa al menos dos documentos.")
