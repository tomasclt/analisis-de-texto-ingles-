import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from deep_translator import GoogleTranslator

st.title("TF-IDF con traducción (Español ↔ Inglés)")

st.write("Cada línea se toma como un documento en español. Primero se traduce al inglés, "
         "se calcula TF-IDF con stopwords en inglés y al final se muestran los resultados en español.")

# Entrada de documentos en español
text_input = st.text_area("Escribe tus documentos (uno por línea):",
                          "El perro ladra fuerte.\nEl gato maúlla en la noche.\nEl perro y el gato juegan juntos.")

if st.button("Calcular TF-IDF"):
    documentos = [doc.strip() for doc in text_input.split("\n") if doc.strip()]

    if len(documentos) > 1:
        # Traducir documentos al inglés
        translator = GoogleTranslator(source="es", target="en")
        docs_en = [translator.translate(doc) for doc in documentos]

        # Vectorizador con stopwords en inglés
        vectorizer = TfidfVectorizer(stop_words="english")
        X = vectorizer.fit_transform(docs_en)

        # DataFrame con resultados
        df_tfidf = pd.DataFrame(
            X.toarray(),
            columns=vectorizer.get_feature_names_out(),
            index=[f"Doc {i+1}" for i in range(len(documentos))]
        )

        st.write("### Matriz TF-IDF (procesada en inglés)")
        st.dataframe(df_tfidf.round(3))

        # Mostrar top palabras traducidas al español
        st.write("### Palabras más importantes por documento (traducidas al español)")
        back_translator = GoogleTranslator(source="en", target="es")
        for i, doc in enumerate(documentos):
            scores = df_tfidf.iloc[i]
            top_terms = scores.sort_values(ascending=False).head(5)
            translated_terms = [back_translator.translate(term) for term in top_terms.index]
            st.write(f"**Doc {i+1}:** {doc}")
            st.bar_chart(pd.Series(top_terms.values, index=translated_terms))

    else:
        st.warning("Por favor, ingresa al menos dos documentos.")


