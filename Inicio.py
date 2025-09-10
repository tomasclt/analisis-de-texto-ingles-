import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

st.title("TF-IDF Demo (English documents only)")

st.write("""
Each line is treated as a **document** (it can be a sentence, a paragraph, or a longer text).  
⚠️ Please write the documents in **English** so that stopwords (like *the, a, and*) are removed automatically.
""")

# Default input in English
text_input = st.text_area(
    "Write your documents (one per line):",
    "The dog barks loudly.\nThe cat meows at night.\nThe dog and the cat play together."
)

if st.button("Compute TF-IDF"):
    documents = [doc.strip() for doc in text_input.split("\n") if doc.strip()]

    if len(documents) > 1:
        # Vectorizer with English stopwords
        vectorizer = TfidfVectorizer(stop_words="english")
        X = vectorizer.fit_transform(documents)

        # DataFrame with TF-IDF matrix
        df_tfidf = pd.DataFrame(
            X.toarray(),
            columns=vectorizer.get_feature_names_out(),
            index=[f"Doc {i+1}" for i in range(len(documents))]
        )

        st.write("### TF-IDF Matrix")
        st.dataframe(df_tfidf.round(3))

        # Show top terms per document
        st.write("### Most important words per document")
        for i, doc in enumerate(documents):
            scores = df_tfidf.iloc[i]
            top_terms = scores.sort_values(ascending=False).head(5)
            st.write(f"**Doc {i+1}:** {doc}")
            st.bar_chart(top_terms)

    else:
        st.warning("Please enter at least two documents.")




