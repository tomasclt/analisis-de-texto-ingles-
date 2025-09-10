import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

st.title("TF-IDF Demo with Question Answering")

st.write("""
Each line is treated as a **document** (a sentence, a paragraph, or longer text).  
⚠️ Please write the documents in **English** so that stopwords (like *the, a, and*) are removed automatically.  

Now you can also **ask a question** and the app will return the most similar document.
""")

# Default input
text_input = st.text_area(
    "Write your documents (one per line):",
    "The dog barks loudly.\nThe cat meows at night.\nThe dog and the cat play together."
)

question = st.text_input("Ask a question (in English):", "Who is playing?")

if st.button("Compute TF-IDF and Find Answer"):
    documents = [doc.strip() for doc in text_input.split("\n") if doc.strip()]

    if len(documents) > 1:
        # Vectorizer with English stopwords
        vectorizer = TfidfVectorizer(stop_words="english")
        X = vectorizer.fit_transform(documents)

        # DataFrame with TF-IDF
        df_tfidf = pd.DataFrame(
            X.toarray(),
            columns=vectorizer.get_feature_names_out(),
            index=[f"Doc {i+1}" for i in range(len(documents))]
        )

        st.write("### TF-IDF Matrix")
        st.dataframe(df_tfidf.round(3))

        # Question vector
        question_vec = vectorizer.transform([question])

        # Cosine similarity
        similarities = cosine_similarity(question_vec, X).flatten()

        # Find best match
        best_idx = similarities.argmax()
        best_doc = documents[best_idx]

        st.write("### Question Answering")
        st.write(f"**Your question:** {question}")
        st.write(f"**Most relevant document (Doc {best_idx+1}):** {best_doc}")
        st.write(f"**Similarity score:** {similarities[best_idx]:.3f}")

        # Show all similarity scores
        sim_df = pd.DataFrame({
            "Document": [f"Doc {i+1}" for i in range(len(documents))],
            "Text": documents,
            "Similarity": similarities
        })
        st.write("### Similarity Scores")
        st.dataframe(sim_df.sort_values("Similarity", ascending=False))

    else:
        st.warning("Please enter at least two documents.")




