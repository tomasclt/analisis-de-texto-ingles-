# app.py
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
from nltk.stem import SnowballStemmer

st.title("TF-IDF QA (with stemming)")

st.write("""
Each line is a document (sentence/paragraph).  
Write documents in **English**. The app applies basic normalization + *stemming* so words like *playing* and *play* match.
""")

# Example docs
text_input = st.text_area(
    "Write your documents (one per line):",
    "The dog barks loudly.\nThe cat meows at night.\nThe dog and the cat play together."
)

question = st.text_input("Ask a question (in English):", "Who is playing?")

# initialize stemmer
stemmer = SnowballStemmer("english")

def tokenize_and_stem(text: str):
    # Lowercase
    text = text.lower()
    # Remove non-letters (keep spaces)
    text = re.sub(r'[^a-z\s]', ' ', text)
    # Simple token split (keeps tokens length > 1)
    tokens = [t for t in text.split() if len(t) > 1]
    # Stem each token
    stems = [stemmer.stem(t) for t in tokens]
    return stems

if st.button("Compute TF-IDF and Find Answer"):
    documents = [d.strip() for d in text_input.split("\n") if d.strip()]
    if len(documents) < 1:
        st.warning("Please enter at least one document.")
    else:
        # Use custom tokenizer (stemming). Set token_pattern=None when passing tokenizer.
        vectorizer = TfidfVectorizer(
            tokenizer=tokenize_and_stem,
            stop_words="english",
            token_pattern=None
        )

        # Fit on documents
        X = vectorizer.fit_transform(documents)

        # DataFrame for display
        df_tfidf = pd.DataFrame(
            X.toarray(),
            columns=vectorizer.get_feature_names_out(),
            index=[f"Doc {i+1}" for i in range(len(documents))]
        )

        st.write("### TF-IDF Matrix (stems)")
        st.dataframe(df_tfidf.round(3))

        # Transform the question with the same vectorizer (same preprocessing + stems)
        question_vec = vectorizer.transform([question])

        # Cosine similarity (question vs each document)
        similarities = cosine_similarity(question_vec, X).flatten()

        best_idx = similarities.argmax()
        best_doc = documents[best_idx]
        best_score = similarities[best_idx]

        st.write("### Question Answering")
        st.write(f"**Your question:** {question}")
        st.write(f"**Most relevant document (Doc {best_idx+1}):** {best_doc}")
        st.write(f"**Similarity score:** {best_score:.3f}")

        # Show all similarity scores
        sim_df = pd.DataFrame({
            "Document": [f"Doc {i+1}" for i in range(len(documents))],
            "Text": documents,
            "Similarity": similarities
        })
        st.write("### Similarity Scores (sorted)")
        st.dataframe(sim_df.sort_values("Similarity", ascending=False))

        # Optional: show which stems in the vocabulary matched between question and best doc
        vocab = vectorizer.get_feature_names_out()
        # stems present in question
        q_stems = tokenize_and_stem(question)
        matched = [s for s in q_stems if s in vocab and df_tfidf.iloc[best_idx].get(s, 0) > 0]
        st.write("### Matching stems from question present in best document:", matched)





