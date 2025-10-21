# -*- coding: utf-8 -*-
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
from nltk.stem import SnowballStemmer

# ----------------------------------------------------------------
# CONFIGURACIÓN DE PÁGINA Y ESTILO GLOBAL
# ----------------------------------------------------------------
st.set_page_config(page_title="TF-IDF Dashboard", page_icon="📊", layout="wide")

st.markdown("""
<style>
:root{
  --bg:#0b1120;
  --panel:#111827;
  --border:#1f2937;
  --text:#f8fafc;
  --accent:#3b82f6;
  --muted:#94a3b8;
  --pos:#10b981;
  --neg:#ef4444;
}
[data-testid="stAppViewContainer"]{
  background: radial-gradient(circle at 30% 10%, #0f172a 0%, #0b1120 100%) !important;
  color: var(--text);
  font-family: 'Inter', ui-sans-serif, system-ui, -apple-system, 'Segoe UI';
}
h1,h2,h3,h4 { color: var(--text); letter-spacing:-.02em; }
small, span, p, label { color: var(--muted); }
.stTextArea textarea, .stTextInput input {
  background-color:#0f172a !important;
  border:1px solid #334155 !important;
  color:var(--text) !important;
  border-radius:12px;
  transition:all .25s ease;
}
.stTextArea textarea:focus, .stTextInput input:focus{
  border-color: var(--accent) !important;
  box-shadow:0 0 0 2px rgba(59,130,246,.25);
}
.stButton > button {
  background:linear-gradient(90deg, #2563eb, #1d4ed8);
  border:none;
  border-radius:999px;
  color:white;
  padding:.7rem 1.2rem;
  font-weight:600;
  box-shadow:0 8px 30px rgba(37,99,235,.35);
  transition:all .2s ease;
}
.stButton > button:hover {transform:translateY(-1px); box-shadow:0 12px 35px rgba(37,99,235,.45);}
.card{
  background:var(--panel);
  border:1px solid var(--border);
  border-radius:18px;
  padding:1.5rem;
  box-shadow:0 10px 30px rgba(0,0,0,.35);
  animation:fadeIn .6s ease;
}
@keyframes fadeIn{from{opacity:0;transform:translateY(10px);}to{opacity:1;transform:translateY(0);}}
.dataframe th { background:#1e293b !important; color:#f1f5f9 !important; }
.dataframe td { color:#e2e8f0 !important; }
.badge {
  display:inline-block; padding:.25rem .6rem; border-radius:999px; font-weight:600;
  font-size:.8rem; border:1px solid rgba(255,255,255,.1);
}
.badge-blue {background:rgba(37,99,235,.15); color:#93c5fd;}
.badge-green {background:rgba(16,185,129,.15); color:#6ee7b7;}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------
# INTERFAZ DE USUARIO
# ----------------------------------------------------------------
st.markdown("<h1 style='text-align:center;'>📊 Demo interactiva TF-IDF</h1>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; color:#cbd5e1;'>
Cada línea se trata como un <b>documento independiente</b>.<br>
El modelo usa <b>TF-IDF + cosine similarity</b> con <i>stemming</i> en inglés.
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([2,1])

with col1:
    text_input = st.text_area(
        "✏️ Escribe tus documentos (uno por línea, en inglés):",
        "The dog barks loudly.\nThe cat meows at night.\nThe dog and the cat play together."
    )

with col2:
    question = st.text_input("💬 Escribe una pregunta (en inglés):", "Who is playing?")

# Stemmer
stemmer = SnowballStemmer("english")
def tokenize_and_stem(text: str):
    text = text.lower()
    text = re.sub(r'[^a-z\\s]', ' ', text)
    tokens = [t for t in text.split() if len(t) > 1]
    return [stemmer.stem(t) for t in tokens]

# ----------------------------------------------------------------
# PROCESAMIENTO
# ----------------------------------------------------------------
if st.button("🚀 Calcular TF-IDF y buscar respuesta"):
    documents = [d.strip() for d in text_input.split("\\n") if d.strip()]
    if len(documents) < 1:
        st.warning("⚠️ Ingresa al menos un documento.")
    else:
        vectorizer = TfidfVectorizer(
            tokenizer=tokenize_and_stem,
            stop_words="english",
            token_pattern=None
        )
        X = vectorizer.fit_transform(documents)

        df_tfidf = pd.DataFrame(
            X.toarray(),
            columns=vectorizer.get_feature_names_out(),
            index=[f"Doc {i+1}" for i in range(len(documents))]
        )

        with st.container():
            st.markdown("<h3>🧠 Matriz TF-IDF (stems)</h3>", unsafe_allow_html=True)
            st.dataframe(df_tfidf.round(3), use_container_width=True)

        question_vec = vectorizer.transform([question])
        similarities = cosine_similarity(question_vec, X).flatten()

        best_idx = similarities.argmax()
        best_doc = documents[best_idx]
        best_score = similarities[best_idx]

        # RESULTADOS
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3>🔍 Resultado del análisis</h3>", unsafe_allow_html=True)
        st.markdown(f"<p><b>Tu pregunta:</b> {question}</p>", unsafe_allow_html=True)
        st.markdown(f"<p><b>Documento más relevante:</b> <span class='badge badge-green'>Doc {best_idx+1}</span></p>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:#f9fafb; font-size:1.05rem;'>{best_doc}</p>", unsafe_allow_html=True)
        st.markdown(f"<p><b>Puntaje de similitud:</b> <span class='badge badge-blue'>{best_score:.3f}</span></p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Tabla de similitud
        sim_df = pd.DataFrame({
            "Documento": [f"Doc {i+1}" for i in range(len(documents))],
            "Texto": documents,
            "Similitud": similarities
        })
        st.markdown("<h3>📈 Puntajes de similitud</h3>", unsafe_allow_html=True)
        st.dataframe(sim_df.sort_values("Similitud", ascending=False), use_container_width=True)

        vocab = vectorizer.get_feature_names_out()
        q_stems = tokenize_and_stem(question)
        matched = [s for s in q_stems if s in vocab and df_tfidf.iloc[best_idx].get(s, 0) > 0]

        st.markdown("<h3>🧩 Stems de la pregunta presentes en el documento elegido</h3>", unsafe_allow_html=True)
        st.write(matched)



