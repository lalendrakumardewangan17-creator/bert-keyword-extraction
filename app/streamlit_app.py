import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import streamlit as st
from preprocessing import clean_text
from candidate_generator import generate_candidates
from keyword_ranker import rank_keywords

st.title("CPU-friendly BERT Keyword Extraction")

text = st.text_area("Enter marketing text here:")

if st.button("Extract Keywords"):
    clean = clean_text(text)
    candidates = generate_candidates(clean)
    keywords = rank_keywords(clean, candidates)
    st.subheader("Top Keywords")
    for k, score in keywords:
        st.write(f"{k} → {round(float(score), 3)}")
