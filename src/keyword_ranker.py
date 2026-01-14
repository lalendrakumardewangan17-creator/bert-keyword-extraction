from sentence_transformers import util
from bert_embeddings import get_embedding

def rank_keywords(text, candidates, top_n=10):
    text_emb = get_embedding(text)
    candidate_embs = [get_embedding(c) for c in candidates]

    scores = util.cos_sim(text_emb, candidate_embs)[0]

    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]
