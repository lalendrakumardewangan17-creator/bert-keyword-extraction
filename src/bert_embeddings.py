from sentence_transformers import SentenceTransformer

# CPU-friendly transformer
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text):
    return model.encode(text)