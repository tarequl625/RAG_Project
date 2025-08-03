import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
dimension = 384
index = faiss.IndexFlatL2(dimension)
stored_chunks = []

def embed_chunks(chunks):
    vectors = model.encode(chunks)
    index.add(np.array(vectors).astype("float32"))
    stored_chunks.extend(chunks)

def query_chunks(question, k=3):
    q_vec = model.encode([question])
    D, I = index.search(np.array(q_vec).astype("float32"), k)
    return [stored_chunks[i] for i in I[0]]
