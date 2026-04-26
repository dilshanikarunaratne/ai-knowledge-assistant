from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os

DATA_FILE = "data/company_notes.txt"
INDEX_FOLDER = "vector_store"

model = SentenceTransformer("all-MiniLM-L6-v2")

def split_text(text, chunk_size=300, overlap=50):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap

    return chunks

with open(DATA_FILE, "r", encoding="utf-8") as file:
    text = file.read()

chunks = split_text(text)

chunk_embeddings = model.encode(chunks)
chunk_embeddings = np.array(chunk_embeddings).astype("float32")

dimension = chunk_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(chunk_embeddings)

os.makedirs(INDEX_FOLDER, exist_ok=True)

faiss.write_index(index, f"{INDEX_FOLDER}/company.index")

with open(f"{INDEX_FOLDER}/chunks.pkl", "wb") as file:
    pickle.dump(chunks, file)

print("Vector store created successfully.")
print(f"Saved {len(chunks)} chunks.")