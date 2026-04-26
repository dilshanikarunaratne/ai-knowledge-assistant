from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os
import glob
from pypdf import PdfReader

DATA_FOLDER = "data"
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


def read_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def read_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text


all_chunks = []

files = glob.glob(f"{DATA_FOLDER}/*.txt") + glob.glob(f"{DATA_FOLDER}/*.pdf")

for file_path in files:
    if file_path.endswith(".txt"):
        text = read_txt(file_path)
    elif file_path.endswith(".pdf"):
        text = read_pdf(file_path)
    else:
        continue

    chunks = split_text(text)

    for chunk in chunks:
        all_chunks.append({
            "source": file_path,
            "text": chunk
        })

texts = [chunk["text"] for chunk in all_chunks]

chunk_embeddings = model.encode(texts)
chunk_embeddings = np.array(chunk_embeddings).astype("float32")

dimension = chunk_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(chunk_embeddings)

os.makedirs(INDEX_FOLDER, exist_ok=True)

faiss.write_index(index, f"{INDEX_FOLDER}/knowledge.index")

with open(f"{INDEX_FOLDER}/chunks.pkl", "wb") as file:
    pickle.dump(all_chunks, file)

print("Vector store created successfully.")
print(f"Processed {len(files)} files.")
print(f"Saved {len(all_chunks)} chunks.")