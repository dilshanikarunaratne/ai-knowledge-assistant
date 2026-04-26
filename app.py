import ollama
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

INDEX_FOLDER = "vector_store"

model = SentenceTransformer("all-MiniLM-L6-v2")

index = faiss.read_index(f"{INDEX_FOLDER}/knowledge.index")

with open(f"{INDEX_FOLDER}/chunks.pkl", "rb") as file:
    chunks = pickle.load(file)

question = input("Ask a question: ")

question_embedding = model.encode([question])
question_embedding = np.array(question_embedding).astype("float32")

k = 3
distances, indices = index.search(question_embedding, k)

relevant_chunks = [chunks[i] for i in indices[0]]

context_parts = []
for chunk in relevant_chunks:
    context_parts.append(
        f"Source: {chunk['source']}\nText: {chunk['text']}"
    )

context = "\n\n".join(context_parts)

print("\n--- Retrieved Context ---")
print(context)
print("-------------------------")

prompt = f"""
You are an AI assistant.
Answer using the context below.
You may infer the answer if it is clearly implied by the context.
Do not use outside knowledge.

Context:
{context}

Question:
{question}

If answer not found or clearly implied, say:
"I don't know based on the provided knowledge."
"""

response = ollama.chat(
    model="llama3.2:1b",
    messages=[{"role": "user", "content": prompt}]
)

print("\nAnswer:")
print(response["message"]["content"])