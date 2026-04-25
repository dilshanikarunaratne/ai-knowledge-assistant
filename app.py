import ollama
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load document
with open("data/company_notes.txt", "r", encoding="utf-8") as file:
    text = file.read()

# Split into chunks
def split_text(text, chunk_size=300):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

chunks = split_text(text)

# Create embeddings
chunk_embeddings = model.encode(chunks)

# Create FAISS index
dimension = chunk_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(chunk_embeddings))

# Ask user
question = input("Ask a question: ")

# Embed question
question_embedding = model.encode([question])

# Search top 2 similar chunks
k = 2
distances, indices = index.search(np.array(question_embedding), k)

# Get relevant chunks
relevant_chunks = [chunks[i] for i in indices[0]]
context = "\n\n".join(relevant_chunks)

# Prompt
prompt = f"""
You are an AI assistant.
Answer using ONLY the context below.

Context:
{context}

Question:
{question}

If answer not found, say:
"I don't know based on the provided knowledge."
"""

# LLM response
response = ollama.chat(
    model="llama3.2:1b",
    messages=[{"role": "user", "content": prompt}]
)

print("\nAnswer:")
print(response["message"]["content"])