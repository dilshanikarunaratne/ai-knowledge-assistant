import streamlit as st
import ollama
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

INDEX_FOLDER = "vector_store"

st.set_page_config(page_title="AI Knowledge Assistant", page_icon="🤖")

st.title("AI Knowledge Assistant")
st.write("Ask questions from your local knowledge base.")


@st.cache_resource
def load_resources():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index(f"{INDEX_FOLDER}/knowledge.index")

    with open(f"{INDEX_FOLDER}/chunks.pkl", "rb") as file:
        chunks = pickle.load(file)

    return model, index, chunks


model, index, chunks = load_resources()

question = st.text_input("Ask a question:")

if st.button("Get Answer") and question:

    # Embed question
    question_embedding = model.encode([question])
    question_embedding = np.array(question_embedding).astype("float32")

    # Retrieve top chunks
    k = 3
    distances, indices = index.search(question_embedding, k)
    relevant_chunks = [chunks[i] for i in indices[0]]

    # Build context
    context_parts = []
    for chunk in relevant_chunks:
        context_parts.append(
            f"Source: {chunk['source']}\nText: {chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    # Main prompt
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

    # Generate answer
    with st.spinner("Thinking..."):
        response = ollama.chat(
            model="llama3.2:1b",
            messages=[{"role": "user", "content": prompt}]
        )

    answer = response["message"]["content"]

    st.subheader("Answer")
    st.write(answer)

    # QA validation step
    qa_prompt = f"""
You are a QA validation assistant.

Check whether the answer is supported by the context.

Context:
{context}

Question:
{question}

Answer:
{answer}

Respond in this format:
Verdict: Supported / Not Supported / Partially Supported
Reason: short explanation
"""

    with st.spinner("Checking answer quality..."):
        qa_response = ollama.chat(
            model="llama3.2:1b",
            messages=[{"role": "user", "content": qa_prompt}]
        )

    st.subheader("QA Check")
    st.write(qa_response["message"]["content"])

    # Show retrieved chunks
    with st.expander("Retrieved Context"):
        st.write(context)