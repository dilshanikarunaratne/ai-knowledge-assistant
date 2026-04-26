import streamlit as st
import ollama
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os
import subprocess

INDEX_FOLDER = "vector_store"
DATA_FOLDER = "data"

st.set_page_config(page_title="AI Knowledge Assistant", page_icon="🤖")

st.title("AI Knowledge Assistant")
st.write("Ask questions from your local knowledge base.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


@st.cache_resource
def load_resources():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index(f"{INDEX_FOLDER}/knowledge.index")

    with open(f"{INDEX_FOLDER}/chunks.pkl", "rb") as file:
        chunks = pickle.load(file)

    return model, index, chunks


st.sidebar.header("Document Upload")

uploaded_file = st.sidebar.file_uploader(
    "Upload a TXT or PDF file",
    type=["txt", "pdf"]
)

if uploaded_file is not None:
    os.makedirs(DATA_FOLDER, exist_ok=True)

    file_path = os.path.join(DATA_FOLDER, uploaded_file.name)

    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())

    st.sidebar.success(f"Uploaded: {uploaded_file.name}")
    st.sidebar.info("Now rebuild the vector store.")

if st.sidebar.button("Rebuild Vector Store"):
    with st.sidebar:
        with st.spinner("Rebuilding vector store..."):
            result = subprocess.run(
                ["python", "ingest.py"],
                capture_output=True,
                text=True
            )

        if result.returncode == 0:
            st.cache_resource.clear()
            st.success("Vector store rebuilt successfully.")
            st.text(result.stdout)
        else:
            st.error("Failed to rebuild vector store.")
            st.text(result.stderr)


try:
    model, index, chunks = load_resources()
except Exception:
    st.error("Vector store not found. Please run `python ingest.py` or rebuild from the sidebar.")
    st.stop()


if st.button("Clear Chat History"):
    st.session_state.chat_history = []

question = st.text_input("Ask a question:")

if st.button("Get Answer") and question:

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

    with st.spinner("Thinking..."):
        response = ollama.chat(
            model="llama3.2:1b",
            messages=[{"role": "user", "content": prompt}]
        )

    answer = response["message"]["content"]

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

    qa_check = qa_response["message"]["content"]

    st.session_state.chat_history.append({
        "question": question,
        "answer": answer,
        "qa_check": qa_check,
        "context": context
    })

st.subheader("Chat History")

for chat in reversed(st.session_state.chat_history):
    st.markdown(f"**You:** {chat['question']}")
    st.markdown(f"**Assistant:** {chat['answer']}")
    st.markdown(f"**QA Check:** {chat['qa_check']}")

    with st.expander("Retrieved Context"):
        st.write(chat["context"])

    st.divider()