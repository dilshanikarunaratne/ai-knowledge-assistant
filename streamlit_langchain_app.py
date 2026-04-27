import os
import shutil

import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM

DATA_FOLDER = "data"
CHROMA_FOLDER = "chroma_db"

st.set_page_config(page_title="LangChain Knowledge Assistant", page_icon="✌")

st.title("LangChain Knowledge Assistant")
st.write("Ask questions from your Text and PDF documents using LangChain + Ollama + Chroma.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# -----------------------------
# Sidebar: document management
# -----------------------------
st.sidebar.header("Document Upload")

uploaded_file = st.sidebar.file_uploader(
    "Upload a Text(.txt) or PDF file",
    type=["txt", "pdf"]
)

if uploaded_file is not None:
    os.makedirs(DATA_FOLDER, exist_ok=True)

    file_path = os.path.join(DATA_FOLDER, uploaded_file.name)

    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())

    if os.path.exists(CHROMA_FOLDER):
        shutil.rmtree(CHROMA_FOLDER)

    st.cache_resource.clear()
    st.sidebar.success(f"Uploaded: {uploaded_file.name}")
    st.sidebar.info("Vector store cleared. Ask a question to rebuild it.")


st.sidebar.subheader("Current Documents")

os.makedirs(DATA_FOLDER, exist_ok=True)

documents = [
    file for file in os.listdir(DATA_FOLDER)
    if file.endswith((".txt", ".pdf"))
]

if documents:
    selected_file = st.sidebar.selectbox(
        "Select a document",
        documents,
        key="document_selectbox"
    )

    if st.sidebar.button("Delete Selected Document", key="delete_document_button"):
        os.remove(os.path.join(DATA_FOLDER, selected_file))

        if os.path.exists(CHROMA_FOLDER):
            shutil.rmtree(CHROMA_FOLDER)

        st.cache_resource.clear()
        st.sidebar.success(f"Deleted: {selected_file}")
        st.sidebar.info("Vector store cleared. Ask a question to rebuild it.")
else:
    st.sidebar.info("No documents found.")

# Rebuild LangChain vector store 
if st.sidebar.button("Rebuild Vector Store"):
    if os.path.exists(CHROMA_FOLDER):
        shutil.rmtree(CHROMA_FOLDER)

    st.cache_resource.clear()
    st.sidebar.success("Vector store cleared. Ask a question to rebuild it.")


st.sidebar.subheader("System Status")
st.sidebar.write("LLM: llama3.2:1b")
st.sidebar.write("Vector DB: Chroma")
st.sidebar.write(f"Documents loaded: {len(documents)}")


# -----------------------------
# Helper functions
# -----------------------------
def get_conversation_memory(max_turns=5):
    recent_chats = st.session_state.chat_history[-max_turns:]

    memory_parts = []
    for chat in recent_chats:
        memory_parts.append(f"User: {chat['question']}")
        memory_parts.append(f"Assistant: {chat['answer']}")

    return "\n".join(memory_parts)


def format_source_snippets(relevant_docs):
    snippets = []

    for doc in relevant_docs:
        source = doc.metadata.get("source", "Unknown source")
        page = doc.metadata.get("page", None)

        snippet = doc.page_content.strip().replace("\n", " ")

        if len(snippet) > 400:
            snippet = snippet[:400] + "..."

        source_label = source
        if page is not None:
            source_label = f"{source} | page {page + 1}"

        snippets.append({
            "source": source_label,
            "snippet": snippet
        })

    return snippets


# -----------------------------
# Build LangChain RAG pipeline
# -----------------------------
@st.cache_resource
def build_rag_pipeline():
    txt_loader = DirectoryLoader(
        DATA_FOLDER,
        glob="**/*.txt",
        loader_cls=TextLoader
    )

    docs = txt_loader.load()

    pdf_loader = DirectoryLoader(
        DATA_FOLDER,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )

    docs.extend(pdf_loader.load())

    if not docs:
        return None, None

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    splits = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=CHROMA_FOLDER
    )

    retriever = vectordb.as_retriever(
        search_kwargs={"k": 5}
    )

    llm = OllamaLLM(model="llama3.2:1b")

    return retriever, llm


retriever, llm = build_rag_pipeline()

if retriever is None or llm is None:
    st.warning("Please upload at least one TXT or PDF document.")
    st.stop()


# -----------------------------
# Chat controls
# -----------------------------
if st.button("Clear Chat History"):
    st.session_state.chat_history = []

for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(chat["question"])

    with st.chat_message("assistant"):
        st.write(chat["answer"])

        if "Partially Supported" in chat["qa_check"]:
            st.warning("Answer is partially supported")
        elif "Not Supported" in chat["qa_check"]:
            st.error("Answer may not be reliable")
        elif "Supported" in chat["qa_check"]:
            st.success("Answer is supported by documents")
        else:
            st.info("QA result unclear")

        with st.expander("Search Preview"):
            for i, item in enumerate(chat["source_snippets"], start=1):
                st.markdown(f"**Result {i}: {item['source']}**")
                st.write(item["snippet"])

        with st.expander("QA Check"):
            st.write(chat["qa_check"])

        with st.expander("Retrieved Context"):
            st.write(chat["context"])


question = st.chat_input("Ask a question about your documents...")

if question:
    with st.chat_message("user"):
        st.write(question)

    conversation_memory = get_conversation_memory()

    enhanced_query = f"""
Current question:
{question}

Recent conversation:
{conversation_memory}

Find information relevant to the current question.
Focus on relevant sections, technology stack, tools, systems, policies, rules, features, business value, and follow-up context.
"""

    with st.status("Processing...", expanded=True) as status:
        st.write("Retrieving relevant documents...")
        relevant_docs = retriever.invoke(enhanced_query)

        context = "\n\n".join([
            doc.page_content for doc in relevant_docs
        ])

        source_snippets = format_source_snippets(relevant_docs)

        st.write("Generating answer...")

        prompt = f"""
You are a careful document question-answering assistant.

Use ONLY the document context below.
You may use the recent conversation only to understand follow-up questions.
Do not add facts that are not supported by the document context.

Recent conversation:
{conversation_memory}

Document context:
{context}

Current question:
{question}

Answer in 1-3 sentences.
If the answer is not in the document context, say:
"I don't know based on the provided knowledge."
"""

        answer = llm.invoke(prompt)

        qa_prompt = f"""
You are a strict QA checker.

Check whether the answer contains only information supported by the document context.

Document context:
{context}

Question:
{question}

Answer:
{answer}

Return exactly:
Verdict: Supported / Not Supported / Partially Supported
Reason: one short sentence
"""

        st.write("Running QA check...")
        qa_check = llm.invoke(qa_prompt)

        status.update(label="Done", state="complete")

    sources = list(set([
        item["source"] for item in source_snippets
    ]))

    st.session_state.chat_history.append({
        "question": question,
        "answer": answer,
        "qa_check": qa_check,
        "context": context,
        "sources": sources,
        "source_snippets": source_snippets
    })

    with st.chat_message("assistant"):
        st.write(answer)

        if "Partially Supported" in qa_check:
            st.warning("Answer is partially supported")
        elif "Not Supported" in qa_check:
            st.error("Answer may not be reliable")
        elif "Supported" in qa_check:
            st.success("Answer is supported by documents")
        else:
            st.info("QA result unclear")

        with st.expander("Search Preview"):
            for i, item in enumerate(source_snippets, start=1):
                st.markdown(f"**Result {i}: {item['source']}**")
                st.write(item["snippet"])

        with st.expander("QA Check"):
            st.write(qa_check)

        with st.expander("Retrieved Context"):
            st.write(context)