import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM

DATA_FOLDER = "data"
CHROMA_FOLDER = "chroma_db"

st.set_page_config(page_title="LangChain Knowledge Assistant", page_icon="🦜")

st.title("LangChain Knowledge Assistant")
st.write("Ask questions from your documents using LangChain + Ollama + Chroma.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


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

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    splits = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=CHROMA_FOLDER
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    llm = OllamaLLM(model="llama3.2:1b")

    return retriever, llm


if st.sidebar.button("Rebuild LangChain Vector Store"):
    st.cache_resource.clear()
    st.sidebar.success("Cache cleared. Ask a question to rebuild.")

retriever, llm = build_rag_pipeline()

if st.button("Clear Chat History"):
    st.session_state.chat_history = []

question = st.text_input("Ask a question:")

if st.button("Get Answer") and question:
    enhanced_query = f"""
Find information about: {question}
Focus on sections like technology stack, tools, or systems.
"""

    with st.spinner("Retrieving relevant documents..."):
        relevant_docs = retriever.invoke(enhanced_query)
    
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    prompt = f"""
You are a careful document question-answering assistant.

Use ONLY the context below.
If the context contains a direct answer, quote or closely paraphrase it.
Do not add technologies, tools, or facts that are not mentioned in the context.

Context:
{context}

Question:
{question}

Answer in 1-3 sentences.
If the answer is not in the context, say:
"I don't know based on the provided knowledge."
"""

    with st.spinner("Thinking..."):
        answer = llm.invoke(prompt)

    qa_prompt = f"""
You are a strict QA checker.

Check whether the answer contains only information supported by the context.

Context:
{context}

Question:
{question}

Answer:
{answer}

Return exactly:
Verdict: Supported / Not Supported / Partially Supported
Reason: one short sentence
"""

    with st.spinner("Checking answer quality..."):
        qa_check = llm.invoke(qa_prompt)

    sources = list(set([
        doc.metadata.get("source", "Unknown source")
        for doc in relevant_docs
    ]))

    st.session_state.chat_history.append({
        "question": question,
        "answer": answer,
        "qa_check": qa_check,
        "context": context,
        "sources": sources
    })

st.subheader("Chat History")

for chat in reversed(st.session_state.chat_history):
    st.markdown(f"**You:** {chat['question']}")
    st.markdown(f"**Assistant:** {chat['answer']}")
    st.markdown(f"**QA Check:** {chat['qa_check']}")

    st.markdown("**Sources:**")
    for source in chat["sources"]:
        st.write(f"- {source}")

    with st.expander("Retrieved Context"):
        st.write(chat["context"])

    st.divider()