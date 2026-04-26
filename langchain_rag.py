from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM

# Load TXT files
txt_loader = DirectoryLoader(
    "data",
    glob="**/*.txt",
    loader_cls=TextLoader
)

docs = txt_loader.load()

# Load PDF files
pdf_loader = DirectoryLoader(
    "data",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader
)

docs.extend(pdf_loader.load())

# Split documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

splits = text_splitter.split_documents(docs)

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create Chroma vector database
vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory="chroma_db"
)

retriever = vectordb.as_retriever(search_kwargs={"k": 3})

llm = OllamaLLM(model="llama3.2:1b")

while True:
    query = input("\nAsk a question (or type 'exit'): ")

    if query.lower() == "exit":
        break

    relevant_docs = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    prompt = f"""
You are an AI assistant.
Answer using the context below.
Do not use outside knowledge.

Context:
{context}

Question:
{query}

If the answer is not found in the context, say:
"I don't know based on the provided knowledge."
"""

    answer = llm.invoke(prompt)

    print("\nAnswer:")
    print(answer)

    print("\nSources:")
    for doc in relevant_docs:
        print("-", doc.metadata.get("source"))