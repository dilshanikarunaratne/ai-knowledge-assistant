# AI Knowledge Assistant (RAG + LangChain + Ollama)

A ChatGPT-style AI assistant that can answer questions from your own documents (TXT & PDF), using Retrieval-Augmented Generation (RAG) with a local LLM.

---

## Features

- Multi-document support (TXT + PDF)
- Semantic search using embeddings (Chroma vector database)
- Local LLM via Ollama (no API required)
- Chat-style UI (Streamlit)
- Conversation memory (multi-turn chat)
- Source transparency (shows retrieved context)
- QA validation (checks if answer is supported by documents)
- Upload, delete, and manage documents via UI
- Fully local and privacy-friendly

---

## Architecture
User Question
↓
Retriever (Chroma + Embeddings)
↓
Relevant Document Chunks
↓
LLM (Ollama)
↓
Answer
↓
QA Validator (LLM check)


---

## Tech Stack

- **Python**
- **LangChain**
- **Chroma (Vector Database)**
- **Ollama (Local LLM)**
- **Sentence Transformers (Embeddings)**
- **Streamlit (UI)**

---

## 📂 Project Structure
ai-knowledge-assistant/
│
├── data/ # Documents (TXT, PDF)
├── chroma_db/ # Vector database (auto-generated)
├── streamlit_langchain_app.py # Main UI app
├── ingest.py # (Optional) manual ingestion
├── requirements.txt
└── README.md


---

## ⚙️ Installation

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/ai-knowledge-assistant.git
cd ai-knowledge-assistant

python -m venv venv
venv\Scripts\activate   # Windows

pip install -r requirements.txt
```
### Setup Ollama

Install Ollama:
https://ollama.com

Pull model:

```bash
ollama pull llama3.2:1b

streamlit run streamlit_langchain_app.py
```

Usage
Upload TXT or PDF documents from the sidebar
Click Rebuild Vector Store
Ask questions in chat
View:
Answers
QA validation
Source snippets
Retrieved context

Key Learnings
RAG performance depends heavily on retrieval quality
Chunking strategy significantly impacts results
QA validation helps reduce hallucinations
Local LLMs are powerful but require careful prompting

Future Improvements
Multi-agent system using LangGraph
Better document parsing (tables, OCR)
Hybrid search (vector + keyword)
Deployment (Docker / cloud)
Model switching (Mistral, Phi, etc.)





