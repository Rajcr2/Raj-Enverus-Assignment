# Product Intern - Machine Learning & Gen-AI: RAG Assignment

A robust Retrieval-Augmented Generation (RAG) chatbot pipeline built to ingest technical documentation, parse documents, store embeddings, and perform accurate question answering with retrieved context.

---

## 🛠️ Tech Stack & Architecture

* **LLM Engine:** Groq (Openai-gpt-oss-20b)
* **Vector Database:** ChromaDB
* **Orchestration:** LangChain / Custom Python Scripts
* **Environment Management:** Python-dotenv

---

## 🔄 RAG Process Workflow

```mermaid
graph TD
    A[User Query] --> B[App Interface / CLI]
    B --> C[Query Embedding]
    
    D[DevAI PDF Document] --> E[Text Chunking & Processing]
    E --> F[(ChromaDB Vector Store)]
    
    C -->|Similarity Search| F
    F -->|Top-K Retrieved Chunks| G[Prompt Builder]
    
    G --> H[Groq LLM API]
    H --> I[Final Generated Answer with Sources]
```

## 📁 Repository Structure


v3/
│
├── data/
│   └── 2410.10934v2_7033 1.pdf      # Source document for ingestion
│
├── src/
│   ├── app.py                    # Main application / chatbot entry point
│   ├── ingest.py                 # Document loading, splitting, and vector store indexing
│   ├── prompts.py                # System prompt templates
│   └── rag.py                    # RAG chain logic and retrieval implementation
│
├── .env                          # Environment configuration template
├── requirements.txt              # Python package dependencies
└── README.md                     # Project documentation and workflow

## 🚀 Setup and Installation

1. Clone the repository (and ensure you are in the project root directory).
2. Install dependencies :
```
pip install -r requirements.txt
```





