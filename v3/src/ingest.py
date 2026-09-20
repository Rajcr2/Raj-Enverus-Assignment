"""Build the ChromaDB knowledge base from the Agent-as-a-Judge PDF."""
import re
from pathlib import Path

import chromadb
import pymupdf
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
PDF_PATH = BASE_DIR / "data" / "2410.10934v2_7033 1.pdf"
CHROMA_PATH = BASE_DIR / "chroma_embeddings"
COLLECTION_NAME = "agent_as_a_judge"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Small chunks keep one table row / one figure caption per chunk, so the
# diagram and table text is not drowned out by neighbouring text.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def clean_text(text):
    text = text.replace("\u00ad", "")
    text = re.sub(r"-\n(?=\w)", "", text)  # join line-break hyphenation
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            # end on a sentence boundary when one is close enough
            boundary = text.rfind(". ", start, end)
            if boundary > start + int(size * 0.6):
                end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def build_documents():
    documents, metadatas, ids = [], [], []
    pdf = pymupdf.open(PDF_PATH)
    for page_no, page in enumerate(pdf, start=1):
        for chunk_no, chunk in enumerate(chunk_text(clean_text(page.get_text("text"))), start=1):
            documents.append(chunk)
            metadatas.append({"source": PDF_PATH.name, "page": page_no, "chunk": chunk_no})
            ids.append(f"page-{page_no}-chunk-{chunk_no}")
    pdf.close()
    return documents, metadatas, ids


def main():
    documents, metadatas, ids = build_documents()
    print(f"Extracted {len(documents)} chunks from {PDF_PATH.name}")

    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(documents, batch_size=32, show_progress_bar=True,
                              normalize_embeddings=True).tolist()

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    try:
        client.delete_collection(COLLECTION_NAME)  # rebuild from scratch
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    for start in range(0, len(documents), 100):  # add in batches
        end = start + 100
        collection.add(ids=ids[start:end], documents=documents[start:end],
                       metadatas=metadatas[start:end], embeddings=embeddings[start:end])

    print(f"Stored {collection.count()} vectors in {CHROMA_PATH}")


if __name__ == "__main__":
    main()
