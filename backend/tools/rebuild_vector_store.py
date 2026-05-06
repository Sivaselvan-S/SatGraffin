"""Utility to rebuild the FAISS vector store from processed_data/*.txt."""
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
PROCESSED_DIR = Path("processed_data")
VECTOR_STORE_PATH = "vector_store"


def rebuild() -> None:
    if not PROCESSED_DIR.exists():
        raise SystemExit(f"Processed data directory '{PROCESSED_DIR}' not found.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
    chunks = []
    files = sorted(PROCESSED_DIR.glob("*.txt"))
    for file_path in files:
        loader = TextLoader(str(file_path), encoding="utf-8")
        docs = loader.load()
        chunks.extend(splitter.split_documents(docs))

    if not chunks:
        raise SystemExit("No text files found under processed_data/. Nothing to index.")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": "cpu"})
    store = FAISS.from_documents(chunks, embeddings)
    store.save_local(VECTOR_STORE_PATH)
    print(f"Rebuilt vector store with {len(chunks)} chunks from {len(files)} files.")


if __name__ == "__main__":
    rebuild()
