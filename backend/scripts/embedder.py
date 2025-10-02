import os
# LangChain has updated its structure. Let's use the new recommended imports.
from langchain_community.document_loaders import TextLoader, DirectoryLoader # <-- UPDATED
from langchain_community.embeddings import HuggingFaceEmbeddings # <-- UPDATED
from langchain_community.vectorstores import FAISS # <-- UPDATED
from langchain.text_splitter import RecursiveCharacterTextSplitter

data_path = "processed_data"
output_path = "vector_store"

def create_vector_db():
    print("Loading documents...")
    # HERE IS THE FIX: We pass 'loader_kwargs' to specify the encoding for TextLoader.
    loader = DirectoryLoader(
        data_path, 
        glob="*.txt", 
        loader_cls=TextLoader,
        loader_kwargs={'encoding': 'utf-8'} # <-- FIX
    )
    documents = loader.load()
    # Tag each document chunk with mission/page metadata based on filename
    for doc in documents:
        source_path = doc.metadata.get('source', '')
        mission = os.path.splitext(os.path.basename(source_path))[0]
        doc.metadata['mission'] = mission
    print(f"Loaded {len(documents)} documents")

    print("Splitting documents...")
    # Use tighter chunks with overlap to keep contiguous context
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks")

    print("Creating embeddings... (This may take a moment the first time)")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    print("Creating vector store...")
    db = FAISS.from_documents(texts, embeddings)

    db.save_local(output_path)
    print(f"Vector store successfully saved to {output_path}")

if __name__ == "__main__":
    create_vector_db()

