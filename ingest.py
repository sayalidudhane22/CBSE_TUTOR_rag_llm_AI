"""
ingest.py
Ingestion pipeline: loads CBSE science content, splits into chunks,
creates embeddings using sentence-transformers, and stores in ChromaDB.

Run once to build the vector database:
    python ingest.py
"""

import os
from pathlib import Path

from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


# ── Config ──────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data"
CHROMA_DIR = Path(__file__).parent / "chroma_db"

# Free, high-quality multilingual embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 500        # Characters per chunk
CHUNK_OVERLAP = 80      # Overlap between chunks for context continuity
# ────────────────────────────────────────────────────────────────────────────


def load_documents():
    """Load all .txt and .pdf files from the data directory."""
    documents = []

    # Load .txt files
    txt_files = list(DATA_DIR.glob("*.txt"))
    for txt_file in txt_files:
        loader = TextLoader(str(txt_file), encoding="utf-8")
        docs = loader.load()
        # Add source metadata
        for doc in docs:
            doc.metadata["source"] = txt_file.name
        documents.extend(docs)
        print(f"  ✓ Loaded: {txt_file.name} ({len(docs)} doc(s))")

    # Load .pdf files
    pdf_files = list(DATA_DIR.glob("*.pdf"))
    for pdf_file in pdf_files:
        loader = PyPDFLoader(str(pdf_file))
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = pdf_file.name
        documents.extend(docs)
        print(f"  ✓ Loaded PDF: {pdf_file.name} ({len(docs)} page(s))")

    return documents


def split_documents(documents):
    """Split documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    return chunks


def get_embeddings():
    """Load the HuggingFace embedding model (free, local)."""
    print(f"  Loading embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return embeddings


def build_vectorstore(chunks, embeddings):
    """Create or update ChromaDB vector store."""
    CHROMA_DIR.mkdir(exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name="cbse_science",
    )
    vectorstore.persist()
    return vectorstore


def load_vectorstore(embeddings):
    """Load existing ChromaDB vector store from disk."""
    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name="cbse_science",
    )
    return vectorstore


def vectorstore_exists():
    """Check if a persisted ChromaDB exists."""
    return (CHROMA_DIR / "chroma.sqlite3").exists()


def run_ingestion():
    """Full ingestion pipeline."""
    print("\n🔧 CBSE Tutor — Ingestion Pipeline")
    print("=" * 45)

    print("\n📂 Loading documents...")
    documents = load_documents()
    if not documents:
        print("  ⚠️  No documents found in data/ directory!")
        return None

    print(f"\n✂️  Splitting into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    chunks = split_documents(documents)
    print(f"  ✓ Created {len(chunks)} chunks from {len(documents)} document(s)")

    print("\n🔢 Loading embedding model...")
    embeddings = get_embeddings()
    print("  ✓ Embedding model ready")

    print("\n💾 Building ChromaDB vector store...")
    vectorstore = build_vectorstore(chunks, embeddings)
    print(f"  ✓ Stored {len(chunks)} vectors in {CHROMA_DIR}")

    print("\n✅ Ingestion complete!")
    return vectorstore


if __name__ == "__main__":
    run_ingestion()
