"""
retriever_pinecone.py

Handles loading of various document types (PDF, DOCX, MD, TXT),
chunking, embedding, and uploading to Pinecone vector database.
"""

import os
import time
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from tqdm.auto import tqdm
from itertools import islice

from pinecone import Pinecone, ServerlessSpec
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader,
    TextLoader,
)
from voyageai import Client

from rag.paths import DATA_DIR

load_dotenv()

# ==========================================================
# Globals
# ==========================================================
_pinecone_index = None


# ==========================================================
# Helper Functions
# ==========================================================
def chunked_iterable(iterable, size: int):
    """Yield successive chunks from iterable of given size."""
    it = iter(iterable)
    while True:
        chunk = list(islice(it, size))
        if not chunk:
            break
        yield chunk


def embed_texts_in_batches(embedding_model, texts: List[str], batch_size: int = 50):
    """Safely embed texts in smaller batches to avoid API rate limits."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            batch_embeddings = embedding_model.embed_documents(batch)
            all_embeddings.extend(batch_embeddings)
        except Exception as e:
            print(f"Error embedding batch {i//batch_size + 1}: {e}")
            # Retry individually
            for text in batch:
                try:
                    single_embedding = embedding_model.embed_documents([text])[0]
                    all_embeddings.append(single_embedding)
                except Exception as inner_e:
                    print(f"Failed on single text: {inner_e}")
    return all_embeddings


# ==========================================================
# Pinecone Setup
# ==========================================================
def get_pinecone_index(index_name="agentic-robo-advisor"):
    """Return or create a Pinecone index."""
    global _pinecone_index
    if _pinecone_index is not None:
        return _pinecone_index

    api_key = os.getenv("PINECONE_API_KEY")
    environment = os.getenv("PINECONE_ENVIRONMENT")

    pc = Pinecone(api_key=api_key)
    spec = ServerlessSpec(cloud="aws", region=environment)

    # Create index if it doesn't exist
    existing = {idx["name"]: idx for idx in pc.list_indexes()}
    if index_name not in existing:
        print(f"Creating Pinecone index '{index_name}' ...")
        pc.create_index(
            name=index_name,
            dimension=1024,
            metric="cosine",
            spec=spec,
        )
        # Wait until ready
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)
        print("Index created.")

    _pinecone_index = pc.Index(index_name)
    return _pinecone_index


# ==========================================================
# Document Loader
# ==========================================================
def load_documents_from_dir(data_dir: str) -> List[Document]:
    """Load PDF, DOCX, MD, and TXT files into Document objects."""
    docs = []
    for file_path in Path(data_dir).glob("*"):
        ext = file_path.suffix.lower()

        try:
            if ext == ".pdf":
                loader = PyPDFLoader(str(file_path))
            elif ext == ".docx":
                loader = Docx2txtLoader(str(file_path))
            elif ext in (".md", ".markdown"):
                loader = UnstructuredMarkdownLoader(str(file_path))
            elif ext in (".txt",):
                loader = TextLoader(str(file_path))
            else:
                print(f"Skipping unsupported file type: {file_path.name}")
                continue

            file_docs = loader.load()
            for d in file_docs:
                d.metadata["source"] = file_path.name
            docs.extend(file_docs)

        except Exception as e:
            print(f"Error loading {file_path.name}: {e}")

    print(f"Loaded {len(docs)} documents from {data_dir}")
    return docs


# ==========================================================
# Vectorstore Pipeline
# ==========================================================
def build_pinecone_retriever(index_name="roboadvisor"):
    """Main pipeline to build/update Pinecone retriever."""
    docs = load_documents_from_dir(DATA_DIR)
    if not docs:
        raise ValueError(f"No documents found in {DATA_DIR}")

    # Split text
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    # Initialize embedding model
    embeddings = Client(
        api_key=os.getenv("VOYAGE_API_KEY")
    )

    # Compute embeddings
    print(f" Embedding {len(chunks)} chunks ...")
    texts = [chunk.page_content for chunk in chunks]

    # Embed in batches of 1000 (Voyage handles batching internally)
    response = embeddings.embed(texts=texts, model="voyage-finance-2")
    embeddings_list = response.embeddings
    vectors = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings_list)):
        vectors.append(
            (
                f"{chunk.metadata.get('source', 'doc')}-{i}",
                embedding,
                {"source": chunk.metadata.get("source"), "text": chunk.page_content},
            )
        )

    # Push to Pinecone
    index = get_pinecone_index(index_name)
    with tqdm(total=len(vectors), desc="Uploading to Pinecone") as pbar:
        for batch in chunked_iterable(vectors, 100):
            index.upsert(vectors=batch)
            pbar.update(len(batch))

    print(f"Upload complete. Indexed {len(vectors)} chunks.")
    return index


def query_pinecone(query: str, index_name="roboadvisor", top_k: int = 3):
    """Retrieve the most relevant documents for a user query."""
    index = get_pinecone_index(index_name)
    embeddings = Client(
        api_key=os.getenv("VOYAGE_API_KEY")
    )

    # Generate query embedding
    response = embeddings.embed(texts=[query], model="voyage-finance-2")
    query_vector = response.embeddings[0]

    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )

    # Format results into readable context text
    matches = results.get("matches", [])
    if not matches:
        return ""

    context = "\n\n".join(
        [match.get('metadata', {}).get('text', '') for match in matches if match.get('metadata', {}).get('text')])
    return context


