"""
rag_pipeline.py

Core RAG (Retrieval-Augmented Generation) logic shared between the development
notebook and the Streamlit app. Keeping this in one module means the notebook
and the app can never drift out of sync with each other.

Pipeline: URLs -> scrape -> chunk -> embed -> vector store (Chroma) -> retrieve -> Groq LLM -> answer
"""

import os
import torch
from typing import List, Optional

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_PERSIST_DIR = "chroma_db"
DEFAULT_LLM_MODEL = "llama-3.1-8b-instant"


# ---------------------------------------------------------------------------
# 1. Loading web pages
# ---------------------------------------------------------------------------

def load_documents_from_urls(urls: List[str], verbose: bool = True):
    """
    Scrapes a list of URLs into LangChain Document objects.
    Skips (and reports) any URL that fails to load, instead of crashing the
    whole pipeline over one bad link.
    """
    documents = []
    failed = []

    for url in urls:
        url = url.strip()
        if not url:
            continue
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            documents.extend(docs)
            if verbose:
                print(f"  Loaded: {url}  ({len(docs[0].page_content)} chars)")
        except Exception as e:
            failed.append((url, str(e)))
            if verbose:
                print(f"  FAILED: {url}  -> {e}")

    if verbose:
        print(f"\nLoaded {len(documents)} document(s), {len(failed)} failure(s).")

    return documents, failed


# ---------------------------------------------------------------------------
# 2. Chunking
# ---------------------------------------------------------------------------

def split_documents(documents, chunk_size: int = 1000, chunk_overlap: int = 150):
    """Splits documents into overlapping chunks for embedding/retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


# ---------------------------------------------------------------------------
# 3. Embeddings + Vector store
# ---------------------------------------------------------------------------

def get_embedding_model():
    """
    Loads a local sentence-transformers embedding model — free, no API cost,
    runs on GPU automatically if available (falls back to CPU otherwise).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"batch_size": 32, "normalize_embeddings": True},
    )


def build_vectorstore(chunks, persist_directory: str = DEFAULT_PERSIST_DIR):
    """
    Builds a fresh Chroma vector store from document chunks and persists it
    to disk. Embeds in batches (via encode_kwargs above) rather than one
    document at a time, which is what caused the slow/unresponsive embedding
    step in earlier CPU-only runs.
    """
    embeddings = get_embedding_model()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
    )
    vectorstore.persist()
    return vectorstore


def load_vectorstore(persist_directory: str = DEFAULT_PERSIST_DIR):
    """Reloads an existing vector store from disk (no re-embedding needed)."""
    embeddings = get_embedding_model()
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)


def add_urls_to_vectorstore(urls: List[str], vectorstore, chunk_size: int = 1000, chunk_overlap: int = 150):
    """
    Refines/extends an existing vector store with new URLs, without rebuilding
    from scratch. This is what lets the knowledge base grow incrementally.
    """
    documents, failed = load_documents_from_urls(urls)
    if not documents:
        return vectorstore, failed

    chunks = split_documents(documents, chunk_size, chunk_overlap)
    vectorstore.add_documents(chunks)
    vectorstore.persist()
    return vectorstore, failed


# ---------------------------------------------------------------------------
# 4. LLM + Conversational RAG chain
# ---------------------------------------------------------------------------

def get_qa_chain(vectorstore, groq_api_key: str, model_name: str = DEFAULT_LLM_MODEL, k: int = 4):
    """
    Builds a conversational retrieval chain: retrieves top-k relevant chunks
    for each question, feeds them + chat history to the Groq LLM, and returns
    an answer along with the source documents used.
    """
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=model_name,
        temperature=0.2,
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
    )
    return chain


def ask(chain, question: str):
    """
    Runs a question through the chain. Returns (answer, list_of_source_urls).
    """
    result = chain.invoke({"question": question})
    answer = result["answer"]
    sources = sorted(set(
        doc.metadata.get("source", "unknown") for doc in result.get("source_documents", [])
    ))
    return answer, sources
