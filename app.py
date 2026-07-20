"""
app.py

Streamlit front-end for the RAG chatbot. Lets a user paste in URLs to build
a knowledge base, refine it by adding more URLs later, and chat with an
LLM (Groq) that answers grounded in the scraped content — with sources shown.

Run locally with:
    streamlit run app.py

Run from Colab (no native Streamlit hosting there) using pyngrok — see the
notebook's final section for that setup.
"""

import os
import streamlit as st
from dotenv import load_dotenv

from rag_pipeline import (
    load_documents_from_urls,
    split_documents,
    build_vectorstore,
    load_vectorstore,
    add_urls_to_vectorstore,
    get_qa_chain,
    ask,
    DEFAULT_PERSIST_DIR,
)

load_dotenv()

st.set_page_config(page_title="RAG Chatbot", page_icon="💬", layout="wide")


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chain" not in st.session_state:
    st.session_state.chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (role, content, sources)
if "sources_used" not in st.session_state:
    st.session_state.sources_used = []  # all URLs ever added


# ---------------------------------------------------------------------------
# Sidebar — knowledge base setup
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Knowledge Base")

    groq_api_key = st.text_input(
        "Groq API Key",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
        help="Get a free key at console.groq.com",
    )

    url_input = st.text_area(
        "Paste URLs (one per line)",
        height=150,
        placeholder="https://example.com/article-1\nhttps://example.com/article-2",
    )

    col1, col2 = st.columns(2)

    with col1:
        build_clicked = st.button("Build Knowledge Base", use_container_width=True)
    with col2:
        refine_clicked = st.button("Add / Refine", use_container_width=True, disabled=st.session_state.vectorstore is None)

    st.divider()

    if os.path.exists(DEFAULT_PERSIST_DIR) and st.session_state.vectorstore is None:
        if st.button("Load existing knowledge base", use_container_width=True):
            with st.spinner("Loading saved knowledge base..."):
                st.session_state.vectorstore = load_vectorstore(DEFAULT_PERSIST_DIR)
                if groq_api_key:
                    st.session_state.chain = get_qa_chain(st.session_state.vectorstore, groq_api_key)
            st.success("Loaded.")

    if st.session_state.sources_used:
        st.subheader("Sources in knowledge base")
        for s in st.session_state.sources_used:
            st.caption(f"• {s}")

    st.divider()
    if st.button("Reset conversation", use_container_width=True):
        st.session_state.chat_history = []
        if st.session_state.vectorstore is not None and groq_api_key:
            st.session_state.chain = get_qa_chain(st.session_state.vectorstore, groq_api_key)
        st.rerun()


# ---------------------------------------------------------------------------
# Build / refine actions
# ---------------------------------------------------------------------------
urls = [u for u in url_input.split("\n") if u.strip()] if url_input else []

if build_clicked:
    if not groq_api_key:
        st.sidebar.error("Enter your Groq API key first.")
    elif not urls:
        st.sidebar.error("Paste at least one URL first.")
    else:
        with st.spinner(f"Scraping {len(urls)} URL(s) and building knowledge base..."):
            documents, failed = load_documents_from_urls(urls, verbose=False)
            if not documents:
                st.sidebar.error("Could not load any of the provided URLs.")
            else:
                chunks = split_documents(documents)
                st.session_state.vectorstore = build_vectorstore(chunks, DEFAULT_PERSIST_DIR)
                st.session_state.chain = get_qa_chain(st.session_state.vectorstore, groq_api_key)
                st.session_state.sources_used = list(set(st.session_state.sources_used + urls))
                st.session_state.chat_history = []
                if failed:
                    st.sidebar.warning(f"{len(failed)} URL(s) failed to load:\n" + "\n".join(u for u, _ in failed))
                st.sidebar.success(f"Knowledge base built from {len(documents)} page(s), {len(chunks)} chunks.")

if refine_clicked:
    if not urls:
        st.sidebar.error("Paste at least one URL to add first.")
    else:
        with st.spinner(f"Adding {len(urls)} URL(s) to the knowledge base..."):
            st.session_state.vectorstore, failed = add_urls_to_vectorstore(urls, st.session_state.vectorstore)
            st.session_state.chain = get_qa_chain(st.session_state.vectorstore, groq_api_key)
            st.session_state.sources_used = list(set(st.session_state.sources_used + urls))
            if failed:
                st.sidebar.warning(f"{len(failed)} URL(s) failed to load:\n" + "\n".join(u for u, _ in failed))
            st.sidebar.success("Knowledge base refined.")


# ---------------------------------------------------------------------------
# Main chat interface
# ---------------------------------------------------------------------------
st.title("💬 RAG Chatbot")
st.caption("Ask questions grounded in the web pages you've added to the knowledge base.")

if st.session_state.chain is None:
    st.info("Add URLs and click **Build Knowledge Base** in the sidebar to get started.")
else:
    for role, content, sources in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(content)
            if sources:
                with st.expander("Sources"):
                    for s in sources:
                        st.caption(s)

    question = st.chat_input("Ask a question...")
    if question:
        st.session_state.chat_history.append(("user", question, None))
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, sources = ask(st.session_state.chain, question)
                st.markdown(answer)
                if sources:
                    with st.expander("Sources"):
                        for s in sources:
                            st.caption(s)

        st.session_state.chat_history.append(("assistant", answer, sources))
