# RAG Chatbot — End-to-End Capstone Project

A Retrieval-Augmented Generation (RAG) chatbot that answers questions grounded in web pages you provide — every answer is backed by retrieved source content, with sources shown alongside each response.

**Stack:** LangChain · ChromaDB · Groq (`llama-3.1-8b-instant`) · Sentence-Transformers (local embeddings, no embedding API cost) · Streamlit

## How it works

```
URLs → scrape (paragraph-only extraction) → chunk → embed → Chroma vector store → retrieve top-k → Groq LLM → answer + sources
```

The knowledge base can be **refined incrementally** — new URLs are added to the existing vector store without rebuilding it from scratch.

## Project structure

```
rag-chatbot/
├── notebook/
│   └── rag_chatbot_dev.ipynb   # development notebook: build, test, evaluate the pipeline
├── rag_pipeline.py              # shared core logic (imported by both the notebook and app.py)
├── app.py                       # production Streamlit app
├── requirements.txt
├── .env.example
└── .gitignore
```

`rag_pipeline.py` is the single source of truth for the retrieval/generation logic — the notebook and the app both import it, so they can never drift out of sync with each other.

## Setup

1. Clone the repo and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and add your [Groq API key](https://console.groq.com):
   ```
   GROQ_API_KEY=your_key_here
   ```

## Running the app locally

```bash
streamlit run app.py
```

1. Enter your Groq API key in the sidebar (or it's pre-filled from `.env`)
2. Paste in URLs (one per line) and click **Build Knowledge Base**
3. Ask questions in the chat — each answer shows which URLs it drew from
4. Add more URLs anytime and click **Add / Refine** to expand the knowledge base without starting over

## Running the notebook

Open `notebook/rag_chatbot_dev.ipynb` in Google Colab (GPU not required — embeddings run fine on CPU for small-to-medium knowledge bases). The notebook:
- Walks through the full pipeline step by step, with explanations
- Scrapes pages using paragraph-only extraction (filters out nav menus, sidebars, and boilerplate)
- Includes an evaluation section with a real labeled test set — measures both retrieval accuracy (correct source found) and answer quality (keyword match)
- Regenerates `rag_pipeline.py` and `app.py` via `%%writefile`, so the whole project can be reproduced from the notebook alone
- Includes a `pyngrok`-based section to launch the Streamlit app directly from Colab for a live demo

## Deployment

For a permanent deployment (rather than a temporary Colab/ngrok demo):
1. Push this repo to GitHub
2. Deploy on [share.streamlit.io](https://share.streamlit.io), pointing at `app.py`
3. Add `GROQ_API_KEY` under the app's Secrets settings (not hardcoded)

## Evaluation

The notebook evaluates the pipeline against a labeled test set, measuring:
- **Correct source retrieved** — whether the vector search found the right source page for a given question (isolates retriever quality)
- **Keyword match rate** — whether the generated answer contains expected key terms (a proxy for answer quality)

Both metrics scored 100% on the initial test set (3 questions across 3 distinct source pages).

## Possible extensions

- Re-ranking retrieved chunks (e.g. a cross-encoder) before passing them to the LLM
- LLM-as-judge evaluation instead of keyword matching
- Source de-duplication when the same URL is refined multiple times
- Streaming token-by-token responses in the chat UI
