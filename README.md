---
title: RAG Scholar
emoji: 📜
colorFrom: yellow
colorTo: gray
sdk: gradio
python_version: "3.11"
app_file: app.py
pinned: false
---

# RAG Scholar

Multi-PDF research chatbot powered by Groq, Gradio, FAISS, BM25, Hugging Face
embeddings, and CrossEncoder reranking.

## Local setup

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_key
```

Run the app:

```bash
python app.py
```

Open the local URL printed by Gradio.

## Notes

- The first upload or question may download the embedding and reranker models
  from Hugging Face. If the machine is offline, cache those models first.
- Uploaded PDFs are stored in a local FAISS index under `rag_index/`.
- BM25 chunks are stored in `rag_bm25.pkl`.

## Pipeline

```text
PDF upload
  -> 1200-character chunks
  -> BGE-large embeddings
  -> FAISS index + BM25 index

Query
  -> Hybrid retrieval: FAISS + BM25
  -> CrossEncoder rerank
  -> Groq answer with page citations
```
