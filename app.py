import os
import pickle
import warnings
from pathlib import Path

from dotenv import load_dotenv


def _patch_typing_extensions_for_langchain_protocol():
    """Allow current LangChain protocol typings to import on Python 3.13."""
    try:
        import typing_extensions as te

        typed_dict_meta = type(te.TypedDict("_CompatTypedDict", {}))
        original_new = typed_dict_meta.__new__

        def typed_dict_new(cls, name, bases, namespace, total=True, closed=False, **kwargs):
            kwargs.pop("extra_items", None)
            try:
                return original_new(
                    cls,
                    name,
                    bases,
                    namespace,
                    total=total,
                    closed=closed,
                    **kwargs,
                )
            except TypeError:
                return original_new(cls, name, bases, namespace, total=total)

        typed_dict_meta.__new__ = typed_dict_new
    except Exception:
        pass


_patch_typing_extensions_for_langchain_protocol()
load_dotenv()
warnings.filterwarnings(
    "ignore",
    message="`langchain-community` is being sunset.*",
    category=DeprecationWarning,
)

import gradio as gr
import torch
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_classic.retrievers import (
        ContextualCompressionRetriever,
        EnsembleRetriever,
    )
    from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
except ImportError:
    from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
    from langchain.retrievers.document_compressors import CrossEncoderReranker


GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MODEL_ID = os.getenv("MODEL_ID", "llama-3.3-70b-versatile")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.4"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "3000"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-large-en-v1.5")
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "300"))
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "20"))
TOP_K = int(os.getenv("TOP_K", "5"))
FAISS_W = float(os.getenv("FAISS_W", "0.6"))
BM25_W = float(os.getenv("BM25_W", "0.4"))
MEMORY_TURNS = int(os.getenv("MEMORY_TURNS", "4"))
INDEX_DIR = os.getenv("INDEX_DIR", "rag_index")
BM25_PATH = os.getenv("BM25_PATH", "rag_bm25.pkl")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SYSTEM = (
    "You are an expert research assistant with deep knowledge of the documents "
    "in the knowledge base. Answer questions thoroughly, naturally, and in clean "
    "Markdown. Adapt the answer format to the question: use a compact table for "
    "definitions, numbered steps for processes, bullets for comparisons, and short "
    "paragraphs for explanations. Always start with a one-sentence direct answer, "
    "then organize the evidence under clear headings. Cite every factual claim with "
    "page references like (p.12). Quote directly only when useful and keep quotes "
    "short. If the retrieved passages do not cover the question, say so honestly "
    "and suggest a better follow-up question."
)

TEMPLATE = """{system}

{history}Retrieved passages:
{context}

Question: {question}
Answer:"""

prompt = PromptTemplate(
    input_variables=["system", "history", "context", "question"],
    template=TEMPLATE,
)

embeddings = None
vectorstore = None
all_chunks = []
bm25 = None
hybrid = None
retriever = None
chain = None
llm = None
cross_encoder = None
reranker = None
doc_names = []
startup_notice = ""
index_loaded = False


def fmt_docs(docs):
    if not docs:
        return "No documents are indexed yet."
    return "\n\n---\n\n".join(
        f"[p.{doc.metadata.get('page', '?')} | "
        f"{Path(doc.metadata.get('source', '')).name}]\n{doc.page_content}"
        for doc in docs
    )


def fmt_history(history):
    if not history:
        return ""

    lines = ["Previous conversation:"]
    for item in history[-MEMORY_TURNS * 2 :]:
        if not isinstance(item, dict):
            continue
        content = (item.get("content") or "").strip()
        if not content:
            continue
        if item.get("role") == "user":
            lines.append(f"User: {content[:240]}")
        elif item.get("role") == "assistant":
            answer = content.split("\n\nSources:")[0][:500]
            lines.append(f"Assistant: {answer}")
    return "\n".join(lines) + "\n\n" if len(lines) > 1 else ""


def get_pages(docs):
    seen = set()
    pages = []
    for doc in docs:
        page = doc.metadata.get("page", "?")
        source = Path(doc.metadata.get("source", "")).name
        key = f"{source}:p{page}"
        if key not in seen:
            seen.add(key)
            pages.append(f"{source} p.{page}")
    return pages


def clean(text):
    for token in ("<|assistant|>", "<|end|>", "<|user|>", "<|system|>", "<think>", "</think>"):
        text = text.replace(token, "")
    return text


def init_embeddings():
    global embeddings
    if embeddings is None:
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name=EMBED_MODEL,
                model_kwargs={"device": DEVICE},
                encode_kwargs={"normalize_embeddings": True},
            )
        except Exception as exc:
            raise RuntimeError(
                "Could not load the embedding model. Connect to the internet once "
                "so Hugging Face can download it, or use a locally cached model."
            ) from exc


def init_llm():
    global llm, chain
    if chain is not None:
        return
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set. Add it to the .env file first.")
    llm = ChatGroq(
        model=MODEL_ID,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        api_key=GROQ_API_KEY,
    )
    chain = prompt | llm | StrOutputParser()


def build_retriever():
    global bm25, hybrid, retriever, cross_encoder, reranker
    if not all_chunks or vectorstore is None:
        retriever = None
        return

    bm25 = BM25Retriever.from_documents(all_chunks)
    bm25.k = RETRIEVAL_K
    faiss_ret = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})
    hybrid = EnsembleRetriever(retrievers=[faiss_ret, bm25], weights=[FAISS_W, BM25_W])

    if cross_encoder is None:
        cross_encoder = HuggingFaceCrossEncoder(model_name=RERANK_MODEL)
        reranker = CrossEncoderReranker(model=cross_encoder, top_n=TOP_K)

    retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=hybrid,
    )


def index_pdf(pdf_path):
    global vectorstore, all_chunks, doc_names, index_loaded
    init_embeddings()
    if vectorstore is None and all_chunks and Path(INDEX_DIR).exists():
        load_saved_index()

    pages = PyPDFLoader(pdf_path).load()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    ).split_documents(pages)

    name = Path(pdf_path).name
    new_store = FAISS.from_documents(chunks, embeddings)
    if vectorstore is None:
        vectorstore = new_store
    else:
        vectorstore.merge_from(new_store)

    all_chunks.extend(chunks)
    Path(INDEX_DIR).mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(INDEX_DIR)
    with open(BM25_PATH, "wb") as fh:
        pickle.dump(all_chunks, fh)

    if name not in doc_names:
        doc_names.append(name)

    build_retriever()
    index_loaded = True
    return name, len(chunks)


def _names_from_chunks(chunks):
    return sorted(
        {
            Path(doc.metadata.get("source", "")).name
            for doc in chunks
            if doc.metadata.get("source")
        }
    )


def load_saved_metadata():
    global all_chunks, doc_names
    if not Path(BM25_PATH).exists():
        return
    with open(BM25_PATH, "rb") as fh:
        all_chunks = pickle.load(fh)
    doc_names = _names_from_chunks(all_chunks)


def load_saved_index():
    global vectorstore, all_chunks, doc_names, index_loaded
    init_embeddings()
    if not (Path(INDEX_DIR).exists() and Path(BM25_PATH).exists()):
        return

    vectorstore = FAISS.load_local(
        INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    with open(BM25_PATH, "rb") as fh:
        all_chunks = pickle.load(fh)

    doc_names = _names_from_chunks(all_chunks)
    build_retriever()
    index_loaded = True


def retrieve(question):
    if not index_loaded and Path(INDEX_DIR).exists() and Path(BM25_PATH).exists():
        load_saved_index()
    if retriever is None:
        return []
    return retriever.invoke(question)


def stream_answer(question, history=None):
    init_llm()
    docs = retrieve(question)
    pages = get_pages(docs)
    answer = ""

    for chunk in chain.stream(
        {
            "system": SYSTEM,
            "history": fmt_history(history or []),
            "context": fmt_docs(docs),
            "question": question,
        }
    ):
        chunk = clean(chunk)
        if chunk:
            answer += chunk
            yield answer

    if pages:
        answer = answer.rstrip()
        answer += "\n\n### Sources\n" + "\n".join(f"- {page}" for page in pages)
        yield answer


def doc_list_html():
    if not doc_names:
        return """
        <div class="empty-state">
          <span>No PDFs indexed yet</span>
          <small>Upload documents to start asking grounded questions.</small>
        </div>
        """
    return "".join(
        f"""
        <div class="doc-list-item">
          <span class="doc-dot"></span>
          <span>{name}</span>
        </div>
        """
        for name in doc_names
    )


def health_html():
    index_state = f"{len(doc_names)} documents" if doc_names else "Empty"
    return f"""
    <div class="health-grid">
      <div><span>Library</span><strong>{index_state}</strong></div>
      <div><span>Answers</span><strong>Cited from uploaded PDFs</strong></div>
    </div>
    """


try:
    load_saved_metadata()
except Exception as exc:
    startup_notice = f"Metadata not loaded: {exc}"
    print(f"Startup: {exc}")


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@500;600;700&family=Spectral:wght@400;500;600;700&display=swap');

:root {
  --night: #17110d;
  --night-2: #241810;
  --parchment: #efe1bf;
  --parchment-2: #f7edcf;
  --papyrus: #fff7df;
  --ink: #24170f;
  --muted: #66513d;
  --line: #b78d55;
  --line-soft: rgba(183, 141, 85, 0.42);
  --gold: #d8ad5c;
  --copper: #9f4f2f;
  --cedar: #34533d;
  --lapis: #244b68;
  --shadow: 0 24px 70px rgba(23, 17, 13, 0.3);
}

body,
.gradio-container {
  background:
    radial-gradient(circle at 15% 10%, rgba(216,173,92,0.18), transparent 28%),
    radial-gradient(circle at 84% 4%, rgba(36,75,104,0.18), transparent 24%),
    linear-gradient(135deg, rgba(23,17,13,0.98), rgba(52,34,22,0.94) 46%, rgba(23,17,13,0.98)),
    var(--night) !important;
  color: var(--ink) !important;
  font-family: "Spectral", Georgia, serif !important;
}

.gradio-container {
  max-width: 1280px !important;
  margin: 0 auto !important;
}

.app-shell {
  padding: 26px 18px 18px;
}

.hero {
  position: relative;
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 24px;
  align-items: end;
  padding: 30px;
  border: 1px solid rgba(216,173,92,0.46);
  border-radius: 8px;
  background:
    linear-gradient(90deg, rgba(255,247,223,0.96), rgba(239,225,191,0.92)),
    var(--parchment);
  box-shadow: var(--shadow);
  overflow: hidden;
}

.hero:before {
  content: "";
  position: absolute;
  inset: 10px;
  border: 1px solid rgba(159,79,47,0.34);
  pointer-events: none;
}

.hero:after {
  content: "";
  position: absolute;
  inset: 0;
  background:
    repeating-linear-gradient(0deg, rgba(36,23,15,0.035) 0 1px, transparent 1px 7px),
    radial-gradient(circle at 50% -20%, rgba(255,255,255,0.4), transparent 28%);
  pointer-events: none;
}

.hero > * {
  position: relative;
  z-index: 1;
}

.eyebrow {
  color: var(--copper);
  font-family: "Cinzel", serif;
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.18em;
  margin: 0 0 10px;
  text-transform: uppercase;
}

.hero h1 {
  color: var(--ink);
  font-family: "Cinzel", serif;
  font-size: clamp(2.4rem, 6vw, 5.4rem);
  font-weight: 700;
  letter-spacing: 0;
  line-height: 0.92;
  margin: 0;
}

.hero p {
  color: var(--muted);
  font-size: 1.04rem;
  line-height: 1.55;
  margin: 14px 0 0;
  max-width: 700px;
}

.status-strip {
  align-self: stretch;
  background: linear-gradient(180deg, #25170e, #3b2617);
  border: 1px solid rgba(216,173,92,0.55);
  color: var(--parchment-2);
  min-width: 285px;
  padding: 20px;
  border-radius: 8px;
  box-shadow: inset 0 0 0 1px rgba(255,247,223,0.08);
}

.status-strip span {
  color: var(--gold);
  display: block;
  font-family: "Cinzel", serif;
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.13em;
  text-transform: uppercase;
}

.status-strip strong {
  display: block;
  font-family: "Cinzel", serif;
  font-size: 1.7rem;
  line-height: 1;
  margin-top: 10px;
}

.main-tabs {
  padding: 0 18px 24px;
}

.main-tabs > .tab-nav {
  background: transparent !important;
  border-bottom: 1px solid rgba(216,173,92,0.35) !important;
  gap: 8px !important;
  padding: 18px 0 0 !important;
}

.main-tabs > .tab-nav button {
  background: rgba(255,247,223,0.08) !important;
  border: 1px solid rgba(216,173,92,0.22) !important;
  border-radius: 8px 8px 0 0 !important;
  color: #ead39c !important;
  font-family: "Cinzel", serif !important;
  font-size: 0.82rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.08em !important;
  padding: 12px 18px !important;
  text-transform: uppercase !important;
}

.main-tabs > .tab-nav button.selected {
  background: var(--parchment-2) !important;
  border-color: rgba(216,173,92,0.78) !important;
  color: var(--ink) !important;
}

.workspace {
  gap: 18px !important;
  margin-top: 18px;
}

.chat-column,
.side-panel,
.upload-panel,
.settings-panel {
  background: linear-gradient(180deg, rgba(255,247,223,0.98), rgba(239,225,191,0.97));
  border: 1px solid rgba(216,173,92,0.58);
  border-radius: 8px;
  box-shadow: var(--shadow);
  padding: 16px;
}

.side-panel {
  background: linear-gradient(180deg, rgba(48,31,20,0.98), rgba(35,24,17,0.98));
  color: var(--parchment-2);
}

.side-panel p,
.side-panel label,
.side-panel textarea,
.side-panel .upload-copy,
.side-panel .doc-list-item,
.side-panel .empty-state,
.side-panel .empty-state small {
  color: var(--parchment-2) !important;
}

.side-panel textarea {
  background: rgba(255,247,223,0.1) !important;
  border-color: rgba(216,173,92,0.38) !important;
}

.side-panel .doc-list-item {
  background: rgba(255,247,223,0.1);
}

div[data-testid="chatbot"] {
  background:
    linear-gradient(180deg, rgba(255,247,223,0.96), rgba(249,235,198,0.96)) !important;
  border: 1px solid var(--line-soft) !important;
  border-radius: 8px !important;
}

div[data-testid="chatbot"] .prose,
div[data-testid="chatbot"] p,
div[data-testid="chatbot"] li,
div[data-testid="chatbot"] td,
div[data-testid="chatbot"] th {
  color: var(--ink) !important;
  font-size: 1rem !important;
  line-height: 1.65 !important;
}

div[data-testid="chatbot"] h1,
div[data-testid="chatbot"] h2,
div[data-testid="chatbot"] h3 {
  color: var(--copper) !important;
  font-family: "Cinzel", serif !important;
  letter-spacing: 0 !important;
}

div[data-testid="chatbot"] table {
  border-collapse: collapse !important;
  width: 100% !important;
}

div[data-testid="chatbot"] th {
  background: rgba(159,79,47,0.12) !important;
  font-family: "Cinzel", serif !important;
}

div[data-testid="chatbot"] th,
div[data-testid="chatbot"] td {
  border: 1px solid rgba(183,141,85,0.48) !important;
  padding: 9px 10px !important;
}

.message-wrap {
  border-radius: 8px !important;
}

textarea,
div[data-testid="textbox"] textarea {
  background: var(--papyrus) !important;
  border: 1px solid rgba(183,141,85,0.68) !important;
  border-radius: 8px !important;
  color: var(--ink) !important;
  font-family: "Spectral", Georgia, serif !important;
  font-size: 1rem !important;
  line-height: 1.45 !important;
  min-height: 58px !important;
}

textarea:focus {
  border-color: var(--copper) !important;
  box-shadow: 0 0 0 3px rgba(159,79,47,0.16) !important;
}

button[variant="primary"],
.gr-button-primary {
  background: linear-gradient(180deg, #a95a36, #7b351f) !important;
  border: 1px solid #bd7a3f !important;
  border-radius: 8px !important;
  color: #fff7df !important;
  font-family: "Cinzel", serif !important;
  font-weight: 700 !important;
  min-height: 46px !important;
  text-transform: uppercase !important;
}

button[variant="primary"]:hover {
  background: linear-gradient(180deg, #bd6841, #8a3d24) !important;
}

button[variant="secondary"] {
  background: rgba(255,247,223,0.92) !important;
  border: 1px solid rgba(183,141,85,0.66) !important;
  border-radius: 8px !important;
  color: var(--ink) !important;
  font-family: "Cinzel", serif !important;
  font-weight: 700 !important;
}

.panel-title {
  align-items: center;
  border-bottom: 1px solid rgba(183,141,85,0.42);
  display: flex;
  justify-content: space-between;
  margin-bottom: 12px;
  padding-bottom: 12px;
}

.panel-title h3 {
  color: inherit;
  font-family: "Cinzel", serif;
  font-size: 1.25rem;
  line-height: 1;
  margin: 0;
}

.panel-title span {
  color: var(--gold);
  font-family: "Cinzel", serif;
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.metric-grid,
.health-grid {
  display: grid;
  gap: 10px;
  grid-template-columns: repeat(2, minmax(0, 1fr));
}

.metric-grid div,
.health-grid div {
  background: rgba(255,247,223,0.08);
  border: 1px solid rgba(216,173,92,0.25);
  border-radius: 8px;
  padding: 12px;
}

.health-grid div {
  background: rgba(255,247,223,0.78);
}

.metric-grid span,
.health-grid span {
  color: var(--gold);
  display: block;
  font-family: "Cinzel", serif;
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  margin-bottom: 6px;
  text-transform: uppercase;
}

.health-grid span {
  color: var(--copper);
}

.metric-grid strong,
.health-grid strong {
  color: inherit;
  display: block;
  font-size: 0.95rem;
  overflow-wrap: anywhere;
}

.doc-list {
  display: grid;
  gap: 8px;
}

.doc-list-item {
  align-items: center;
  background: rgba(255,247,223,0.1);
  border: 1px solid rgba(216,173,92,0.28);
  border-radius: 8px;
  display: flex;
  gap: 10px;
  padding: 10px 12px;
}

.doc-dot {
  background: var(--gold);
  border-radius: 999px;
  height: 8px;
  width: 8px;
}

.empty-state {
  background: rgba(255,247,223,0.08);
  border: 1px dashed rgba(216,173,92,0.42);
  border-radius: 8px;
  color: var(--parchment-2);
  padding: 14px;
}

.empty-state span,
.empty-state small {
  display: block;
}

.empty-state span {
  font-weight: 700;
}

.empty-state small {
  color: #cfb885;
  margin-top: 4px;
}

.question-panel {
  background: rgba(36,75,104,0.12);
  border: 1px solid rgba(36,75,104,0.28);
  border-radius: 8px;
  color: var(--lapis);
  line-height: 1.45;
  margin: 0 0 12px;
  padding: 12px 14px;
}

.question-panel strong {
  display: block;
  font-family: "Cinzel", serif;
  margin-bottom: 6px;
}

.question-panel ul {
  margin: 0;
  padding-left: 18px;
}

.examples-table td {
  background: var(--papyrus) !important;
  border: 1px solid rgba(183,141,85,0.58) !important;
  border-radius: 8px !important;
  color: var(--ink) !important;
  cursor: pointer !important;
  font-size: 0.95rem !important;
  padding: 10px 12px !important;
}

.examples-table td:hover {
  background: #f6e4b6 !important;
  border-color: var(--copper) !important;
}

.upload-copy {
  color: var(--muted);
  line-height: 1.6;
  margin: 0 0 14px;
}

.pipeline {
  background: rgba(52,83,61,0.12);
  border: 1px solid rgba(52,83,61,0.24);
  border-radius: 8px;
  color: var(--cedar);
  line-height: 1.6;
  padding: 14px;
}

.config-table {
  border-collapse: collapse;
  width: 100%;
}

.config-table td {
  border-bottom: 1px solid rgba(183,141,85,0.42);
  padding: 10px 0;
}

.config-table td:first-child {
  color: var(--muted);
  width: 42%;
}

.config-table td:last-child {
  color: var(--ink);
  font-weight: 700;
}

footer,
.built-with {
  display: none !important;
}

@media (max-width: 760px) {
  .app-shell,
  .main-tabs {
    padding-left: 10px;
    padding-right: 10px;
  }

  .hero {
    grid-template-columns: 1fr;
    padding: 22px;
  }

  .status-strip {
    min-width: 0;
  }

  .metric-grid,
  .health-grid {
    grid-template-columns: 1fr;
  }
}
"""

EXAMPLES = [
    "Create a glossary of the key terms with short definitions and page citations.",
    "Summarize the main argument in five bullet points with evidence.",
    "Which ideas are repeated most often across the document?",
    "Compare direct strategy and indirect strategy using cited examples.",
    "List the practical lessons a leader should take from these passages.",
    "What are the strongest quotes, and why do they matter?",
    "Turn this document into exam notes with definitions, themes, and questions.",
    "What should I ask next to understand this text more deeply?",
]


def respond(message, history):
    history = history or []
    message = (message or "").strip()
    if not message:
        yield "", history
        return

    working_history = history + [{"role": "user", "content": message}]
    yield "", working_history + [{"role": "assistant", "content": "Consulting the indexed scrolls..."}]

    try:
        for answer in stream_answer(message, history):
            yield "", working_history + [{"role": "assistant", "content": answer}]
    except Exception as exc:
        yield "", working_history + [{"role": "assistant", "content": f"Error: {exc}"}]


def clear_chat():
    return [], ""


def do_upload(files):
    if not files:
        return "No files selected. Choose one or more PDF files.", doc_list_html(), doc_list_html()

    messages = []
    for file in files if isinstance(files, list) else [files]:
        try:
            file_path = file if isinstance(file, str) else file.name
            name, chunk_count = index_pdf(file_path)
            messages.append(f"Indexed {name}: {chunk_count} chunks")
        except Exception as exc:
            failed_path = file if isinstance(file, str) else getattr(file, "name", "file")
            messages.append(f"Error indexing {Path(failed_path).name}: {exc}")

    html = doc_list_html()
    return "\n".join(messages), html, html


with gr.Blocks(title="RAG Scholar") as demo:
    gr.HTML(
        f"""
        <main class="app-shell">
          <section class="hero">
            <div>
              <p class="eyebrow">Hybrid PDF Retrieval</p>
              <h1>RAG Scholar</h1>
              <p>
                A focused document research desk for asking cited questions across
                uploaded PDFs with grounded retrieval and page-level sources.
              </p>
            </div>
            <aside class="status-strip">
              <span>Knowledge Base</span>
              <strong>{len(doc_names)} PDFs</strong>
            </aside>
          </section>
        </main>
        """
    )

    with gr.Row(elem_classes=["workspace"]):
        with gr.Column(scale=7, elem_classes=["chat-column"]):
            chatbot = gr.Chatbot(
                height=540,
                render_markdown=True,
                show_label=False,
                layout="bubble",
                placeholder="Upload PDFs, then ask a question grounded in your library.",
            )
            with gr.Row():
                msg_box = gr.Textbox(
                    placeholder="Ask about your documents...",
                    show_label=False,
                    lines=2,
                    scale=9,
                )
                send_btn = gr.Button("Ask", variant="primary", scale=1)
            with gr.Row():
                clear_btn = gr.Button("Clear", variant="secondary", size="sm")
            gr.HTML(
                """
                <div class="question-panel">
                  <strong>Good question patterns</strong>
                  <ul>
                    <li>Ask for a glossary, comparison, timeline, or cited summary.</li>
                    <li>Name the format you want: table, bullets, study notes, or checklist.</li>
                    <li>Request page citations when you need evidence.</li>
                  </ul>
                </div>
                """
            )
            gr.Examples(examples=EXAMPLES, inputs=msg_box, label="Question starters")

        with gr.Column(scale=3, elem_classes=["side-panel"]):
            gr.HTML(
                """
                <div class="panel-title">
                  <h3>Build the library</h3>
                  <span>PDF only</span>
                </div>
                <p class="upload-copy">
                  Choose PDFs to index them into the local knowledge base.
                </p>
                """
            )
            upload_btn = gr.UploadButton(
                "Choose PDFs and index",
                variant="primary",
                file_types=[".pdf"],
                file_count="multiple",
                type="filepath",
            )
            upload_status = gr.Textbox(
                label="Status",
                interactive=False,
                lines=4,
                placeholder="Choose PDFs to index them.",
            )
            gr.HTML(
                f"""
                <div class="panel-title" style="margin-top:16px">
                  <h3>Documents</h3>
                  <span>{len(doc_names)} indexed</span>
                </div>
                """
            )
            doc_list = gr.HTML(doc_list_html(), elem_classes=["doc-list"])
            upload_doc_list = gr.HTML(doc_list_html(), visible=False)
            gr.HTML(
                f"""
                <div class="pipeline" style="margin-top:14px">
                  PDF -> searchable passages -> ranked evidence -> cited answer
                </div>
                """
            )

    send_btn.click(respond, [msg_box, chatbot], [msg_box, chatbot])
    msg_box.submit(respond, [msg_box, chatbot], [msg_box, chatbot])
    clear_btn.click(clear_chat, None, [chatbot, msg_box])
    upload_btn.upload(
        do_upload,
        upload_btn,
        [upload_status, upload_doc_list, doc_list],
    )


if __name__ == "__main__":
    server_port = os.getenv("GRADIO_SERVER_PORT")
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(server_port) if server_port else None,
        share=os.getenv("GRADIO_SHARE", "false").lower() == "true",
        show_error=True,
        css=CSS,
    )
    demo.block_thread()
