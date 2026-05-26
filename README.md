---
title: "RAG Scholar"
emoji: "📚"
colorFrom: "yellow"
colorTo: "gray"
sdk: "gradio"
python_version: "3.11"
app_file: "app.py"
pinned: false
---

# RAG Scholar

RAG Scholar is an NLP-based research assistant for PDF documents. It allows users
to upload one or more PDFs, ask natural-language questions, and receive readable
answers with page-level source citations.

This project was built as a final NLP lab project to demonstrate document
processing, semantic search, lexical search, reranking, prompt engineering, and
retrieval-augmented generation in one working application.

## Project Objective

The goal of this project is to solve a common research problem: long PDF
documents are difficult to search, summarize, and question manually.

RAG Scholar uses NLP techniques to:

- Extract text from uploaded PDF files.
- Split long documents into meaningful chunks.
- Convert text chunks into dense vector embeddings.
- Search documents using both semantic and keyword-based retrieval.
- Rerank retrieved passages for relevance.
- Generate final answers using an LLM.
- Attach page citations so users can verify the answer.

## Key Features

- Multi-PDF upload and indexing.
- Natural-language question answering.
- Hybrid retrieval using FAISS and BM25.
- CrossEncoder reranking for more relevant context.
- Markdown-formatted answers with headings, bullets, tables, and citations.
- Source list with document names and page numbers.
- Clean public-facing Gradio interface.
- Hugging Face Spaces compatible deployment.

## Technology Stack

| Layer | Technology |
|---|---|
| Frontend/UI | Gradio |
| Backend language | Python |
| PDF loading | PyPDFLoader / pypdf |
| Text splitting | LangChain RecursiveCharacterTextSplitter |
| Embeddings | Hugging Face BGE model |
| Vector database | FAISS |
| Keyword retrieval | BM25 |
| Hybrid retrieval | LangChain EnsembleRetriever |
| Reranking | Hugging Face CrossEncoder |
| LLM provider | Groq API |
| LLM model | llama-3.3-70b-versatile |
| Deployment | Hugging Face Spaces |

## System Architecture

```text
User uploads PDFs
        |
        v
PDF text extraction
        |
        v
Text chunking
        |
        v
Embedding generation
        |
        +--------------------+
        |                    |
        v                    v
FAISS semantic index     BM25 keyword index
        |                    |
        +---------+----------+
                  |
                  v
Hybrid retrieval
                  |
                  v
CrossEncoder reranking
                  |
                  v
Top evidence passages
                  |
                  v
Prompt construction
                  |
                  v
Groq LLM answer generation
                  |
                  v
Formatted answer with citations
```

## NLP Implementation

### 1. PDF Text Extraction

The application uses `PyPDFLoader` to read uploaded PDF files page by page.
Each page keeps metadata such as the source file and page number. This metadata
is important because the final answer must cite where the information came from.

### 2. Text Chunking

Large documents cannot be passed directly to an LLM because they exceed context
limits. The text is split into smaller overlapping chunks.

Current chunk settings:

```text
Chunk size: 1200 characters
Chunk overlap: 300 characters
```

Overlap helps preserve context between neighboring chunks so that important
ideas are not cut off at chunk boundaries.

### 3. Embedding Generation

Each chunk is converted into a dense vector embedding using:

```text
BAAI/bge-large-en-v1.5
```

Embeddings represent the semantic meaning of text. This allows the system to
find passages that are conceptually related to the user question, even when the
exact same words are not used.

### 4. Semantic Search with FAISS

The chunk embeddings are stored in a FAISS vector index. FAISS performs fast
similarity search and returns chunks that are semantically close to the query.

This helps answer conceptual questions such as:

```text
What is the main argument of this document?
```

### 5. Keyword Search with BM25

BM25 is used alongside FAISS for lexical retrieval. It is useful when the query
contains exact terms, names, definitions, or technical phrases.

This helps with questions such as:

```text
Find definitions of the key terms.
```

### 6. Hybrid Retrieval

The system combines FAISS and BM25 using weighted hybrid retrieval:

```text
FAISS weight: 0.6
BM25 weight: 0.4
```

This gives the application both semantic understanding and keyword precision.

### 7. CrossEncoder Reranking

The first retrieval stage returns multiple candidate passages. These are then
reranked using:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

The CrossEncoder compares the user question and each retrieved passage together,
then keeps the most relevant passages for answer generation.

Current retrieval settings:

```text
Initial retrieval: top 20 passages
Final reranked context: top 5 passages
```

### 8. Prompt Engineering

The final prompt contains:

- System instructions.
- Conversation history.
- Retrieved document passages.
- The user question.

The assistant is instructed to:

- Answer in clean Markdown.
- Use tables for definitions.
- Use bullets for summaries and comparisons.
- Cite page numbers.
- Admit when the documents do not contain enough evidence.

### 9. Answer Generation

The final answer is generated using the Groq-hosted LLM:

```text
llama-3.3-70b-versatile
```

The model receives only the most relevant retrieved context, which makes the
answer grounded in the uploaded documents instead of relying only on general
model knowledge.

## Application Flow

### Upload Flow

```text
Upload PDF
  -> Extract text by page
  -> Split into chunks
  -> Generate embeddings
  -> Store vectors in FAISS
  -> Store chunks for BM25
  -> Update document list
```

### Question Answering Flow

```text
User question
  -> Search FAISS semantic index
  -> Search BM25 keyword index
  -> Combine retrieval results
  -> Rerank with CrossEncoder
  -> Build prompt with top evidence
  -> Generate answer with Groq LLM
  -> Display answer and sources
```

## Example Questions

Good questions for demonstrating the system:

- Summarize the main argument in five bullet points with evidence.
- Create a glossary of the key terms with short definitions and page citations.
- Which ideas are repeated most often across the document?
- Compare two concepts using cited examples.
- List the practical lessons from the document.
- What are the strongest quotes, and why do they matter?
- Turn this document into study notes.
- What should I ask next to understand this text more deeply?

## Why This Is an NLP Project

This project uses several core NLP concepts:

- Document parsing and preprocessing.
- Text segmentation.
- Vector representations of language.
- Semantic similarity search.
- Lexical information retrieval.
- Query-document relevance scoring.
- Reranking.
- Prompt engineering.
- Natural-language generation.
- Citation-grounded answer synthesis.

It is not just a chatbot. It is a retrieval-augmented NLP pipeline that connects
document understanding, information retrieval, and language generation.

## Local Setup

Install dependencies:

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

## Hugging Face Spaces Deployment

This repository is ready for Hugging Face Spaces.

Required Space secret:

```text
Name: GROQ_API_KEY
Value: your_groq_api_key
```

Do not write `GROQ_API_KEY=` in the value field. Paste only the key.

## Team Members

- Muhammad Ali Abid - FA23-BBD-085
- Ibrahim Zaheer - FA23-BBD-058
- Rana Usman Fakhar - SP23-BBD-049

## Important Files

| File | Purpose |
|---|---|
| `app.py` | Main Gradio application and RAG pipeline |
| `requirements.txt` | Python dependencies |
| `README.md` | Project documentation and Spaces metadata |
| `rag_scholar.ipynb` | Notebook walkthrough of the RAG workflow |
| `.env.example` | Example environment variable file |

## Files Not Pushed to GitHub

The following are intentionally ignored:

```text
.env
rag_index/
rag_bm25.pkl
__pycache__/
.gradio/
*.log
```

These files contain secrets, local indexes, generated caches, or runtime output.

## Limitations

- The first run may take time because embedding and reranking models must be
  downloaded.
- The answer quality depends on the quality of retrieved passages.
- Very large PDFs may require more memory and indexing time.
- The system answers from uploaded documents; if evidence is missing, it should
  say so.

## Future Improvements

- Add document-level filtering.
- Add citation previews.
- Add support for DOCX and web pages.
- Add evaluation metrics for retrieval quality.
- Add user-selectable answer formats.
- Add persistent cloud storage for indexes.

## Conclusion

RAG Scholar demonstrates how modern NLP systems combine retrieval and generation.
It processes real PDF documents, retrieves relevant evidence, reranks context,
and generates cited answers through a clean user interface. This makes it a
complete final NLP lab project covering both theory and practical deployment.
