# 🏥 MedCortex — AI-Powered Clinical Knowledge Assistant

<div align="center">

![MedCortex Banner](https://img.shields.io/badge/MedCortex-AI%20Health%20Assistant-6f4ef2?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0xMiAyQzYuNDggMiAyIDYuNDggMiAxMnM0LjQ4IDEwIDEwIDEwIDEwLTQuNDggMTAtMTBTMTcuNTIgMiAxMiAyek0xMyAxN0gxMXYtNkg3bDUtNSA1IDVoLTR2NnoiLz48L3N2Zz4=)

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Next.js](https://img.shields.io/badge/Next.js-14-black?style=flat-square&logo=next.js)](https://nextjs.org)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0-3178C6?style=flat-square&logo=typescript&logoColor=white)](https://typescriptlang.org)
[![LangChain](https://img.shields.io/badge/LangChain-Latest-1C3C3C?style=flat-square&logo=langchain)](https://langchain.com)
[![Pinecone](https://img.shields.io/badge/Pinecone-Vector%20DB-00B6A1?style=flat-square)](https://pinecone.io)
[![Groq](https://img.shields.io/badge/Groq-Llama%203.3%2070B-F55036?style=flat-square)](https://groq.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

**A full-stack RAG-powered medical knowledge system grounded in NIH MedlinePlus data and curated clinical textbooks. Answers clinical questions with cited, verifiable sources — never hallucinates.**

[Features](#-features) · [Architecture](#-system-architecture) · [Data Pipeline](#-data-pipeline) · [Frontend](#-frontend) · [Getting Started](#-getting-started) · [Demo](#-demo)

</div>

---

## 📌 Overview

MedCortex is an end-to-end **Retrieval-Augmented Generation (RAG)** medical assistant that combines a structured NIH MedlinePlus knowledge base with unstructured clinical textbooks, all stored in a Pinecone vector database and served through a **Llama 3.3 70B** language model via Groq. The frontend is a modern **Next.js 14** application with a polished auth flow.

> ⚠️ **Disclaimer:** MedCortex is intended for educational and research purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 **Dual Knowledge Base** | NIH MedlinePlus API (4,822 chunks) + 4 curated medical textbooks (9,513 vectors) |
| 🧠 **High-Dimensional Embeddings** | `BAAI/bge-large-en-v1.5` producing 1024-dimension vectors for precise semantic search |
| ⚡ **Blazing Fast Inference** | Llama 3.3 70B served via Groq API — responses in ~2–4 seconds |
| 🎯 **Cited Answers** | Every response cites the exact book and section the information came from |
| 🚫 **No Hallucinations** | Strictly grounded — refuses to answer if the context doesn't contain the answer |
| 🔐 **Full Auth System** | JWT-based login/signup with optional Google OAuth |
| 🎨 **Beautiful UI** | Next.js 14 + Tailwind CSS with animated wave layouts and floating inputs |

---

## 🏛️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        MedCortex System                         │
├──────────────────────┬──────────────────────────────────────────┤
│   DATA PIPELINE       │           INFERENCE PIPELINE            │
│                       │                                          │
│  NIH MedlinePlus API  │   User Query                            │
│         ↓             │       ↓                                  │
│  Alphabet Sweep (A-Z) │   BGE-Large Embedder (1024d)            │
│     300 topics/letter │       ↓                                  │
│         ↓             │   Pinecone MMR Search (k=4, fetch=10)   │
│  Sentence Chunker     │       ↓                                  │
│  (512 chars, 80 OL)   │   format_docs() → Source + Heading      │
│         ↓             │       ↓                                  │
│  Dedup + Clean        │   Llama 3.3 70B (Groq API)              │
│  4,822 unique chunks  │       ↓                                  │
│         ↓             │   Structured Answer + Citations         │
│  4 Medical Textbooks  │                                          │
│  (9,513 vectors)      │                                          │
│         ↓             │                                          │
│  Pinecone Index       │                                          │
│  (1024d, cosine)      │                                          │
└──────────────────────┴──────────────────────────────────────────┘
```

---

## 📚 Data Pipeline

### Source 1 — NIH MedlinePlus (Structured)

The structured pipeline fetches every health topic from the [NIH MedlinePlus API](https://wsearch.nlm.nih.gov/ws/query) using a full **alphabet sweep** (A–Z, 300 results per letter).

```python
# Alphabet sweep → 26 queries × 300 topics each
QUERIES = [(letter, 300) for letter in string.ascii_lowercase]
```

**Final dataset stats after cleaning:**

| Metric | Value |
|---|---|
| Raw chunks fetched | 12,016 |
| After deduplication | 4,843 |
| After text-dedup | **4,822** |
| Unique health topics | **772** |
| Columns retained | 15 |
| File size | 37.45 MB |

**Schema overview:**

```
Vector Core   → chunk_id, doc_id, text, embedding_text
Mechanics     → chunk_index, chunk_total
Metadata      → topic_id, title, synonyms, mesh_terms, disease_category
Display       → full_summary, snippet, source_url
Auditing      → data_source, search_term
```

**Chunking strategy:**
- Sentence-aware splitting at `.`, `!`, `?` boundaries
- Chunk size: **512 characters** with **80-character overlap**
- Each chunk inherits full document metadata for filtered retrieval

**Cleaning steps:**
1. Drop columns with 100% missing values (`research_institute`, `date_created`, `date_updated`, `language`, `see_also`)
2. Convert fake empty strings (`""`, `"—"`) to `NaN`
3. Impute remaining nulls: `synonyms` (784 rows), `mesh_terms` (12 rows) → `"Unknown"`
4. Deduplicate on `chunk_id` then on `text`

---

### Source 2 — Medical Textbooks (Unstructured)

Four clinical textbooks processed with **Docling** into LangChain `Document` objects with rich metadata:

| Book | Chunks |
|---|---|
| Gray's Anatomy for Students (4th Ed.) | 2,676 |
| Mosby's Diagnostic & Laboratory Test Reference (15th Ed.) | 3,301 |
| Learning Radiology: Recognizing the Basics (3rd Ed.) | 1,310 |
| Symptoms to Diagnosis | 2,226 |
| **Total** | **9,513** |

Each chunk carries: `book_title`, `source_file`, `docling_headings`, `page_content`.

---

### Pinecone Vector Store

```python
INDEX_NAME = "medical-assistant"
DIMENSION  = 1024          # BAAI/bge-large-en-v1.5
METRIC     = "cosine"
NAMESPACE  = "medical_textbooks_base"
CLOUD      = "aws"
REGION     = "us-east-1"
```

Embeddings are computed on GPU in batches of 100 using `langchain-huggingface` + `langchain-pinecone`.

---

## 🤖 RAG Inference Pipeline

```python
# Retrieval
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 10}
)

# Chain (LangChain Expression Language)
rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0)
    | StrOutputParser()
)
```

**System prompt key rules:**
1. Answer **only** from retrieved context
2. If not found → explicitly state it (no hallucination)
3. Structure answers with bullet points for readability
4. Cite source book and section heading at the end

---

## 🎨 Frontend

Built with **Next.js 14 (App Router)** + **Tailwind CSS**, featuring a distinct visual identity in purple/violet (`#6f4ef2`).

### Pages & Components

```
app/
├── login/page.tsx          → AuthWaveLayout + LoginForm
├── signup/page.tsx         → AuthWaveLayout + SignupForm
└── chat/                   → Main assistant interface

components/auth/
├── AuthWaveLayout.tsx      → Animated SVG wave card layout
├── AuthSplitLayout.tsx     → Hero image split layout (alternative)
├── LoginForm.tsx           → Email/password with floating pill inputs
├── SignupForm.tsx          → Full profile creation form
├── FloatingField.tsx       → Reusable floating label inputs
└── GoogleSignInButton.tsx  → Google OAuth (One Tap)
```

### Design Highlights

- **AuthWaveLayout** — SVG blob/wave decorations with depth layers, glassmorphism card
- **Pill inputs** — gradient icon badges, floating labels, purple focus rings
- **Gradient CTAs** — `from-[#8566FF] to-[#6f4ef2]` with lift-on-hover shadow
- **Fully responsive** — single column on mobile, dual column on desktop

### Auth Flow

```
POST /auth/login  → { access_token, user } → persistSession() → /chat
POST /auth/signup → { access_token, user } → persistSession() → /chat
```

Supports both **email/password** and **Google One Tap** (`NEXT_PUBLIC_GOOGLE_CLIENT_ID`).

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- GPU (recommended for embedding generation)
- Pinecone account (free tier works)
- Groq API key (free at [console.groq.com](https://console.groq.com))

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/medcortex.git
cd medcortex
```

### 2. Backend — Data Pipeline (Google Colab recommended)

```bash
pip install langchain langchain-core langchain-community langchain-groq \
            langchain-pinecone langchain-huggingface pinecone-client \
            sentence-transformers gradio
```

Set your API keys:
```python
import os
os.environ["PINECONE_API_KEY"] = "your-pinecone-key"
os.environ["GROQ_API_KEY"]     = "your-groq-key"
```

Run the notebooks in order:
1. `Copy_of_medical_assis_structure_API.ipynb` — Fetch & clean MedlinePlus data
2. `Copy_of_medical_assistant_rag.ipynb` — Embed textbooks → Pinecone → RAG demo

### 3. Frontend

```bash
cd frontend
npm install
```

Create `.env.local`:
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_GOOGLE_CLIENT_ID=your-google-client-id   # optional
```

```bash
npm run dev
# → http://localhost:3000
```

---

## 🖼️ Demo

### Clinical Question Examples

**Q: What are the common causes and treatments for hypernatremia?**

> Sources cited: *Mosby's Diagnostic & Lab Reference* · *Symptoms to Diagnosis*

**Q: What are the classic symptoms of acute appendicitis, and what tests diagnose it?**

> Sources cited: *Gray's Anatomy for Students* · *Symptoms to Diagnosis*

**Response time:** ~2–4 seconds end-to-end via Groq's ultra-low-latency inference.

---

## 🗂️ Project Structure

```
medcortex/
├── notebooks/
│   ├── Copy_of_medical_assis_structure_API.ipynb    # MedlinePlus ETL pipeline
│   └── Copy_of_medical_assistant_rag.ipynb          # RAG system + Gradio demo
│
├── data/
│   └── MedlinePlus_Structured_API_Medical_Knowledge.csv   # 4,822 × 15 cleaned dataset
│
├── frontend/
│   ├── app/
│   │   ├── login/page.tsx
│   │   ├── signup/page.tsx
│   │   └── chat/
│   ├── components/
│   │   └── auth/
│   │       ├── AuthWaveLayout.tsx
│   │       ├── AuthSplitLayout.tsx
│   │       ├── LoginForm.tsx
│   │       ├── SignupForm.tsx
│   │       ├── FloatingField.tsx
│   │       └── GoogleSignInButton.tsx
│   └── ...
│
└── README.md
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | Llama 3.3 70B via Groq API |
| **Embeddings** | BAAI/bge-large-en-v1.5 (1024d) |
| **Vector DB** | Pinecone (Serverless, AWS us-east-1) |
| **RAG Framework** | LangChain (LCEL) |
| **Data Source 1** | NIH MedlinePlus Web Services API |
| **Data Source 2** | Docling PDF parser (clinical textbooks) |
| **Frontend** | Next.js 14, TypeScript, Tailwind CSS |
| **Auth** | JWT + Google One Tap OAuth |
| **HTTP Client** | Axios |
| **Notebook Env** | Google Colab (T4 GPU) |

---

## 📊 Performance

| Metric | Value |
|---|---|
| Total vectors in Pinecone | 9,513 |
| Embedding dimensions | 1,024 |
| Retrieval strategy | MMR (k=4, fetch_k=10) |
| Average response time | ~2–4 seconds |
| Knowledge base topics | 772 unique NIH topics |
| Medical textbooks indexed | 4 |

---

## 🔮 Roadmap

- [ ] Add structured NIH MedlinePlus chunks to Pinecone (separate namespace)
- [ ] Implement hybrid search (dense + sparse BM25)
- [ ] Add chat history with multi-turn conversation memory
- [ ] Drug interaction checker module
- [ ] Symptom checker with differential diagnosis scoring
- [ ] Mobile app (React Native)
- [ ] Fine-tuned medical embedding model

---

## 🤝 Contributing

Contributions are welcome! Please open an issue first to discuss what you'd like to change.

```bash
# Fork → clone → create a branch
git checkout -b feature/your-feature-name

# Make changes, then
git commit -m "feat: add your feature"
git push origin feature/your-feature-name
# Open a Pull Request
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- [NIH MedlinePlus](https://medlineplus.gov/) for the open health topics API
- [BAAI](https://huggingface.co/BAAI/bge-large-en-v1.5) for the BGE embedding model
- [Groq](https://groq.com/) for ultra-fast Llama 3 inference
- [Pinecone](https://pinecone.io/) for the serverless vector database
- [LangChain](https://langchain.com/) for the RAG orchestration framework

---

<div align="center">
  <sub>Built with ❤️ for educational and research purposes · Not a substitute for professional medical advice</sub>
</div>
