# SatGraffin v2.0 — AI Research Assistant

A general-purpose AI research assistant powered by **Hybrid RAG** (BM25 + FAISS + CrossEncoder) and **Gemini 2.5 Flash**. It searches the web in real-time, scrapes sources, and generates grounded answers with citations.

---

## ⚡ Quick Start (any machine)

### Prerequisites
- Python 3.11+
- Node.js 18+
- A Google Gemini API key → [Get one here](https://aistudio.google.com/app/apikey)

---

### Backend

```bash
cd backend

# 1. Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
copy .env.example .env        # Windows
# cp .env.example .env        # Linux/Mac
# Then open .env and fill in your GOOGLE_API_KEY

# 4. Run the server
uvicorn main:app --reload --port 8000
```

> **First run note:** On first startup, the server will download:
> - `all-MiniLM-L6-v2` embedding model (~90 MB)
> - `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker (~80 MB)
> - NLTK `punkt_tab` tokenizer data
>
> Subsequent starts are instant.

---

### Frontend

```bash
cd frontend

# Install dependencies
npm install

# (Optional) Configure backend URL if not localhost
copy .env.example .env.local   # Windows
# cp .env.example .env.local   # Linux/Mac
# Edit VITE_API_BASE_URL if your backend is on a remote server

# Start dev server
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) in your browser.

---

## 🔑 Environment Variables

### Backend (`backend/.env`)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GOOGLE_API_KEY` | ✅ Yes | — | Gemini API key |
| `GEMINI_MODEL_NAME` | No | `gemini-2.5-flash` | Gemini model to use |
| `CACHE_DURATION_HOURS` | No | `6` | How long to cache scraped pages |
| `MAX_SEARCH_RESULTS` | No | `5` | Max DuckDuckGo results to fetch |
| `MAX_SCRAPE_PAGES` | No | `3` | Max pages to scrape per query |

### Frontend (`frontend/.env.local`)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `VITE_API_BASE_URL` | No | `http://localhost:8000` | Backend API URL |

---

## 🏗️ Architecture

```
User Query
    ↓
[QueryAnalyzer (Gemini)]  ← rewrites query for search engine
    ↓
[DuckDuckGo Search]       ← finds top 5 URLs
    ↓
[Parallel Scraper]        ← scrapes top 3 URLs simultaneously
    ↓
[Knowledge Base Update]   ← chunks + embeds + stores in FAISS + BM25
    ↓
[Hybrid RAG Pipeline]
  BM25 (top-20) + FAISS dense (top-20)
       ↓ RRF Fusion → top-20
       ↓ CrossEncoder rerank → top-5
       ↓ Context packing (3000 token budget)
    ↓
[Gemini 2.5 Flash]        ← generates grounded answer
    ↓
Response with citations
```

---

## 📁 Project Structure

```
SatGraffin/
├── backend/
│   ├── main.py               # FastAPI app + all pipeline logic
│   ├── hybrid_rag/           # Modular RAG components
│   │   ├── chunker.py        # Sentence-aware chunking
│   │   ├── bm25_retriever.py # BM25 keyword search
│   │   ├── rrf_fusion.py     # Reciprocal Rank Fusion
│   │   ├── cross_encoder_reranker.py
│   │   ├── context_packer.py
│   │   ├── prompt_builder.py
│   │   ├── query_analyzer.py
│   │   └── feedback_loop.py
│   ├── requirements.txt      # pip-installable dependencies
│   ├── .env.example          # Environment template
│   └── ARCHITECTURE.md       # Detailed architecture docs
└── frontend/
    ├── src/
    │   ├── App.tsx            # Main React component
    │   ├── components/        # UI components
    │   └── hooks/
    ├── package.json
    └── .env.example           # Frontend env template
```

---

## 🐛 Known Limitations

- **Conversation memory is per-process** — if you run multiple uvicorn workers, memory is not shared across workers
- **Models download on first use** — requires internet access on first startup
- **torch is CPU-only by default** — for GPU support, install a CUDA-enabled torch build from [pytorch.org](https://pytorch.org)