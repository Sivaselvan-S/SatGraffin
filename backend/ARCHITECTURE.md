# SatGraffin v2.0 - Architecture Documentation

**Date:** January 27, 2026  
**Status:** Production-Ready

---

## 🚀 What's New in v2.0

SatGraffin has been transformed from a single-website MOSDAC-focused bot into a **general-purpose AI research assistant** that can answer any question by:

1. **Searching the web** in real-time
2. **Scraping relevant sources** dynamically
3. **Building a knowledge base** on-the-fly
4. **Generating accurate, sourced answers** using RAG

---

## 📊 New Architecture Overview

### **Flow Diagram**
```
User Query
    ↓
┌─────────────────────────────────────┐
│ 1. Web Search (DuckDuckGo)          │
│    - Searches for relevant content  │
│    - Returns top 5 results          │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 2. Dynamic Scraping                 │
│    - Scrapes top 3 URLs             │
│    - Extracts clean text content    │
│    - Intelligent caching (6 hours)  │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 3. Knowledge Base Update            │
│    - Chunks content (500 tokens)    │
│    - Embeds with sentence-transformers│
│    - Stores in FAISS vector store   │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 4. RAG Query                        │
│    - Retrieves relevant chunks (k=5)│
│    - Uses MMR for diversity         │
│    - Generates answer with Gemini   │
└─────────────────────────────────────┘
    ↓
Response with Sources
```

---

## 🔧 Core Components

### 1. **WebSearcher**
- Uses DuckDuckGo's HTML interface (no API key needed)
- Parses search results for title, URL, and snippet
- Handles redirect URLs properly

### 2. **DynamicScraper**
- Scrapes any website
- Extracts clean text (removes scripts, nav, footer, etc.)
- Intelligent caching with configurable duration
- Handles binary file detection (skips PDFs, images, etc.)

### 3. **KnowledgeBase**
- Manages FAISS vector store
- Uses HuggingFace embeddings (all-MiniLM-L6-v2)
- Dynamically adds new content
- Rebuilds QA chain after updates

### 4. **RAG Pipeline**
- LangChain RetrievalQA with "stuff" chain type
- MMR retrieval for diverse results
- Gemini 2.5 Flash for generation
- Low temperature (0.2) for factual responses

---

## ⚙️ Configuration

Environment variables in `.env`:

```env
GOOGLE_API_KEY="your-api-key"
GEMINI_MODEL_NAME="gemini-2.5-flash"
CACHE_DURATION_HOURS=6
MAX_SEARCH_RESULTS=5
MAX_SCRAPE_PAGES=3
```

---

## 📡 API Endpoints

### `GET /`
Health check endpoint.

### `GET /api/health`
Detailed health check with knowledge base status.

### `POST /api/query`
Main query endpoint.

**Request:**
```json
{
  "query": "What is quantum computing?",
  "user_id": "optional-user-id",
  "force_refresh": false
}
```

**Response:**
```json
{
  "response": "Quantum computing is...",
  "answer": "Quantum computing is...",
  "source_links": ["https://..."],
  "source_documents": [...],
  "search_query": "quantum computing"
}
```

### `POST /api/clear-cache`
Clears all cached scraped content.

---

## ✅ Key Improvements

1. **No More Single-Site Limitation**: Can answer questions about anything
2. **Real-Time Information**: Searches and scrapes the web for current data
3. **No Hallucination**: All answers are grounded in scraped sources
4. **Source Transparency**: Every answer includes source links
5. **Smart Caching**: Reduces redundant scraping
6. **Cleaner Architecture**: Modular, class-based design

---

## 🏃 Running the Application

### Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

---

## 🔮 Future Enhancements

- [ ] Add PDF parsing support
- [ ] Implement query refinement with LLM
- [ ] Add conversation memory
- [ ] Support for specific site searches (e.g., "search Wikipedia for...")
- [ ] Add rate limiting for scraping
- [ ] Implement parallel scraping for faster responses
