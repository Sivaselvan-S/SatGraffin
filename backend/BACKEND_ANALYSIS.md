# SatGraffin Backend - Comprehensive Flow Analysis

**Date:** October 3, 2025  
**Status:** Production-Ready with Improvements Needed

---

## 📊 Current Architecture Overview

### **Flow Diagram**
```
User Query → FastAPI Endpoint
    ↓
Query Classification (Intent Detection)
    ↓
Link Extraction (Fuzzy + Intent-Based Matching)
    ↓
Cache Check (Smart Caching - 24h window)
    ↓
┌─────────────────────────────────────┐
│ If Missing/Stale: Synchronous Scrape│
│ If Fresh: Use Cached Data           │
└─────────────────────────────────────┘
    ↓
Vector Store Retrieval (FAISS + MMR)
    ↓
LLM Generation (Google Gemini)
    ↓
Response with Sources
```

---

## ✅ Strengths

### 1. **Smart Caching System**
- ✅ Avoids redundant scraping within 24-hour window
- ✅ File modification time fallback
- ✅ In-memory timestamp tracking
- ✅ Configurable via environment variable

### 2. **Intent Classification**
- ✅ Detects satellite-specific queries
- ✅ Identifies comparative questions
- ✅ Recognizes technical/API queries
- ✅ Temporal query detection

### 3. **Dynamic Page Fetching**
- ✅ On-demand scraping with synchronous wait
- ✅ Link index built from homepage + sitemap
- ✅ Fuzzy matching for URL discovery
- ✅ Numeric pattern matching (e.g., "oceansat 3" → "oceansat-3")

### 4. **Robust RAG Chain**
- ✅ Map-reduce chain for long documents
- ✅ MMR retriever for diversity
- ✅ Proper error handling
- ✅ Source document tracking

---

## 🚨 Critical Issues

### **Issue #1: Vector Store Duplication**
**Severity:** HIGH  
**Location:** `update_page_data()`

**Problem:**
```python
db.add_documents(texts)  # Adds chunks WITHOUT checking if they already exist
db.save_local(DB_FAISS_PATH)
```

Every time a page is refreshed, chunks are **ADDED** instead of **REPLACED**, causing:
- Growing vector store size (memory bloat)
- Duplicate search results
- Slower retrieval over time

**Solution:**
```python
# Remove old chunks for this source before adding new ones
# OR use a document ID system to update instead of append
```

---

### **Issue #2: Global State Mutation**
**Severity:** MEDIUM  
**Location:** `update_page_data()`, startup

**Problem:**
```python
global qa_chain  # Modified in update_page_data()
qa_chain = RetrievalQA.from_chain_type(...)  # Race condition risk
```

Concurrent requests could access `qa_chain` while it's being reloaded, causing:
- Potential `None` reference errors
- Inconsistent retrieval results
- Thread safety issues

**Solution:**
```python
# Use threading locks or reload chain atomically
# Consider lazy loading per request instead of global mutation
```

---

### **Issue #3: No Embedding Model Caching**
**Severity:** MEDIUM  
**Location:** `update_page_data()`

**Problem:**
```python
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, ...)  # Reloaded every time!
```

The embedding model is reloaded from disk on **every page update**, taking 2-5 seconds unnecessarily.

**Solution:**
```python
# Load embeddings once at startup and reuse
global embeddings_model  # Initialized in startup, reused in updates
```

---

### **Issue #4: Synchronous Blocking on Stale Pages**
**Severity:** LOW-MEDIUM  
**Location:** `process_query()`

**Problem:**
```python
if should_refresh:
    update_page_data(page_url)  # Blocks user even for stale (not missing) pages
```

When a page is **stale but exists**, the user waits 10-15 seconds for a refresh even though cached data could answer their question.

**Solution:**
```python
# For stale pages: return cached answer + refresh in background
# Only block synchronously for missing pages
```

---

### **Issue #5: Missing Error Handling in Scraper**
**Severity:** MEDIUM  
**Location:** `scraper.py`

**Problem:**
```python
def scrape_and_update(url: str) -> str | None:
    text, _ = fetch_page(url)  # Returns None on error
    # No logging, no retry, no fallback
```

Failed scrapes are silent, causing:
- Empty responses when pages fail to load
- No visibility into scraping errors
- No retry mechanism for transient failures

**Solution:**
```python
# Add exponential backoff retry logic
# Log scraping failures with details
# Return error info to caller
```

---

### **Issue #6: Link Index Not Persisted**
**Severity:** LOW  
**Location:** Startup

**Problem:**
```python
link_index: dict[str, str] = {}  # Rebuilt on every server restart
```

Homepage + sitemap are re-fetched on every restart (adds 5-10 seconds to startup).

**Solution:**
```python
# Cache link_index to disk (JSON file)
# Only rebuild if older than X days
```

---

## 🔧 Performance Bottlenecks

### **Bottleneck #1: Synchronous Embedding**
**Impact:** 5-10 seconds per page update

```python
texts = splitter.split_documents(docs)  # Fast
embeddings = HuggingFaceEmbeddings(...)  # 2-5 seconds (reloaded!)
db.add_documents(texts)  # 3-5 seconds (embedding + indexing)
```

**Fix:** Reuse embedding model, consider batch processing

---

### **Bottleneck #2: FAISS Save/Load**
**Impact:** 1-3 seconds per update

```python
db.save_local(DB_FAISS_PATH)  # Writes entire index to disk
```

For frequent updates, saving the entire index is wasteful.

**Fix:** Use in-memory index, periodic batch saves

---

### **Bottleneck #3: Chain Reload**
**Impact:** 1-2 seconds per update

```python
qa_chain = RetrievalQA.from_chain_type(...)  # Rebuilds entire chain
```

**Fix:** Update retriever only, not entire chain

---

## 🎯 Recommended Improvements

### **Priority 1: Fix Vector Store Duplication**
```python
def update_page_data(url: str):
    # ... scraping code ...
    
    # Remove old documents for this source
    source_filter = {"source": page_path}
    # Note: FAISS doesn't support deletion by metadata easily
    # Solution: Rebuild index OR use document IDs
    
    # Better approach:
    all_docs = []  # Load all existing docs
    all_docs = [d for d in all_docs if d.metadata['source'] != page_path]  # Remove old
    all_docs.extend(texts)  # Add new
    
    # Rebuild index
    db = FAISS.from_documents(all_docs, embeddings)
    db.save_local(DB_FAISS_PATH)
```

---

### **Priority 2: Cache Embedding Model**
```python
# At startup
global embeddings_model
embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, ...)

# In update_page_data()
db = FAISS.load_local(DB_FAISS_PATH, embeddings_model, ...)  # Reuse!
```

---

### **Priority 3: Smarter Stale Page Handling**
```python
if should_refresh:
    if not os.path.exists(page_path):
        # Missing: Block and fetch
        update_page_data(page_url)
    else:
        # Stale: Return cached answer, refresh in background
        background_tasks.add_task(update_page_data, page_url)
        logger.info(f"Returning cached data, refreshing {page_filename} in background")
```

---

### **Priority 4: Add Retry Logic to Scraper**
```python
import time
from functools import wraps

def retry_on_failure(max_retries=3, delay=2):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Failed after {max_retries} attempts: {e}")
                        raise
                    time.sleep(delay * (2 ** attempt))  # Exponential backoff
            return None
        return wrapper
    return decorator

@retry_on_failure(max_retries=3)
def fetch_page(url: str) -> tuple[str | None, set[str]]:
    # ... existing code ...
```

---

### **Priority 5: Thread-Safe Chain Updates**
```python
import threading

chain_lock = threading.Lock()

def update_page_data(url: str):
    # ... scraping and embedding ...
    
    global qa_chain
    with chain_lock:
        # Atomic update
        new_retriever = db.as_retriever(...)
        new_chain = RetrievalQA.from_chain_type(llm, retriever=new_retriever, ...)
        qa_chain = new_chain  # Single assignment
```

---

## 🔒 Security Concerns

### **Concern #1: CORS Wide Open**
```python
allow_origins=["*"]  # Allows any origin!
```
**Risk:** CSRF attacks, unauthorized API access  
**Fix:** Whitelist specific frontend domains in production

---

### **Concern #2: No Rate Limiting**
**Risk:** DoS attacks, resource exhaustion  
**Fix:** Implement rate limiting middleware

```python
from slowapi import Limiter
limiter = Limiter(key_func=lambda: request.client.host)

@app.post("/api/query")
@limiter.limit("10/minute")  # Max 10 queries per minute
def process_query(...):
```

---

### **Concern #3: Unsanitized URL Input**
```python
page_url = extract_relevant_link(request.query)
update_page_data(page_url)  # Could scrape arbitrary URLs!
```
**Risk:** SSRF attacks, scraping internal resources  
**Fix:** Validate URLs against whitelist

---

## 📈 Scalability Issues

### **Issue #1: In-Memory Vector Store**
- **Problem:** FAISS index grows indefinitely in memory
- **Limit:** ~10GB RAM for 100k documents
- **Fix:** Use persistent vector DB (Pinecone, Weaviate, Qdrant)

### **Issue #2: Single-Process Bottleneck**
- **Problem:** Synchronous scraping blocks entire server
- **Fix:** Use worker queue (Celery, RQ) for background tasks

### **Issue #3: No Horizontal Scaling**
- **Problem:** Global `qa_chain` prevents multi-instance deployment
- **Fix:** Move state to Redis/database, use stateless endpoints

---

## 🧪 Testing Gaps

### Missing Tests:
1. ❌ Unit tests for intent classification
2. ❌ Integration tests for scraping pipeline
3. ❌ Load testing for concurrent queries
4. ❌ Cache invalidation tests
5. ❌ Error recovery tests

---

## 🎯 Quick Wins (Implement First)

1. **Cache Embedding Model** (5-10 sec savings per update)
2. **Fix Stale Page Blocking** (Better UX)
3. **Add Scraper Retry Logic** (Improved reliability)
4. **Prevent Vector Store Duplication** (Fixes growing memory)
5. **Add Basic Rate Limiting** (Security)

---

## 📊 Metrics to Track

### Add these monitoring points:
```python
# In process_query()
start_time = time.time()
# ... processing ...
duration = time.time() - start_time
logger.info(f"Query processed in {duration:.2f}s")

# Track:
# - Average query time
# - Cache hit rate
# - Scraping success rate
# - Vector store size
# - Error rates
```

---

## 🚀 Next Steps

### Week 1: Critical Fixes
- [ ] Fix vector store duplication
- [ ] Cache embedding model globally
- [ ] Add thread locks for qa_chain updates

### Week 2: Performance
- [ ] Improve stale page handling
- [ ] Add retry logic to scraper
- [ ] Optimize FAISS save/load

### Week 3: Security & Monitoring
- [ ] Tighten CORS policy
- [ ] Add rate limiting
- [ ] Implement metrics tracking

---

## 📝 Code Quality Score: 7.5/10

**Strengths:**
- Good intent classification
- Smart caching implementation
- Proper error handling in query endpoint

**Weaknesses:**
- Vector store duplication bug
- Global state management
- Missing retry logic
- No comprehensive tests

---

## 📞 Contact Points for Help

- **LangChain FAISS Docs:** https://python.langchain.com/docs/integrations/vectorstores/faiss
- **FastAPI Background Tasks:** https://fastapi.tiangolo.com/tutorial/background-tasks/
- **Gemini API:** https://ai.google.dev/gemini-api/docs

