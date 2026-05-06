"""
SatGraffin - General Purpose AI Research Assistant
===================================================
A RAG-based assistant that searches the web, scrapes relevant content,
and provides accurate answers grounded in real sources.
"""

import logging
import os
import json
import re
import hashlib
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from functools import lru_cache
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, urljoin
import urllib.parse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# LangChain imports
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import RetrievalQA
from langchain_core.documents import Document
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate

# Semantic chunking (optional - falls back to regular chunking if not available)
try:
    from langchain_experimental.text_splitter import SemanticChunker
    SEMANTIC_CHUNKING_AVAILABLE = True
except ImportError:
    SEMANTIC_CHUNKING_AVAILABLE = False

# Hybrid RAG components
from hybrid_rag.chunker import SentenceAwareChunker
from hybrid_rag.bm25_retriever import BM25Retriever
from hybrid_rag.rrf_fusion import reciprocal_rank_fusion
from hybrid_rag.cross_encoder_reranker import CrossEncoderReranker
from hybrid_rag.context_packer import ContextPacker
from hybrid_rag.prompt_builder import PromptBuilder

# Load environment variables
load_dotenv()

# --- Configuration ---
DATA_DIR = Path("data")
PROCESSED_DIR = Path("processed_data")
VECTOR_STORE_PATH = "vector_store"
BM25_INDEX_PATH = os.path.join(VECTOR_STORE_PATH, "bm25_index.pkl")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")
CACHE_DURATION_HOURS = int(os.getenv("CACHE_DURATION_HOURS", "6"))
MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "5"))
MAX_SCRAPE_PAGES = int(os.getenv("MAX_SCRAPE_PAGES", "3"))

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("satgraffin")

# --- Ensure directories exist ---
DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# --- Source Quality Scoring ---
# Domains are scored for trustworthiness (higher = more trusted)
TRUSTED_DOMAINS = {
    # Educational & Government (highest trust)
    ".edu": 10, ".gov": 10, ".ac.": 9,
    # Major encyclopedias & reference
    "wikipedia.org": 9, "britannica.com": 9, "scholarpedia.org": 8,
    # Academic & Research
    "arxiv.org": 9, "scholar.google": 8, "researchgate.net": 7, "academia.edu": 7,
    # Official documentation
    "docs.python.org": 8, "developer.mozilla.org": 8, "docs.microsoft.com": 8,
    # Reputable tech sources
    "stackoverflow.com": 7, "github.com": 6, "medium.com": 5,
    # News (moderate trust)
    "reuters.com": 7, "bbc.com": 7, "nytimes.com": 6,
}

def score_source(url: str) -> int:
    """Score a URL based on domain trustworthiness."""
    url_lower = url.lower()
    for domain, score in TRUSTED_DOMAINS.items():
        if domain in url_lower:
            return score
    return 3  # Default score for unknown sources


# --- Conversation Memory Store ---
# Simple in-memory store for conversation history per user
conversation_memory: dict[str, list[dict]] = {}
MAX_MEMORY_TURNS = 5  # Keep last 5 conversation turns per user

# --- Context Preference Store ---
# Stores user's selected disambiguation context (e.g., "Distributed Computing")
context_preferences: dict[str, str] = {}

def get_conversation_context(user_id: str) -> str:
    """Get formatted conversation history for context."""
    if not user_id or user_id not in conversation_memory:
        return ""
    
    history = conversation_memory[user_id]
    if not history:
        return ""
    
    context_parts = ["Previous conversation:"]
    for turn in history[-MAX_MEMORY_TURNS:]:
        context_parts.append(f"User: {turn['query']}")
        context_parts.append(f"Assistant: {turn['response'][:200]}...")
    
    return "\n".join(context_parts)

def add_to_memory(user_id: str, query: str, response: str):
    """Add a conversation turn to memory."""
    if not user_id:
        return
    
    if user_id not in conversation_memory:
        conversation_memory[user_id] = []
    
    conversation_memory[user_id].append({
        "query": query,
        "response": response,
        "timestamp": datetime.now().isoformat()
    })
    
    # Keep only last MAX_MEMORY_TURNS
    if len(conversation_memory[user_id]) > MAX_MEMORY_TURNS:
        conversation_memory[user_id] = conversation_memory[user_id][-MAX_MEMORY_TURNS:]


def get_context_preference(user_id: str) -> Optional[str]:
    """Get user's selected context preference."""
    if not user_id:
        return None
    return context_preferences.get(user_id)


def set_context_preference(user_id: str, context: str):
    """Set user's context preference for disambiguation."""
    if not user_id:
        return
    context_preferences[user_id] = context
    logger.info(f"Set context preference for user {user_id}: {context}")


def clear_context_preference(user_id: str):
    """Clear user's context preference."""
    if user_id and user_id in context_preferences:
        del context_preferences[user_id]


# =============================================================================
# DATA MODELS
# =============================================================================

class QueryRequest(BaseModel):
    query: str
    user_id: Optional[str] = None
    force_refresh: bool = False


class SourceDocument(BaseModel):
    source: str
    content: str
    title: Optional[str] = None


class QueryResponse(BaseModel):
    response: str
    source_links: list[str] = []
    answer: Optional[str] = None
    source_documents: list[SourceDocument] = []
    search_query: Optional[str] = None
    is_ambiguous: bool = False
    disambiguation_options: list[str] = []


class SetContextRequest(BaseModel):
    user_id: str
    selected_context: str


# =============================================================================
# WEB SEARCH MODULE (Using DuckDuckGo)
# =============================================================================

class WebSearcher:
    """
    Searches the web using DuckDuckGo's HTML interface.
    Free, no API key required.
    """
    
    SEARCH_URL = "https://html.duckduckgo.com/html/"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        })
        # Add retry logic with exponential backoff
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
    
    def search(self, query: str, max_results: int = 5) -> list[dict]:
        """
        Search the web and return a list of results with title, url, and snippet.
        """
        try:
            response = self.session.post(
                self.SEARCH_URL,
                data={"q": query, "b": ""},
                timeout=15
            )
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "html.parser")
            results = []
            
            for result in soup.select(".result"):
                title_elem = result.select_one(".result__title a")
                snippet_elem = result.select_one(".result__snippet")
                
                if title_elem and snippet_elem:
                    href = title_elem.get("href", "")
                    # DuckDuckGo redirects through uddg parameter
                    if "uddg=" in href:
                        parsed = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
                        url = parsed.get("uddg", [href])[0]
                    else:
                        url = href
                    
                    results.append({
                        "title": title_elem.get_text(strip=True),
                        "url": url,
                        "snippet": snippet_elem.get_text(strip=True)
                    })
                    
                    if len(results) >= max_results:
                        break
            
            logger.info(f"Found {len(results)} search results for: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []


# =============================================================================
# DYNAMIC WEB SCRAPER
# =============================================================================

class DynamicScraper:
    """
    Scrapes any website and extracts clean text content.
    Handles caching and content extraction intelligently.
    """
    
    # Only block truly binary/unreadable content types
    DISALLOWED_EXTENSIONS = {
        # Images
        "jpg", "jpeg", "png", "gif", "svg", "webp", "ico", "bmp", "tiff",
        # Archives
        "zip", "rar", "7z", "tar", "gz", "bz2",
        # Audio/Video
        "mp3", "mp4", "avi", "mov", "wmv", "flv", "mkv", "wav", "ogg", "webm",
        # Executables
        "exe", "msi", "dmg", "bin", "dll", "so",
        # Fonts
        "woff", "woff2", "ttf", "otf", "eot"
    }
    
    # Content types that are blocked (truly binary/unreadable)
    BLOCKED_CONTENT_TYPES = {
        "image/", "audio/", "video/", 
        "application/octet-stream", "application/zip", "application/x-rar",
        "application/x-7z-compressed", "application/gzip",
        "application/x-executable", "application/x-msdownload",
        "font/", "application/font"
    }
    
    def __init__(self, cache_dir: Path = PROCESSED_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        })
        # Add retry logic with exponential backoff
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self.last_request_time = 0.0  # For rate limiting
    
    def url_to_filename(self, url: str) -> str:
        """Convert URL to a safe filename using hash."""
        parsed = urlparse(url)
        safe_name = f"{parsed.netloc}{parsed.path}".replace("/", "_").replace(".", "_")
        # Add hash to ensure uniqueness
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        return f"{safe_name}_{url_hash}.txt"
    
    def is_cache_valid(self, filepath: Path) -> bool:
        """Check if cached file is still fresh."""
        if not filepath.exists():
            return False
        
        try:
            mtime = datetime.fromtimestamp(filepath.stat().st_mtime)
            age = datetime.now() - mtime
            return age < timedelta(hours=CACHE_DURATION_HOURS)
        except Exception:
            return False
    
    def can_scrape(self, url: str) -> bool:
        """Check if URL is scrapeable (not a binary file)."""
        parsed = urlparse(url)
        extension = parsed.path.split(".")[-1].lower() if "." in parsed.path else ""
        return extension not in self.DISALLOWED_EXTENSIONS
    
    def extract_text(self, soup: BeautifulSoup) -> str:
        """Extract clean text from HTML."""
        # Remove non-content elements
        for element in soup(["script", "style", "header", "footer", "nav", 
                            "aside", "form", "noscript", "iframe", "svg"]):
            element.decompose()
        
        # Get text with proper spacing
        text = soup.get_text(separator=" ", strip=True)
        
        # Clean up whitespace
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\n\s*\n", "\n\n", text)
        
        return text.strip()
    
    def extract_title(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract page title."""
        if soup.title and soup.title.string:
            return soup.title.string.strip()
        
        h1 = soup.find("h1")
        if h1:
            return h1.get_text(strip=True)
        
        return None
    
    def _extract_pdf_text(self, content: bytes, url: str) -> str:
        """Extract text from PDF content."""
        try:
            import io
            try:
                import pypdf
                reader = pypdf.PdfReader(io.BytesIO(content))
                text_parts = []
                for page in reader.pages:
                    text_parts.append(page.extract_text() or "")
                text = "\n".join(text_parts)
            except ImportError:
                try:
                    # Fallback to PyPDF2 if pypdf not available
                    import PyPDF2
                    reader = PyPDF2.PdfReader(io.BytesIO(content))
                    text_parts = []
                    for page in reader.pages:
                        text_parts.append(page.extract_text() or "")
                    text = "\n".join(text_parts)
                except ImportError:
                    logger.warning(f"PDF parsing libraries not available. Install pypdf or PyPDF2.")
                    return ""
            
            # Clean up text
            text = re.sub(r"\s+", " ", text)
            return text.strip()
        except Exception as e:
            logger.error(f"Failed to extract PDF text from {url}: {e}")
            return ""
    
    def scrape(self, url: str, force_refresh: bool = False) -> Optional[dict]:
        """
        Scrape a URL and return extracted content.
        Returns dict with: url, title, content, filename, cached
        """
        if not self.can_scrape(url):
            logger.warning(f"Cannot scrape binary file: {url}")
            return None
        
        filename = self.url_to_filename(url)
        filepath = self.cache_dir / filename
        meta_path = self.cache_dir / f"{filename}.meta.json"
        
        # Check cache
        if not force_refresh and self.is_cache_valid(filepath):
            try:
                content = filepath.read_text(encoding="utf-8")
                meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
                logger.info(f"Using cached content for: {url}")
                return {
                    "url": url,
                    "title": meta.get("title"),
                    "content": content,
                    "filename": filename,
                    "cached": True
                }
            except Exception as e:
                logger.warning(f"Cache read failed: {e}")
        
        # Fetch fresh content
        try:
            # Rate limiting: wait at least 0.5s between requests
            elapsed = time.time() - self.last_request_time
            if elapsed < 0.5:
                time.sleep(0.5 - elapsed)
            
            logger.info(f"Scraping: {url}")
            response = self.session.get(url, timeout=15)
            self.last_request_time = time.time()
            response.raise_for_status()
            
            # Validate Content-Type - block only truly binary/unreadable content
            content_type = response.headers.get("Content-Type", "").lower()
            
            # Check if content type is blocked
            is_blocked = any(blocked in content_type for blocked in self.BLOCKED_CONTENT_TYPES)
            if is_blocked:
                logger.warning(f"Skipping binary content ({content_type}): {url}")
                return None
            
            # Handle PDF files
            if "application/pdf" in content_type:
                content = self._extract_pdf_text(response.content, url)
                title = url.split("/")[-1].replace(".pdf", "").replace("_", " ").replace("-", " ")
                if not content or len(content.split()) < 30:
                    logger.warning(f"Insufficient content from PDF: {url}")
                    return None
            else:
                # Handle HTML and other text content
                soup = BeautifulSoup(response.content, "html.parser")
                content = self.extract_text(soup)
                title = self.extract_title(soup)
            
            # Minimum content threshold
            if len(content.split()) < 30:
                logger.warning(f"Insufficient content from: {url}")
                return None
            
            # Save to cache
            filepath.write_text(content, encoding="utf-8")
            meta_path.write_text(json.dumps({
                "url": url,
                "title": title,
                "scraped_at": datetime.now().isoformat(),
                "filename": filename
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            
            return {
                "url": url,
                "title": title,
                "content": content,
                "filename": filename,
                "cached": False
            }
            
        except Exception as e:
            logger.error(f"Scrape failed for {url}: {e}")
            return None


# =============================================================================
# KNOWLEDGE BASE MANAGER
# =============================================================================

class KnowledgeBase:
    """
    Manages the vector store, BM25 index, and hybrid RAG pipeline.
    Dynamically adds new content from web scrapes.
    Uses sentence-aware chunking + dual retrieval (BM25 + FAISS).
    """
    
    def __init__(self):
        self.embeddings = None
        self.vector_store = None
        self.qa_chain = None  # Legacy chain (kept for benchmark)
        self.llm = None
        self.indexed_urls: set[str] = set()
        self.semantic_splitter = None
        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        # --- Hybrid RAG components ---
        self.sentence_chunker = SentenceAwareChunker(min_tokens=256, max_tokens=512)
        self.bm25_retriever = BM25Retriever()
        self.cross_encoder = CrossEncoderReranker()
        self.context_packer = ContextPacker(max_tokens=3000, jaccard_threshold=0.85)
        self.prompt_builder = PromptBuilder()
    
    def initialize(self) -> bool:
        """Initialize embeddings, vector store, BM25 index, and LLM."""
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            logger.error("GOOGLE_API_KEY not found in environment")
            return False
        
        try:
            # Initialize embeddings
            logger.info("Loading embedding model...")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                model_kwargs={'device': 'cpu'}
            )
            
            # Initialize semantic chunker (requires embeddings)
            if SEMANTIC_CHUNKING_AVAILABLE:
                try:
                    self.semantic_splitter = SemanticChunker(
                        embeddings=self.embeddings,
                        breakpoint_threshold_type="percentile",
                        breakpoint_threshold_amount=90
                    )
                    logger.info("Semantic chunking enabled")
                except Exception as e:
                    logger.warning(f"Failed to initialize semantic chunker: {e}. Using fallback.")
                    self.semantic_splitter = None
            else:
                logger.info("Semantic chunking not available. Using regular chunking.")
            
            # Load or create vector store
            if os.path.exists(VECTOR_STORE_PATH):
                logger.info("Loading existing vector store...")
                self.vector_store = FAISS.load_local(
                    VECTOR_STORE_PATH,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                self._load_indexed_urls()
            else:
                logger.info("Creating new vector store...")
                placeholder = Document(
                    page_content="SatGraffin knowledge base initialized.",
                    metadata={"source": "system", "type": "placeholder"}
                )
                self.vector_store = FAISS.from_documents([placeholder], self.embeddings)
                self.vector_store.save_local(VECTOR_STORE_PATH)
            
            # Load or rebuild BM25 index
            if not self.bm25_retriever.load(BM25_INDEX_PATH):
                logger.info("BM25 index not found — rebuilding from FAISS docstore...")
                self._rebuild_bm25_from_faiss()
            
            # Initialize LLM
            logger.info(f"Initializing Gemini model: {GEMINI_MODEL_NAME}")
            self.llm = ChatGoogleGenerativeAI(
                model=GEMINI_MODEL_NAME,
                temperature=0.2,
                google_api_key=google_api_key,
                transport="rest"
            )
            
            # Build legacy QA chain (kept for benchmark comparison)
            self._rebuild_qa_chain()
            
            logger.info("Knowledge base initialized successfully (Hybrid RAG enabled)")
            return True
            
        except Exception as e:
            logger.exception(f"Failed to initialize knowledge base: {e}")
            return False
    
    def _rebuild_bm25_from_faiss(self):
        """Rebuild the BM25 index from all documents in the FAISS docstore."""
        if not self.vector_store:
            return
        try:
            docstore = self.vector_store.docstore
            chunks = []
            for doc_id in self.vector_store.index_to_docstore_id.values():
                doc = docstore.search(doc_id)
                if doc and hasattr(doc, 'page_content') and hasattr(doc, 'metadata'):
                    if doc.metadata.get('source') == 'system':
                        continue
                    chunks.append({"text": doc.page_content, "metadata": doc.metadata})
            if chunks:
                self.bm25_retriever.add_documents(chunks)
                self.bm25_retriever.save(BM25_INDEX_PATH)
                logger.info(f"BM25 index rebuilt with {len(chunks)} chunks from FAISS docstore")
        except Exception as e:
            logger.warning(f"Failed to rebuild BM25 index: {e}")
    
    def _rebuild_qa_chain(self):
        """Rebuild the legacy QA chain (kept for benchmark comparison)."""
        if not self.vector_store or not self.llm:
            return
        
        retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 20}
        )
        
        prompt_template = """You are SatGraffin, an intelligent AI research assistant. Answer the user's question comprehensively and accurately based on the provided context.

Rules:
1. If the context contains relevant information, use it to provide a detailed, helpful answer.
2. If the context is insufficient or irrelevant, use your general knowledge to answer the question directly.
3. Never say "according to the provided text" or "based on the context" - just answer naturally.
4. Provide clear, well-structured answers with examples when helpful.
5. For factual questions, be accurate and informative.
6. If you cite sources, mention them at the end.

DISAMBIGUATION RULE:
If the query is ambiguous or general (could apply to multiple fields/domains), structure your response as follows:
- Start with "<<DISAMBIGUATION>>" on its own line
- List each possible interpretation as "[[OPTION: Field Name]]" followed by the explanation for that field
- End with "<<END_DISAMBIGUATION>>"

Example for ambiguous query "Parallel Distributed Systems":
<<DISAMBIGUATION>>
[[OPTION: Distributed Computing and Parallel Processing]]
In computer science, parallel distributed systems refer to...

[[OPTION: Business and Supply Chain Management]]
In business contexts, distribution systems...

[[OPTION: Electrical Power Systems]]
In electrical engineering, distributed systems...
<<END_DISAMBIGUATION>>

If the user has a CONTEXT PREFERENCE set (shown below), focus your answer ONLY on that specific domain/field.
If no context preference is set and the query is specific enough, just answer normally without disambiguation markers.

Context from web sources:
{context}

User's Question: {question}

Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
    
    def _load_indexed_urls(self):
        """Load already indexed URLs from vector store metadata."""
        if not self.vector_store:
            return
        
        try:
            docstore = self.vector_store.docstore
            for doc_id in self.vector_store.index_to_docstore_id.values():
                doc = docstore.search(doc_id)
                if doc and hasattr(doc, 'metadata'):
                    source = doc.metadata.get('source', '')
                    if source and source != 'system':
                        self.indexed_urls.add(source)
            logger.info(f"Loaded {len(self.indexed_urls)} indexed URLs from vector store")
        except Exception as e:
            logger.warning(f"Failed to load indexed URLs: {e}")
    
    def is_url_indexed(self, url: str) -> bool:
        """Check if a URL is already indexed."""
        return url in self.indexed_urls
    
    def add_content(self, content: str, metadata: dict) -> int:
        """Add new content to FAISS + BM25 using sentence-aware chunking."""
        if not self.vector_store:
            return 0
        
        # Check for duplicate URLs
        url = metadata.get("source", "")
        if url and self.is_url_indexed(url):
            logger.info(f"Skipping already indexed URL: {url}")
            return 0
        
        # --- Sentence-aware chunking (hybrid RAG) ---
        hybrid_chunks = self.sentence_chunker.chunk(content, metadata)
        
        if not hybrid_chunks:
            # Fallback to legacy chunking if sentence chunker returns nothing
            doc = Document(page_content=content, metadata=metadata)
            legacy_chunks = self.fallback_splitter.split_documents([doc])
            if not legacy_chunks:
                return 0
            # Convert legacy chunks to hybrid format
            hybrid_chunks = [
                {"text": c.page_content, "metadata": {**c.metadata, "chunk_id": i}}
                for i, c in enumerate(legacy_chunks)
            ]
        
        # Add to FAISS vector store
        langchain_docs = [
            Document(page_content=c["text"], metadata=c["metadata"])
            for c in hybrid_chunks
        ]
        self.vector_store.add_documents(langchain_docs)
        self.vector_store.save_local(VECTOR_STORE_PATH)
        
        # Add to BM25 index
        self.bm25_retriever.add_documents(hybrid_chunks)
        self.bm25_retriever.save(BM25_INDEX_PATH)
        
        # Track the indexed URL
        if url:
            self.indexed_urls.add(url)
        
        # Rebuild legacy chain
        self._rebuild_qa_chain()
        
        logger.info(f"Added {len(hybrid_chunks)} chunks to FAISS+BM25 from: {metadata.get('source', 'unknown')}")
        return len(hybrid_chunks)
    
    def hybrid_query(
        self,
        question: str,
        conversation_context: str = "",
        context_preference: str | None = None,
    ) -> dict:
        """
        Full hybrid RAG query pipeline:
          1. BM25 search (top-20) + FAISS search (top-20) in parallel
          2. RRF fusion → merged top-20
          3. Cross-encoder re-rank → top-5
          4. Context packing (dedup + 3000 token budget)
          5. Citation-aware prompt → Gemini LLM
        """
        if not self.vector_store or not self.llm:
            return {
                "result": "Knowledge base not initialized.",
                "source_documents": [],
                "packed_chunks": [],
            }
        
        try:
            # --- Step 1: Dual retrieval ---
            # FAISS dense search
            faiss_docs_with_scores = self.vector_store.similarity_search_with_score(question, k=20)
            faiss_results = []
            for doc, score in faiss_docs_with_scores:
                faiss_results.append({
                    "text": doc.page_content,
                    "metadata": doc.metadata,
                    "faiss_score": float(score),
                })
            
            # BM25 sparse search
            bm25_results = self.bm25_retriever.search(question, top_k=20)
            
            logger.info(f"Dual retrieval: FAISS={len(faiss_results)}, BM25={len(bm25_results)}")
            
            # --- Step 2: RRF Fusion ---
            fused = reciprocal_rank_fusion(
                [bm25_results, faiss_results], k=60, top_n=20
            )
            logger.info(f"RRF fusion produced {len(fused)} candidates")
            
            # --- Step 3: Cross-encoder re-ranking ---
            reranked = self.cross_encoder.rerank(question, fused, top_k=5)
            logger.info(f"Cross-encoder re-ranked to {len(reranked)} results")
            
            # --- Step 4: Context packing ---
            packed = self.context_packer.pack(reranked)
            logger.info(f"Context packer selected {len(packed)} chunks")
            
            # --- Step 5: Build prompt and call LLM ---
            prompt = self.prompt_builder.build(
                query=question,
                packed_chunks=packed,
                conversation_context=conversation_context,
                context_preference=context_preference,
            )
            
            llm_response = self.llm.invoke(prompt)
            answer_text = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
            
            # Build source_documents for API response compatibility
            source_docs = [
                Document(page_content=c["text"], metadata=c["metadata"])
                for c in packed
            ]
            
            return {
                "result": answer_text,
                "source_documents": source_docs,
                "packed_chunks": packed,
            }
            
        except Exception as e:
            logger.error(f"Hybrid query failed: {e}")
            return {
                "result": f"Query failed: {str(e)}",
                "source_documents": [],
                "packed_chunks": [],
            }
    
    def query_legacy(self, question: str) -> dict:
        """Legacy FAISS-only query (kept for benchmark comparison)."""
        if not self.qa_chain:
            return {
                "result": "Knowledge base not initialized.",
                "source_documents": []
            }
        
        try:
            return self.qa_chain.invoke({"query": question})
        except Exception as e:
            logger.error(f"Legacy query failed: {e}")
            return {
                "result": f"Query failed: {str(e)}",
                "source_documents": []
            }


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    success = knowledge_base.initialize()
    if not success:
        logger.error("Failed to initialize knowledge base - check your API key")
    yield
    # Shutdown (cleanup if needed)
    logger.info("Shutting down SatGraffin API")


app = FastAPI(
    title="SatGraffin API",
    description="General-purpose AI research assistant that searches the web and provides grounded answers.",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
web_searcher = WebSearcher()
scraper = DynamicScraper()
knowledge_base = KnowledgeBase()
executor = ThreadPoolExecutor(max_workers=4)  # Thread pool for blocking I/O


@app.get("/")
def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "message": "SatGraffin API is running",
        "version": "2.0.0"
    }


@app.get("/api/health")
def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "knowledge_base_ready": knowledge_base.qa_chain is not None,
        "embeddings_loaded": knowledge_base.embeddings is not None
    }


@lru_cache(maxsize=100)
def generate_search_query(user_query: str) -> str:
    """
    Refine the user query for better search results.
    Cached to avoid reprocessing identical queries.
    """
    # Remove common question words that don't help search
    cleaned = re.sub(r"^(what is|who is|how to|why does|when did|where is|can you|tell me about)\s+", "", user_query.lower(), flags=re.IGNORECASE)
    return cleaned.strip() or user_query


@app.post("/api/query", response_model=QueryResponse)
async def process_query(request: QueryRequest, background_tasks: BackgroundTasks):
    """
    Process a user query:
    1. Search the web for relevant information
    2. Scrape top results (prioritized by source quality)
    3. Add content to knowledge base
    4. Generate grounded answer using RAG with conversation context
    """
    
    if not knowledge_base.qa_chain:
        return QueryResponse(
            response="The knowledge base is not ready. Please check the server logs.",
            answer="The knowledge base is not ready. Please check the server logs."
        )
    
    user_query = request.query.strip()
    user_id = request.user_id
    logger.info(f"Processing query: {user_query} (user: {user_id})")
    
    loop = asyncio.get_event_loop()
    
    # Step 1: Search the web (run in thread pool)
    search_query = generate_search_query(user_query)
    search_results = await loop.run_in_executor(
        executor, web_searcher.search, search_query, MAX_SEARCH_RESULTS
    )
    
    if not search_results:
        logger.warning("No search results found, querying existing knowledge base")
    else:
        # Sort search results by source quality score
        search_results = sorted(
            search_results, 
            key=lambda r: score_source(r.get("url", "")), 
            reverse=True
        )
        
        # Step 2: Scrape top results and add to knowledge base
        scraped_count = 0
        for result in search_results[:MAX_SCRAPE_PAGES]:
            url = result.get("url", "")
            if not url:
                continue
            
            # Run scraping in thread pool
            scraped = await loop.run_in_executor(
                executor, scraper.scrape, url, request.force_refresh
            )
            if scraped and scraped.get("content"):
                # Add to knowledge base with quality score
                metadata = {
                    "source": url,
                    "title": scraped.get("title") or result.get("title", ""),
                    "search_query": search_query,
                    "scraped_at": datetime.now().isoformat(),
                    "quality_score": score_source(url)
                }
                chunks_added = knowledge_base.add_content(scraped["content"], metadata)
                if chunks_added > 0:
                    scraped_count += 1
        
        logger.info(f"Scraped and indexed {scraped_count} pages")
    
    # Step 3: Build query with conversation context and context preference
    conversation_context = get_conversation_context(user_id) if user_id else ""
    context_preference = get_context_preference(user_id) if user_id else None
    
    # Step 4: Hybrid RAG query (BM25 + FAISS → RRF → cross-encoder → context pack → LLM)
    result = await loop.run_in_executor(
        executor,
        knowledge_base.hybrid_query,
        user_query,
        conversation_context,
        context_preference,
    )
    
    # Step 5: Format response and parse disambiguation
    raw_answer = result.get("result", "I couldn't find a good answer to your question.")
    
    # Parse disambiguation markers from response
    is_ambiguous = False
    disambiguation_options = []
    answer_text = raw_answer
    
    if "<<DISAMBIGUATION>>" in raw_answer and "<<END_DISAMBIGUATION>>" in raw_answer:
        is_ambiguous = True
        # Extract content between markers
        start = raw_answer.find("<<DISAMBIGUATION>>") + len("<<DISAMBIGUATION>>")
        end = raw_answer.find("<<END_DISAMBIGUATION>>")
        disambiguation_content = raw_answer[start:end].strip()
        
        # Parse options using regex
        import re
        option_pattern = r'\[\[OPTION:\s*([^\]]+)\]\]'
        options = re.findall(option_pattern, disambiguation_content)
        disambiguation_options = [opt.strip() for opt in options]
        
        # Clean up the response text - remove markers but keep content
        answer_text = disambiguation_content
        # Remove the OPTION markers for cleaner display
        answer_text = re.sub(r'\[\[OPTION:\s*([^\]]+)\]\]', r'**\1:**', answer_text)
        answer_text = answer_text.strip()
        
        logger.info(f"Detected ambiguous query with {len(disambiguation_options)} options: {disambiguation_options}")
    
    source_documents = []
    source_links = []
    seen_sources = set()
    
    # Collect source documents with quality scores
    sources_with_scores = []
    for doc in result.get("source_documents", []):
        source = doc.metadata.get("source", "")
        if source and source not in seen_sources and source != "system":
            seen_sources.add(source)
            quality = doc.metadata.get("quality_score", score_source(source))
            sources_with_scores.append({
                "source": source,
                "content": doc.page_content[:500],
                "title": doc.metadata.get("title"),
                "quality": quality
            })
    
    # Sort by quality score (highest first)
    sources_with_scores.sort(key=lambda x: x["quality"], reverse=True)
    
    for s in sources_with_scores:
        source_links.append(s["source"])
        source_documents.append(SourceDocument(
            source=s["source"],
            content=s["content"],
            title=s["title"]
        ))
    
    # Also add search results as potential sources (sorted by quality)
    remaining_results = [r for r in search_results if r.get("url") not in seen_sources]
    remaining_results.sort(key=lambda r: score_source(r.get("url", "")), reverse=True)
    for res in remaining_results:
        url = res.get("url", "")
        if url:
            source_links.append(url)
    
    # Step 6: Save to conversation memory
    if user_id:
        add_to_memory(user_id, user_query, answer_text)
    
    return QueryResponse(
        response=answer_text,
        answer=answer_text,
        source_links=source_links[:10],  # Limit sources
        source_documents=source_documents[:5],
        search_query=search_query,
        is_ambiguous=is_ambiguous,
        disambiguation_options=disambiguation_options
    )


@app.post("/api/clear-cache")
def clear_cache():
    """Clear all cached scraped content."""
    try:
        count = 0
        for file in PROCESSED_DIR.glob("*"):
            file.unlink()
            count += 1
        return {"status": "ok", "files_deleted": count}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/clear-memory")
def clear_memory(user_id: Optional[str] = None):
    """Clear conversation memory for a user or all users."""
    try:
        if user_id:
            if user_id in conversation_memory:
                del conversation_memory[user_id]
            # Also clear context preference
            clear_context_preference(user_id)
            return {"status": "ok", "message": f"Memory cleared for user {user_id}"}
        else:
            conversation_memory.clear()
            context_preferences.clear()
            return {"status": "ok", "message": "All conversation memory cleared"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/set-context")
def set_context(request: SetContextRequest):
    """Set user's context preference for disambiguation."""
    try:
        set_context_preference(request.user_id, request.selected_context)
        return {
            "status": "ok", 
            "message": f"Context set to: {request.selected_context}",
            "selected_context": request.selected_context
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/clear-context")
def clear_context(user_id: str):
    """Clear user's context preference."""
    try:
        clear_context_preference(user_id)
        return {"status": "ok", "message": "Context preference cleared"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
