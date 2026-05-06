import logging
import os
import json
from dotenv import load_dotenv
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from datetime import datetime, timedelta
from pathlib import Path

# --- LANGCHAIN IMPORTS for the RAG Chain ---
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_google_genai import ChatGoogleGenerativeAI # The Google Gemini model
from langchain.chains import RetrievalQA # The RAG chain
from langchain.schema import Document  # For routing index

# --- 1. Load API Key ---
# This loads the GOOGLE_API_KEY from your .env file
load_dotenv()
# Build homepage link index for on-demand scraping
import requests
from bs4 import BeautifulSoup
from difflib import get_close_matches
from urllib.parse import urljoin
HOMEPAGE_URL = "https://mosdac.gov.in/"
link_index: dict[str, str] = {}

# --- Intent classification for routing ---
def classify_intent(query: str) -> str:
    q = query.lower()
    if 'download' in q or 'api' in q:
        return 'download_api'
    if any(k in q for k in ['mission', 'satellite', 'insat', 'scatsat', 'kalpana']):
        return 'satellite_info'
    if any(k in q for k in ['release', 'announcement', 'product']):
        return 'product_announcement'
    return 'general'

INTENT_HINTS: dict[str, list[str]] = {
    'download_api': ['download api', 'api client', 'mdapi.zip', 'downloadapi-manual'],
    'satellite_info': ['insat-3dr', 'insat-3a', 'kalpana-1', 'scatsat-1'],
    'product_announcement': ['release announcement', 'nwcsaf', 'product release'],
    'general': []
}

DATA_DIR = Path("data")
PROCESSED_DIR = Path("processed_data")
LINK_INDEX_CACHE_PATH = DATA_DIR / "link_index.json"

MANUAL_LINKS: dict[str, str] = {
    "Kalpana 1": "https://mosdac.gov.in/kalpana-1",
    "Kalpana-1": "https://mosdac.gov.in/kalpana-1",
    "INSAT 3DR": "https://mosdac.gov.in/insat-3dr",
    "INSAT-3DR": "https://mosdac.gov.in/insat-3dr",
    "SCATSAT-1": "https://mosdac.gov.in/scatsat-1",
    "SCATSAT 1": "https://mosdac.gov.in/scatsat-1",
}


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _sanitize_cache_entries(cached: dict[str, str]) -> dict[str, str]:
    return {
        str(key): str(value)
        for key, value in cached.items()
        if isinstance(key, str) and isinstance(value, str)
    }


def register_link_candidates(url: str, candidates: list[str]) -> int:
    added = 0
    for candidate in candidates:
        if not candidate:
            continue
        key = candidate.strip()
        if not key or key in link_index:
            continue
        link_index[key] = url
        added += 1
    return added


def load_link_index_cache() -> None:
    if LINK_INDEX_CACHE_PATH.exists():
        try:
            cached = json.loads(LINK_INDEX_CACHE_PATH.read_text(encoding="utf-8"))
            link_index.update(_sanitize_cache_entries(cached))
            logger.info("Loaded %d cached link entries", len(link_index))
        except Exception as exc:
            logger.warning("Failed to load link index cache: %s", exc)


def persist_link_index_cache() -> None:
    try:
        _ensure_data_dir()
        LINK_INDEX_CACHE_PATH.write_text(
            json.dumps(link_index, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Persisted link index cache (%d entries)", len(link_index))
    except Exception as exc:
        logger.warning("Failed to persist link index cache: %s", exc)


def hydrate_link_index_from_metadata() -> None:
    if not PROCESSED_DIR.exists():
        return
    added = 0
    for meta_path in PROCESSED_DIR.glob("*.meta.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        url = meta.get("url")
        if not url:
            continue
        title = meta.get("title")
        candidates: list[str] = []
        if title:
            candidates.append(title)
        slug = meta.get("filename")
        if slug:
            cleaned = slug.replace(".txt", "")
            candidates.append(cleaned)
            candidates.append(cleaned.replace("_", " "))
        candidates.append(url)
        added += register_link_candidates(url, candidates)
    if added:
        logger.info("Hydrated %d link entries from metadata", added)
        persist_link_index_cache()


def register_link_from_metadata(filename: str, url: str) -> None:
    meta_path = PROCESSED_DIR / f"{filename}.meta.json"
    if not meta_path.exists():
        return
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return
    title = meta.get("title")
    candidates: list[str] = []
    if title:
        candidates.append(title)
    candidates.append(meta.get("filename", filename).replace(".txt", ""))
    candidates.append(url)
    if register_link_candidates(url, candidates):
        persist_link_index_cache()


def seed_manual_links() -> None:
    added = 0
    for label, url in MANUAL_LINKS.items():
        added += register_link_candidates(url, [label, label.lower()])
    if added:
        logger.info("Seeded %d manual link aliases", added)

# --- Utility: fallback link extractor if routing fails ---
def extract_relevant_link(query: str) -> str | None:
    """Fuzzy-match query against link_index keys"""
    import re
    from difflib import get_close_matches
    keys = list(link_index.keys())
    lower_keys = [k.lower() for k in keys]
    q = query.lower()
    # Direct containment of key tokens
    tokens = [tok for tok in re.findall(r"\w+", q) if len(tok) > 3]
    for key, url in link_index.items():
        lk = key.lower()
        if any(tok in lk for tok in tokens):
            return url
    # Fuzzy match on link text
    match = get_close_matches(q, lower_keys, n=1, cutoff=0.4)
    if match:
        orig = keys[lower_keys.index(match[0])]
        return link_index[orig]
    return None

# --- Smart Caching Configuration ---
CACHE_DURATION_HOURS = int(os.getenv("CACHE_DURATION_HOURS", "24"))  # Default 24 hours
page_cache_timestamps: dict[str, datetime] = {}  # url -> last_scraped_time

# --- 2. Data Models ---
class QueryRequest(BaseModel):
    query: str
    user_id: str | None = None
    force_refresh: bool = False  # If true, forces re-scraping of the relevant page

class SourceDocument(BaseModel):
    source: str
    content: str

class QueryResponse(BaseModel):
    response: str
    source_links: list[str] = []
    answer: str | None = None
    source_documents: list[SourceDocument] = []

# --- 3. Initialize FastAPI App ---
app = FastAPI(
    title="Satgraffin API",
    description="API for retrieving information from the Satgraffin knowledge base.",
    version="0.1.0",
)

# --- Set up logging and CORS ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("satgraffin")

seed_manual_links()
load_link_index_cache()
hydrate_link_index_from_metadata()

allowed_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173",
).split(",")

app.add_middleware(
    CORSMiddleware,
    # Allow all origins for development; tighten in production
    allow_origins=["*"],
    allow_origin_regex=r".*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 4. Load Knowledge Base and Set Up RAG Chain ---
DB_FAISS_PATH = "vector_store"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")

 # Global variables for the RAG chain, embeddings model, and routing retriever
qa_chain = None
embeddings_model = None  # Cached HuggingFaceEmbeddings instance
route_retriever = None  # FAISS retriever for routing queries to URLs

@app.on_event("startup")
def load_knowledge_base():
    """Loads all necessary models and sets up the RAG chain on server startup."""
    global qa_chain, embeddings_model, route_retriever
    logger.info("Loading knowledge base and setting up RAG chain...")

    # Verify API key
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        logger.warning("GOOGLE_API_KEY not found. RAG chain will not be initialized.")
        qa_chain = None
        return

    # Initialize and cache embeddings model once, then set up chain
    try:
        from langchain_huggingface import HuggingFaceEmbeddings as _HuggingFaceEmbeddings
        embeddings_model = _HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'}
        )
        db = FAISS.load_local(
            DB_FAISS_PATH, embeddings_model, allow_dangerous_deserialization=True
        )
        # --- Pre-index any existing .txt files in processed_data (HTML or PDF) ---
        try:
            from pathlib import Path
            from langchain_community.document_loaders import TextLoader
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            docs_to_add = []
            splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
            for file_path in Path("processed_data").glob("*.txt"):
                loader = TextLoader(str(file_path), encoding="utf-8")
                docs = loader.load()
                chunks = splitter.split_documents(docs)
                docs_to_add.extend(chunks)
            if docs_to_add:
                db.add_documents(docs_to_add)
                db.save_local(DB_FAISS_PATH)
                logger.info(f"Pre-indexed {len(docs_to_add)} existing files into FAISS store.")
        except Exception as e:
            logger.warning(f"Pre-indexing existing files failed: {e}")
        # MMR retriever
        retriever = db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 3, "fetch_k": 20}
        )
        logger.info("Using Gemini model %s", GEMINI_MODEL_NAME)
        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL_NAME,
            temperature=0.3,
            google_api_key=google_api_key,
            transport="rest",
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="map_reduce",
            retriever=retriever,
            return_source_documents=True,
        )
        logger.info("RAG chain setup complete.")
    except Exception as e:
        logger.exception("Error during RAG chain setup: %s", e)
        qa_chain = None
        return
    # Build homepage link index from HTML anchors
    try:
        resp = requests.get(HOMEPAGE_URL, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "html.parser")
        for a in soup.find_all("a", href=True):
            text = a.get_text(strip=True)
            href = a["href"]
            if text:
                full = urljoin(HOMEPAGE_URL, href)
                link_index[text] = full
        logger.info(f"Homepage link index built ({len(link_index)} entries)")
    except Exception as e:
        logger.warning(f"Failed to build homepage links: {e}")
    # Attempt to fetch sitemap.xml to seed deeper pages
    try:
        sitemap_url = urljoin(HOMEPAGE_URL, 'sitemap.xml')
        sm = requests.get(sitemap_url, timeout=10)
        sm.raise_for_status()
        from xml.etree import ElementTree as ET
        root = ET.fromstring(sm.content)
        for loc in root.findall('.//{*}loc'):
            url = loc.text.strip()
            key = url.split('/')[-1] or url
            link_index[key] = url
        logger.info(f"Sitemap indexed ({len(link_index)} total entries)")
    except Exception as e:
        logger.warning(f"Failed to fetch sitemap.xml: {e}")
    persist_link_index_cache()
    # Load persisted routing index if available, then kick off background rebuild
    ROUTING_STORE = "routing_store"
    if os.path.exists(ROUTING_STORE):
        try:
            disk_db = FAISS.load_local(ROUTING_STORE, embeddings_model, allow_dangerous_deserialization=True)
            route_retriever = disk_db.as_retriever(search_type="mmr", search_kwargs={"k":1, "fetch_k":5})
            logger.info(f"Loaded routing index from disk ({len(disk_db.index_to_docstore_id)} entries)")
        except Exception as e:
            logger.warning(f"Failed to load persisted routing index: {e}")
    # Background rebuild of routing index
    def _rebuild_routing():
        docs: list[Document] = []
        snapshot = list(link_index.items())
        for key, url in snapshot:
            if not url.lower().startswith(("http://","https://")):
                continue
            try:
                resp = requests.get(url, timeout=5)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.content, 'html.parser')
                p = soup.find('p')
                snippet = p.get_text(strip=True) if p else key
            except Exception:
                snippet = key
            docs.append(Document(page_content=snippet, metadata={"source":url}))
        # If no pages were found, skip rebuild
        if not docs:
            logger.warning("Background routing rebuild found no documents; skipping FAISS rebuild.")
            return
        idx = FAISS.from_documents(docs, embeddings_model)
        try:
            idx.save_local(ROUTING_STORE)
            global route_retriever
            route_retriever = idx.as_retriever(search_type='mmr', search_kwargs={'k':1,'fetch_k':5})
            logger.info(f"Background routing rebuild complete ({len(docs)} entries)")
        except Exception as e:
            logger.warning(f"Failed to persist routing index: {e}")
    import threading
    threading.Thread(target=_rebuild_routing, daemon=True).start()

# --- 5. Define API Endpoints ---
@app.get("/")
def read_root():
    return {"status": "ok", "message": "Welcome to the MOSDAC Bot API!"}

def needs_refresh(page_path: str, url: str) -> bool:
    """
    Check if a page needs to be re-scraped based on cache duration.
    Returns True if page doesn't exist OR is older than CACHE_DURATION_HOURS.
    """
    # If file doesn't exist, definitely needs refresh
    if not os.path.exists(page_path):
        return True
    
    # Check in-memory cache timestamp
    last_scraped = page_cache_timestamps.get(url)
    
    # If not in memory, check file modification time
    if not last_scraped:
        try:
            file_mtime = os.path.getmtime(page_path)
            last_scraped = datetime.fromtimestamp(file_mtime)
            page_cache_timestamps[url] = last_scraped
        except Exception as e:
            logger.warning(f"Could not get file mtime for {page_path}: {e}")
            return True
    
    # Calculate age
    age = datetime.now() - last_scraped
    cache_duration = timedelta(hours=CACHE_DURATION_HOURS)
    
    if age > cache_duration:
        logger.info(f"Page {url} is {age.total_seconds()/3600:.1f} hours old (cache: {CACHE_DURATION_HOURS}h). Needs refresh.")
        return True
    else:
        logger.info(f"Page {url} is {age.total_seconds()/3600:.1f} hours old. Using cached version.")
        return False

def update_page_data(url: str):
    # Scrape page and update vector store
    from scripts.scraper import scrape_and_update, slugify_url
    from langchain_community.document_loaders import TextLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from os import path
    filename = scrape_and_update(url)
    if not filename:
        return
    page_path = path.join("processed_data", filename)
    register_link_from_metadata(filename, url)
    
    # Update cache timestamp
    page_cache_timestamps[url] = datetime.now()
    
    # Load and split
    loader = TextLoader(page_path, encoding="utf-8")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
    texts = splitter.split_documents(docs)
    # Update vector store
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={'device':'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    db.add_documents(texts)
    db.save_local(DB_FAISS_PATH)
    # Reload chain
    global qa_chain
    retriever = db.as_retriever(search_type="mmr", search_kwargs={"k":3, "fetch_k":20})
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL_NAME, temperature=0.3, google_api_key=os.getenv("GOOGLE_API_KEY"), transport="rest")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="map_reduce", retriever=retriever, return_source_documents=True)

@app.post("/api/query", response_model=QueryResponse)
def process_query(request: QueryRequest, background_tasks: BackgroundTasks):
    """
    Receives a user query, processes it through the RAG chain,
    and returns a generated answer with sources.
    """

    # --- Dynamic Routing via Sitemap Vectorization ---
    # Attempt explicit URL in query
    import re
    url_match = re.search(r"https?://[\w./?=&%-]+", request.query)
    page_url = url_match.group(0) if url_match else None
    # If no explicit URL, use routing retriever to pick best URL
    if not page_url and route_retriever:
        route_docs = route_retriever.invoke(request.query)
        if route_docs:
            page_url = route_docs[0].metadata.get('source')
            logger.info(f"Routing-based URL selected: {page_url}")
    # Fallback to fuzzy link matching
    if not page_url:
        page_url = extract_relevant_link(request.query)
    from scripts.scraper import slugify_url
    # If the resolved URL is HTML, check for embedded PDF links and redirect to PDF
    if page_url and not page_url.lower().endswith('.pdf'):
        try:
            resp = requests.get(page_url, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.content, 'html.parser')
            pdf_anchors = [a for a in soup.find_all('a', href=True) if a['href'].lower().endswith('.pdf')]
            if pdf_anchors:
                # choose the PDF link with anchor text/URL best matching the query
                from difflib import get_close_matches
                # prepare candidate texts for matching (anchor text or filename)
                candidates = []
                for a in pdf_anchors:
                    text = a.get_text(strip=True)
                    if not text:
                        # fallback to filename from URL
                        text = os.path.basename(a['href'])
                    candidates.append(text.lower())
                q = request.query.lower()
                match = get_close_matches(q, candidates, n=1, cutoff=0.3)
                if match:
                    idx = candidates.index(match[0])
                    selected = pdf_anchors[idx]
                else:
                    selected = pdf_anchors[0]
                pdf_link = urljoin(page_url, selected['href'])
                logger.info(f"Redirecting to PDF link: {pdf_link}")
                page_url = pdf_link
        except Exception as e:
            logger.warning(f"Error fetching PDF links from {page_url}: {e}")
    # Only proceed if a URL was determined
    if page_url:
        # Determine local filename and path
        page_filename = slugify_url(page_url)
        page_path = os.path.join("processed_data", page_filename)
        
        # Forced user refresh: always re-scrape synchronously
        if request.force_refresh:
            logger.info(f"User requested force refresh for %s", page_filename)
            update_page_data(page_url)
        else:
            # Smart caching: Check if page needs refresh
            needs_update = needs_refresh(page_path, page_url)
            
            # First-time fetch: synchronous block
            if needs_update and not os.path.exists(page_path):
                logger.info(f"Page data for %s missing; fetching synchronously.", page_filename)
                update_page_data(page_url)
                logger.info(f"Fetched and processed %s; continuing.", page_filename)
            # Stale but exists: background refresh, use cache now
            elif needs_update and os.path.exists(page_path):
                logger.info(f"Page data for %s stale; using cache and refreshing in background.", page_filename)
                background_tasks.add_task(update_page_data, page_url)
            # Fresh cache: use immediately
            else:
                logger.info(f"Using fresh cache for %s; no refresh needed.", page_filename)
    # Continue with current qa_chain
    if qa_chain is None:
        message = (
            "The retrieval chain isn't ready yet (missing credentials or vector store). "
            "Please configure the backend and try again."
        )
        logger.error(message)
        return QueryResponse(
            response=message,
            answer=message,
            source_links=[],
            source_documents=[],
        )

    logger.info("Processing query: %s", request.query)

    try:
        result = qa_chain.invoke({"query": request.query})
    except Exception as err:
        logger.exception("Query processing failed: %s", err)
        message = (
            "I hit a snag while talking to the SatGraffin knowledge graph. Please retry shortly."
        )
        return QueryResponse(
            response=message,
            answer=message,
            source_links=[],
            source_documents=[],
        )

    # Format the source documents
    source_documents_formatted: list[SourceDocument] = []
    source_links: list[str] = []
    for doc in result.get("source_documents", []):
        source = doc.metadata.get("source", "Unknown")
        source_documents_formatted.append(SourceDocument(source=source, content=doc.page_content))
        if source and source not in source_links:
            source_links.append(source)

    answer_text = result.get("result", "No answer could be generated.")

    return QueryResponse(
        response=answer_text,
        answer=answer_text,
        source_links=source_links,
        source_documents=source_documents_formatted,
    )

