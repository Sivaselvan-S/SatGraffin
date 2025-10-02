import logging
import os
from dotenv import load_dotenv
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- LANGCHAIN IMPORTS for the RAG Chain ---
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_google_genai import ChatGoogleGenerativeAI # The Google Gemini model
from langchain.chains import RetrievalQA # The RAG chain

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

# --- 2. Data Models ---
class QueryRequest(BaseModel):
    query: str
    user_id: str | None = None

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

# Global variable for the RAG chain
qa_chain = None

@app.on_event("startup")
def load_knowledge_base():
    """Loads all necessary models and sets up the RAG chain on server startup."""
    global qa_chain
    logger.info("Loading knowledge base and setting up RAG chain...")

    try:
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            logger.warning("GOOGLE_API_KEY not found. RAG chain will not be initialized.")
            qa_chain = None
            return

        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'}
        )
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

        # Use MMR retriever for balanced relevance and diversity
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

        # Use a multi-stage (map-reduce) chain for more accurate answers on long documents
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="map_reduce",
            retriever=retriever,
            return_source_documents=True,
        )

        logger.info("RAG chain setup complete.")
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
                # Use last segment as key if no anchor text
                key = url.split('/')[-1] or url
                link_index[key] = url
            logger.info(f"Sitemap indexed ({len(link_index)} total entries)")
        except Exception as e:
            logger.warning(f"Failed to fetch sitemap.xml: {e}")
    except Exception as e:
        logger.exception("Error during startup: %s", e)
        qa_chain = None

# --- 5. Define API Endpoints ---
@app.get("/")
def read_root():
    return {"status": "ok", "message": "Welcome to the MOSDAC Bot API!"}

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

    # --- Real-time page fetch logic ---
    # 1. Try to extract a relevant MOSDAC subpage link from the main page if the query mentions a known mission/page
    # 2. If a new page is needed, scrape it, update processed_data, and re-embed

    def extract_relevant_link(query: str) -> str | None:
        import re
        from difflib import get_close_matches
        # Fuzzy match against homepage link text (case-insensitive)
        keys = list(link_index.keys())
        lower_keys = [k.lower() for k in keys]
        q = query.lower()
        # Exact numeric match for patterns like 'oceansat 3'
        num_match = re.search(r"([a-zA-Z]+)[\s-]*(\d+)", q)
        if num_match:
            candidate = f"{num_match.group(1)}-{num_match.group(2)}"
            if candidate in lower_keys:
                orig = keys[lower_keys.index(candidate)]
                return link_index[orig]
        # Try direct fuzzy match on link text
        match = get_close_matches(q, lower_keys, n=1, cutoff=0.4)
        if match:
            orig = keys[lower_keys.index(match[0])]
            return link_index[orig]
        # Try keyword containment: any major token in a link text
        tokens = [tok for tok in re.findall(r"\w+", q) if len(tok) > 3]
        for key, url in link_index.items():
            lk = key.lower()
            if any(tok in lk for tok in tokens):
                return url
        return None


    from scripts.scraper import slugify_url
    page_url = extract_relevant_link(request.query)
    if page_url:
        # Determine local filename and path
        page_filename = slugify_url(page_url)
        page_path = os.path.join("processed_data", page_filename)
        # If we don't have the file yet, fetch and wait (synchronously)
        if not os.path.exists(page_path):
            logger.info(f"Page data for %s not found locally. Starting synchronous fetch and processing.", page_filename)
            # Synchronously fetch and process the page
            update_page_data(page_url)
            logger.info(f"Processing of %s complete. Continuing with QA chain.", page_filename)
        # Otherwise, schedule a background refresh but proceed with QA
        else:
            background_tasks.add_task(update_page_data, page_url)
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

