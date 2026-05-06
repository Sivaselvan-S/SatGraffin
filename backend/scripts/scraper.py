import os
import re
from collections import deque
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime
from functools import wraps
import json

# --- Retry Decorator for Robust Fetching ---
def retry_on_failure(max_retries: int = 3, delay: float = 2.0):
    """
    Decorator to retry a function on exception with exponential backoff.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries:
                        print(f"All {max_retries} retries failed for {func.__name__}: {e}")
                        raise
                    wait = delay * (2 ** (attempt - 1))
                    print(f"Error in {func.__name__}: {e}. Retrying in {wait}s...")
                    time.sleep(wait)
        return wrapper
    return decorator

# --- Configuration ---
START_URLS = [
    "https://mosdac.gov.in/",
]
OUTPUT_DIR = "processed_data"
MAX_PAGES = int(os.getenv("SCRAPER_MAX_PAGES", "60"))
MIN_WORDS_PER_PAGE = int(os.getenv("SCRAPER_MIN_WORDS", "40"))
ALLOWED_DOMAINS = {urlparse(url).netloc for url in START_URLS}
DISALLOWED_EXTENSIONS = {"pdf", "jpg", "jpeg", "png", "gif", "svg", "zip", "ppt", "pptx", "xls", "xlsx", "doc", "docx"}


def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    normalized_path = parsed.path.rstrip("/") or "/"
    return parsed._replace(fragment="", query="", path=normalized_path).geturl()


def should_follow(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    if parsed.netloc not in ALLOWED_DOMAINS:
        return False
    extension = parsed.path.split(".")[-1].lower()
    if extension and extension in DISALLOWED_EXTENSIONS:
        return False
    return True


def extract_links(soup: BeautifulSoup, base_url: str) -> set[str]:
    links = set()
    for anchor in soup.find_all("a", href=True):
        absolute = urljoin(base_url, anchor["href"])
        if should_follow(absolute):
            links.add(normalize_url(absolute))
    return links


def clean_text(soup: BeautifulSoup) -> str:
    for element in soup(["script", "style", "header", "footer", "nav", "aside"]):
        element.decompose()
    text = soup.get_text(separator=" ", strip=True)
    cleaned_text = re.sub(r"\s+", " ", text)
    return cleaned_text


@retry_on_failure(max_retries=3, delay=2.0)
def fetch_page(url: str) -> tuple[str | None, set[str], str | None]:
    print(f"Scraping URL: {url}")
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"Error fetching {url}: {exc}")
        return None, set(), None
    except Exception as exc:  # Catch anything unexpected (SSL errors, etc.)
        print(f"Unexpected error fetching {url}: {exc}")
        return None, set(), None

    soup = BeautifulSoup(response.content, "html.parser")
    discovered_links = extract_links(soup, url)
    text = clean_text(soup)
    title = soup.title.string.strip() if soup.title and soup.title.string else None
    if len(text.split()) < MIN_WORDS_PER_PAGE:
        print(f"Skipping {url}: not enough textual content")
        return None, discovered_links, title
    return text, discovered_links, title

def extract_pdf_text(url: str) -> str | None:
    """
    Download and extract text from a PDF URL using pdfplumber for better accuracy and table extraction.
    Falls back to pypdf if pdfplumber is not available.
    """
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        import io
        text_content: list[str] = []
        # Prefer pdfplumber for improved extraction
        try:
            import pdfplumber
            use_pdfplumber = True
        except ImportError:
            use_pdfplumber = False
        if use_pdfplumber:
            with pdfplumber.open(io.BytesIO(response.content)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    if page_text:
                        text_content.append(page_text)
                    # Extract and format tables as key-value pairs
                    tables = page.extract_tables() or []
                    for table in tables:
                        if not table:
                            continue
                        headers = table[0]
                        normalized_headers: list[str] = []
                        for idx, header in enumerate(headers):
                            header_text = (header or "").strip()
                            if not header_text:
                                header_text = f"Column {idx + 1}"
                            normalized_headers.append(header_text)
                        for row in table[1:]:
                            if not row:
                                continue
                            pairs: list[str] = []
                            for idx, cell in enumerate(row[:len(normalized_headers)]):
                                cell_text = (cell or "").strip()
                                if not cell_text:
                                    continue
                                header = normalized_headers[idx]
                                pairs.append(f"{header}: {cell_text}")
                            if pairs:
                                text_content.append(" | ".join(pairs))
        else:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(response.content))
            for page in reader.pages:
                page_text = page.extract_text() or ""
                if page_text:
                    text_content.append(page_text)
        full_text = "\n".join(text_content)
        # Normalize bullet characters for clearer lists
        try:
            full_text = full_text.replace("•", "- ")
        except Exception:
            pass
        # Collapse duplicate consecutive lines and normalize whitespace
        cleaned_lines: list[str] = []
        last_line: str | None = None
        for line in full_text.splitlines():
            normalized = line.strip()
            if not normalized:
                if cleaned_lines and cleaned_lines[-1] != "":
                    cleaned_lines.append("")
                continue
            if normalized == last_line:
                continue
            cleaned_lines.append(normalized)
            last_line = normalized
        return "\n".join(cleaned_lines).strip()
    except Exception as e:
        print(f"Error extracting PDF text from {url}: {e}")
        return None


def slugify_url(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.strip("/") or "home"
    safe_path = re.sub(r"[^a-zA-Z0-9_-]", "_", path)[:120]
    if not safe_path:
        safe_path = "page"
    return f"{parsed.netloc.replace('.', '_')}_{safe_path}.txt"


def save_text(filename: str, text: str) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as handle:
        handle.write(text)
    print(f"Saved cleaned text to {filepath}")


def save_metadata(filename: str, url: str, title: str | None = None) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    meta_path = os.path.join(OUTPUT_DIR, f"{filename}.meta.json")
    metadata = {
        "url": url,
        "filename": filename,
        "fetched_at": datetime.utcnow().isoformat() + "Z",
    }
    if title:
        metadata["title"] = title
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)



# --- On-demand real-time page fetch and update ---
def scrape_and_update(url: str) -> str | None:
    """
    Scrape a single page or PDF, save its cleaned text, and return the filename if successful.
    """
    # Handle PDF URLs separately
    if url.lower().endswith('.pdf'):
        text = extract_pdf_text(url)
        if text:
            filename = slugify_url(url)
            save_text(filename, text)
            save_metadata(filename, url)
            return filename
        return None
    # Otherwise, fetch HTML page
    text, _, title = fetch_page(url)
    if text:
        filename = slugify_url(url)
        save_text(filename, text)
        save_metadata(filename, url, title)
        return filename
    return None

# For compatibility, keep the old crawl_site for batch mode (not used in real-time)
def crawl_site() -> None:
    ... # ...existing code...

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # Usage: python scraper.py <url>
        url = sys.argv[1]
        scrape_and_update(url)
    else:
        crawl_site()
