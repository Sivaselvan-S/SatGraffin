import os
import re
from collections import deque
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

# --- Configuration ---
START_URLS = [
    "https://mosdac.gov.in/",
]
OUTPUT_DIR = "processed_data"
MAX_PAGES = int(os.getenv("SCRAPER_MAX_PAGES", "60"))
MIN_WORDS_PER_PAGE = int(os.getenv("SCRAPER_MIN_WORDS", "120"))
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


def fetch_page(url: str) -> tuple[str | None, set[str]]:
    print(f"Scraping URL: {url}")
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"Error fetching {url}: {exc}")
        return None, set()
    except Exception as exc:  # Catch anything unexpected (SSL errors, etc.)
        print(f"Unexpected error fetching {url}: {exc}")
        return None, set()

    soup = BeautifulSoup(response.content, "html.parser")
    discovered_links = extract_links(soup, url)
    text = clean_text(soup)
    if len(text.split()) < MIN_WORDS_PER_PAGE:
        print(f"Skipping {url}: not enough textual content")
        return None, discovered_links
    return text, discovered_links


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



# --- On-demand real-time page fetch and update ---
def scrape_and_update(url: str) -> str | None:
    """
    Scrape a single page, save its cleaned text, and return the filename if successful.
    """
    text, _ = fetch_page(url)
    if text:
        filename = slugify_url(url)
        save_text(filename, text)
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
