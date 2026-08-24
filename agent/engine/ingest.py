# Copyright (C) 2026 Teodor Smith
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Autonomous Ingestor (Subsystem 2 & Section 9.1).
Handles searching and extracting text from web and academic sources.
"""
from __future__ import annotations

import logging
import requests
import trafilatura
from bs4 import BeautifulSoup
from ddgs import DDGS

logger = logging.getLogger(__name__)

class IngestionAbortError(Exception):
    """Raised when extracted text does not meet the minimum requirements (Mitigation #4)."""
    pass

def search_sources(query: str, max_results: int = 3) -> list[str]:
    """Dispatch query to DuckDuckGo and return a list of URLs."""
    logger.info(f"Searching web for: '{query}'")
    try:
        urls = []
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=max_results)
            for r in results:
                urls.append(r["href"])
        return urls
    except Exception as e:
        logger.warning(f"Search failed for '{query}': {e}")
        return []

def extract_clean_text(url: str) -> str:
    """
    Fetch text using a fallback chain:
    1. Jina Reader (r.jina.ai/<url>)
    2. Direct HTTP + Trafilatura
    3. BeautifulSoup fallback
    
    Raises IngestionAbortError if the resulting text is < 100 characters.
    """
    text = ""
    
    # Attempt 1: Jina Reader
    jina_url = f"https://r.jina.ai/{url}"
    try:
        resp = requests.get(jina_url, timeout=10)
        if resp.status_code == 200:
            text = resp.text.strip()
    except Exception as e:
        logger.debug(f"Jina fetch failed for {url}: {e}")

    # Attempt 2: Trafilatura
    if not text or len(text) < 100:
        try:
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                extracted = trafilatura.extract(downloaded)
                if extracted:
                    text = extracted.strip()
        except Exception as e:
            logger.debug(f"Trafilatura extraction failed for {url}: {e}")

    # Attempt 3: BeautifulSoup fallback
    if not text or len(text) < 100:
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.content, "html.parser")
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.extract()
                extracted = soup.get_text(separator="\n")
                # Collapse multiple newlines
                text = "\n".join(line.strip() for line in extracted.splitlines() if line.strip())
        except Exception as e:
            logger.debug(f"BeautifulSoup fallback failed for {url}: {e}")

    # Mitigation #4: Abort Guard
    if len(text) < 100:
        raise IngestionAbortError(f"Extracted text from {url} is < 100 characters ({len(text)} chars). Aborting.")
        
    return text
