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

logger = logging.getLogger(__name__)


class IngestionAbortError(Exception):
    """Raised when extracted text does not meet the minimum requirements (Mitigation #4)."""


import httpx

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

def fetch_page_markdown(url: str, timeout: float = 10.0) -> str | None:
    """Fetches a URL and returns clean extracted text/markdown, or None on failure."""
    # 1. Try Jina Reader first if configured
    try:
        jina_url = f"https://r.jina.ai/{url}"
        with httpx.Client(timeout=timeout, follow_redirects=True, headers=HEADERS) as client:
            resp = client.get(jina_url)
            if resp.status_code == 200 and isinstance(resp.text, str) and resp.text.strip():
                return resp.text
    except Exception as e:
        logger.debug(f"Jina fetch failed for {url}: {e}")

    # 2. Fallback to direct HTTP GET with browser User-Agent
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True, headers=HEADERS) as client:
            resp = client.get(url)
            if resp.status_code == 200 and isinstance(resp.text, str):
                # Extract clean text via BeautifulSoup
                soup = BeautifulSoup(resp.text, "html.parser")
                for tag in soup(["script", "style", "nav", "footer", "header"]):
                    tag.decompose()
                text = soup.get_text(separator="\n").strip()
                return text if text else None
    except Exception as e:
        logger.debug(f"BeautifulSoup fallback failed for {url}: {e}")

    return None

def extract_clean_text(url: str) -> str | None:
    """
    Fetch text using a fallback chain, enforcing string returns only.
    Raises IngestionAbortError if the resulting text is < 100 characters,
    or returns None if the fetch outright failed.
    """
    text = fetch_page_markdown(url)
    if not text or not isinstance(text, str):
        return None
        
    text = text.strip()
    
    # Mitigation #4: Abort Guard
    if len(text) < 100:
        raise IngestionAbortError(
            f"Extracted text from {url} is < 100 characters ({len(text)} chars). Aborting."
        )

    return text
