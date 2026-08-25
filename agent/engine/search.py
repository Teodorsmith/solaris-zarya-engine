# Solaris Zarya Engine
# Copyright (C) 2026 Teodor Smith <teosmith.studios@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# For commercial licensing options without AGPLv3 network-copyleft obligations,
# contact: teosmith.studios@gmail.com

import logging
from urllib.parse import urlparse

from ddgs import DDGS

logger = logging.getLogger(__name__)

AUTHORITY_DOMAINS = {
    # Tier 1: Official Documentation & Standards (Score: 100)
    "docs.python.org": 100,
    "peps.python.org": 100,
    "docs.unity.com": 100,
    "docs.unity3d.com": 100,
    "docs.blender.org": 100,
    "developer.mozilla.org": 100,
    "en.wikipedia.org": 90,
    "github.com": 85,
    "arxiv.org": 85,
    # Tier 2: Reputable Technical Portals (Score: 60)
    "realpython.com": 60,
    "geeksforgeeks.org": 50,
    "stackoverflow.com": 50,
    "digitalocean.com": 50,
    # Tier 3: Deprioritized / Paywalled Blogs (Score: 10)
    "medium.com": 10,
}


def prioritize_sources(links: list[dict], max_sources: int = 3) -> list[str]:
    scored_links = []
    for item in links:
        url = item.get("href") or item.get("link") or ""
        if not url:
            continue
        domain = urlparse(url).netloc.lower()

        # Match against authority matrix
        score = 30
        for auth_domain, auth_score in AUTHORITY_DOMAINS.items():
            if auth_domain in domain:
                score = auth_score
                break

        # Penalize medium.com and paywalls
        if "medium.com" in domain:
            score = 5

        scored_links.append((score, url))

    # Sort by authority score descending
    scored_links.sort(key=lambda x: x[0], reverse=True)
    return [url for _, url in scored_links[:max_sources]]


def search_sources(query: str, max_results: int = 3) -> list[str]:
    """Dispatch query to DuckDuckGo and return a list of URLs."""
    logger.info(f"Searching web for: '{query}'")
    try:
        links = []
        with DDGS() as ddgs:
            # Get more results initially so we can filter and sort them
            results = ddgs.text(query, max_results=max_results * 3)
            for r in results:
                links.append(r)
        return prioritize_sources(links, max_sources=max_results)
    except Exception as e:
        logger.warning(f"Search failed for '{query}': {e}")
        return []
