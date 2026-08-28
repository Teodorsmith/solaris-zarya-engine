# Copyright (C) 2026 Teodor Smith
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Academic Ingestion Engine (Track 2).
Handles fetching and layout-aware PDF parsing for arXiv and Semantic Scholar.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from typing import Any

import httpx
from agent.models import Fact

logger = logging.getLogger(__name__)

try:
    import pymupdf
except ImportError:
    pymupdf = None
    logger.warning("pymupdf is not installed. Academic PDF ingestion will fail.")

# Constants
ARXIV_API_URL = "https://export.arxiv.org/api/query"
S2_API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
HTTP_TIMEOUT = 30.0


class AcademicIngester:
    """Client for fetching and parsing academic papers."""

    @staticmethod
    def _parse_arxiv_id(query: str) -> str | None:
        """Extracts an arXiv ID from a URL or raw ID string."""
        match = re.search(r"(\d{4}\.\d{4,5}(v\d+)?)", query)
        if match:
            return match.group(1)
        return None

    def fetch_arxiv_metadata(self, query: str) -> dict[str, Any] | None:
        """Fetches metadata and PDF URL from arXiv API."""
        arxiv_id = self._parse_arxiv_id(query)
        params = {}
        if arxiv_id:
            params["id_list"] = arxiv_id
        else:
            params["search_query"] = f"all:{query}"
            params["max_results"] = "1"

        try:
            with httpx.Client(
                timeout=HTTP_TIMEOUT,
                follow_redirects=True,
                headers={"User-Agent": "AutonomousAgent/1.0 (academic-researcher)"}
            ) as client:
                resp = client.get(ARXIV_API_URL, params=params)
                resp.raise_for_status()
                
            root = ET.fromstring(resp.text)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            entry = root.find("atom:entry", ns)
            if entry is None:
                if arxiv_id:
                    return {
                        "title": f"arXiv:{arxiv_id}",
                        "abstract": "",
                        "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                        "source_id": arxiv_id
                    }
                return None
                
            title = entry.find("atom:title", ns).text.replace("\n", " ").strip()
            summary = entry.find("atom:summary", ns).text.replace("\n", " ").strip()
            
            pdf_url = None
            for link in entry.findall("atom:link", ns):
                if link.attrib.get("title") == "pdf":
                    pdf_url = link.attrib.get("href")
            
            if not pdf_url and arxiv_id:
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                
            return {
                "title": title,
                "abstract": summary,
                "pdf_url": pdf_url,
                "source_id": arxiv_id or query
            }
        except Exception as e:
            logger.error(f"Failed to fetch arXiv metadata: {e}")
            if arxiv_id:
                logger.info(f"Falling back to direct arXiv PDF URL for ID: {arxiv_id}")
                return {
                    "title": f"arXiv:{arxiv_id}",
                    "abstract": "",
                    "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                    "source_id": arxiv_id
                }
            return None

    def fetch_s2_metadata(self, query: str) -> dict[str, Any] | None:
        """Fallback to Semantic Scholar Graph API."""
        params = {
            "query": query,
            "limit": 1,
            "fields": "title,abstract,openAccessPdf"
        }
        try:
            with httpx.Client(
                timeout=HTTP_TIMEOUT,
                follow_redirects=True,
                headers={"User-Agent": "AutonomousAgent/1.0 (academic-researcher)"}
            ) as client:
                resp = client.get(S2_API_URL, params=params)
                resp.raise_for_status()
                data = resp.json()
                
            if not data.get("data"):
                return None
                
            paper = data["data"][0]
            pdf_url = None
            if paper.get("openAccessPdf"):
                pdf_url = paper["openAccessPdf"].get("url")
                
            return {
                "title": paper.get("title"),
                "abstract": paper.get("abstract"),
                "pdf_url": pdf_url,
                "source_id": f"s2:{paper.get('paperId')}"
            }
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                logger.warning("Semantic Scholar rate limit (429) hit. Falling back.")
                return None
            logger.error(f"Failed to fetch Semantic Scholar metadata: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to fetch Semantic Scholar metadata: {e}")
            return None

    def download_pdf(self, url: str) -> bytes | None:
        """Downloads PDF bytes with a hard timeout."""
        try:
            with httpx.Client(
                timeout=HTTP_TIMEOUT, 
                follow_redirects=True,
                headers={"User-Agent": "AutonomousAgent/1.0 (academic-researcher)"}
            ) as client:
                resp = client.get(url)
                resp.raise_for_status()
                return resp.content
        except Exception as e:
            logger.error(f"Failed to download PDF from {url}: {e}")
            return None

    def parse_pdf_layout(self, pdf_bytes: bytes) -> tuple[str, list[dict]]:
        """
        Layout-aware parsing using pymupdf.
        Returns (Markdown string, discarded_blocks).
        """
        if not pymupdf:
            raise RuntimeError("pymupdf is not installed.")

        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        
        blocks = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_height = page.rect.height
            page_dict = page.get_text("dict")
            
            for b in page_dict.get("blocks", []):
                if b.get("type") == 0:  # Text block
                    bbox = b.get("bbox", [0, 0, 0, 0])
                    y0, y1 = bbox[1], bbox[3]
                    
                    # Extract font size
                    max_size = 0.0
                    text_content = ""
                    for line in b.get("lines", []):
                        for span in line.get("spans", []):
                            if span.get("size", 0) > max_size:
                                max_size = span.get("size")
                            text_content += span.get("text", "") + " "
                        text_content += "\n"
                        
                    text_content = text_content.strip()
                    if text_content:
                        blocks.append({
                            "text": text_content,
                            "size": max_size,
                            "y0": y0,
                            "y1": y1,
                            "page_height": page_height,
                            "page_num": page_num
                        })

        if not blocks:
            return "", []

        # Heuristics for headers/footers (top 5% or bottom 5%)
        valid_blocks = []
        discarded_blocks = []
        
        for b in blocks:
            top_margin = b["page_height"] * 0.05
            bottom_margin = b["page_height"] * 0.95
            
            if b["y0"] < top_margin or b["y1"] > bottom_margin:
                discarded_blocks.append(b)
            else:
                valid_blocks.append(b)
                
        if not valid_blocks:
            return "", discarded_blocks

        # Font size median calculation
        sizes = sorted([b["size"] for b in valid_blocks])
        median_size = sizes[len(sizes) // 2]
        
        # Check if font sizes are inconsistent (e.g. all same size)
        unique_sizes = set(sizes)
        
        markdown_output = []
        is_first_line = True
        
        for b in valid_blocks:
            text = b["text"]
            size = b["size"]
            
            is_header = False
            
            if len(unique_sizes) == 1:
                # Fallback 1: Only one font size
                is_header = False
            else:
                if is_first_line:
                    # Fallback 2: First line as title if sizes are inconsistent
                    is_header = True
                elif size > 1.4 * median_size:
                    is_header = True
                elif re.match(r"^\d+(\.\d+)*\s+[A-Z]", text):
                    # Never classify a header as Body if it contains a trailing number like 3.1
                    # Or rather: ALWAYS classify as header if it looks like a numbered section
                    is_header = True
                    
            if is_header:
                markdown_output.append(f"\n## {text}\n")
            else:
                markdown_output.append(text)
                
            is_first_line = False
            
        return "\n".join(markdown_output), discarded_blocks

    def ingest_paper(self, query: str) -> list[Fact]:
        """Orchestrates metadata fetch, PDF download, parsing, and Fact chunking."""
        if not pymupdf:
            logger.warning("pymupdf not installed. Aborting ingest_paper.")
            return []

        # 1. Fetch metadata
        meta = self.fetch_arxiv_metadata(query)
        if not meta:
            meta = self.fetch_s2_metadata(query)
            
        if not meta or not meta.get("pdf_url"):
            logger.error("Could not resolve PDF URL for query.")
            return []

        # 2. Download PDF
        pdf_bytes = self.download_pdf(meta["pdf_url"])
        if not pdf_bytes:
            return []
            
        # 3. Parse Layout
        try:
            markdown_content, discarded = self.parse_pdf_layout(pdf_bytes)
        except Exception as e:
            logger.error(f"Failed to parse PDF: {e}")
            return []
            
        # 4. Chunk into Facts (Simplified chunking by section)
        # In a full system, this would pass through LLM distillation. 
        # For now, we chunk by H2 headers.
        facts = []
        sections = re.split(r"\n##\s+", markdown_content)
        
        source_id = meta.get("source_id", query)
        title = meta.get("title", f"arXiv:{source_id}")
        
        # Add abstract as first fact
        if meta.get("abstract"):
            facts.append(Fact(
                topic=f"arxiv:{source_id}",
                text=f"Paper: {title} (arXiv:{source_id})\nAbstract: {meta['abstract']}",
                source_type="academic_ingestion",
                confidence=0.9,
            ))
            
        for idx, section in enumerate(sections):
            section = section.strip()
            if not section or len(section) < 50:
                continue
                
            facts.append(Fact(
                topic=f"arxiv:{source_id}",
                text=f"Paper: {title} (arXiv:{source_id})\nSection: {section[:4000]}",
                source_type="academic_ingestion",
                confidence=0.85,
            ))
            
        return facts
