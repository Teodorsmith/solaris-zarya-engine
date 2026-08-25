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

import pytest
from unittest.mock import patch, Mock
import pymupdf

from agent.engine.academic import AcademicIngester

@pytest.fixture
def synthetic_pdf():
    """Generates a synthetic PDF with headers, footers, titles, and body text."""
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842) # A4 size

    # Footer (bottom 5% is > 0.95 * 842 = 800)
    page.insert_text((50, 810), "Page 1 - Footer", fontsize=10)
    
    # Header (top 5% is < 0.05 * 842 = 42)
    page.insert_text((50, 30), "Running Header 2026", fontsize=10)

    # Title (large font)
    page.insert_text((50, 100), "A Synthetic Academic Paper", fontsize=24)

    # H1 Section
    page.insert_text((50, 200), "1. Introduction", fontsize=16)

    # Body
    page.insert_text((50, 250), "This is the body of the introduction. It uses standard font.", fontsize=11)
    
    # H2 Section
    page.insert_text((50, 350), "1.1 Methodology", fontsize=14)
    
    # Body
    page.insert_text((50, 400), "Here we describe the methods.", fontsize=11)

    return doc.write()

def test_parse_pdf_layout(synthetic_pdf):
    ingester = AcademicIngester()
    markdown, discarded = ingester.parse_pdf_layout(synthetic_pdf)
    
    assert "A Synthetic Academic Paper" in markdown
    assert "1. Introduction" in markdown
    assert "This is the body" in markdown
    
    # Check that headers/footers were discarded
    assert "Running Header" not in markdown
    assert "Page 1 - Footer" not in markdown
    assert any("Footer" in b["text"] for b in discarded)
    assert any("Header" in b["text"] for b in discarded)

@patch("agent.engine.academic.httpx.Client")
def test_fetch_arxiv_metadata(mock_httpx):
    mock_resp = Mock()
    mock_resp.text = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <title>Attention Is All You Need</title>
        <summary>The dominant sequence transduction models are based on complex recurrent or convolutional neural networks...</summary>
        <link title="pdf" href="http://arxiv.org/pdf/1706.03762v5" rel="related" type="application/pdf"/>
      </entry>
    </feed>"""
    
    mock_client_instance = mock_httpx.return_value.__enter__.return_value
    mock_client_instance.get.return_value = mock_resp
    
    ingester = AcademicIngester()
    meta = ingester.fetch_arxiv_metadata("1706.03762")
    
    assert meta is not None
    assert meta["title"] == "Attention Is All You Need"
    assert "dominant sequence transduction models" in meta["abstract"]
    assert meta["pdf_url"] == "http://arxiv.org/pdf/1706.03762v5"

