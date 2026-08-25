import unittest
from unittest.mock import MagicMock, patch

from agent.engine.ingest import IngestionAbortError, extract_clean_text
from agent.engine.search import search_sources


class TestIngest(unittest.TestCase):
    @patch("agent.engine.search.DDGS")
    def test_search_sources(self, mock_ddgs):
        mock_instance = MagicMock()
        mock_ddgs.return_value.__enter__.return_value = mock_instance
        mock_instance.text.return_value = [
            {"href": "http://test.com/1"},
            {"href": "http://test.com/2"},
        ]

        urls = search_sources("test query", max_results=2)
        self.assertEqual(urls, ["http://test.com/1", "http://test.com/2"])

    @patch("agent.engine.ingest.httpx.Client")
    def test_extract_clean_text_jina_success(self, mock_client_cls):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "A" * 150  # Must be > 100 chars
        
        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp
        mock_client_cls.return_value.__enter__.return_value = mock_client

        text = extract_clean_text("http://test.com")
        self.assertEqual(text, "A" * 150)

    @patch("agent.engine.ingest.httpx.Client")
    def test_extract_clean_text_abort_guard(self, mock_client_cls):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "Too short"
        
        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp
        mock_client_cls.return_value.__enter__.return_value = mock_client

        with self.assertRaises(IngestionAbortError):
            extract_clean_text("http://test.com")


if __name__ == "__main__":
    unittest.main()
