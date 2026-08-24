import unittest
from unittest.mock import patch, MagicMock
from agent.engine.ingest import extract_clean_text, search_sources, IngestionAbortError

class TestIngest(unittest.TestCase):
    @patch("agent.engine.ingest.DDGS")
    def test_search_sources(self, mock_ddgs):
        mock_instance = MagicMock()
        mock_ddgs.return_value.__enter__.return_value = mock_instance
        mock_instance.text.return_value = [
            {"href": "http://test.com/1"},
            {"href": "http://test.com/2"}
        ]
        
        urls = search_sources("test query", max_results=2)
        self.assertEqual(urls, ["http://test.com/1", "http://test.com/2"])

    @patch("agent.engine.ingest.requests.get")
    def test_extract_clean_text_jina_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "A" * 150  # Must be > 100 chars
        mock_get.return_value = mock_resp
        
        text = extract_clean_text("http://test.com")
        self.assertEqual(text, "A" * 150)
        
    @patch("agent.engine.ingest.requests.get")
    def test_extract_clean_text_abort_guard(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "Too short"
        mock_get.return_value = mock_resp
        
        with self.assertRaises(IngestionAbortError):
            extract_clean_text("http://test.com")

if __name__ == "__main__":
    unittest.main()
