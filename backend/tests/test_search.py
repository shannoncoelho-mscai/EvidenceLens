"""Tests for Europe PMC research search service."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from services.research_search import EuropePMCClient, search_publications


class TestSearchNormalization:
    """Tests for record normalization (no network required)."""

    def test_normalize_record_with_all_fields(self):
        client = EuropePMCClient()
        raw = {
            "id": "abc123",
            "source": "MED",
            "title": "Test Study Title",
            "authorString": "Author One, Author Two",
            "journalTitle": "Test Journal",
            "pubYear": "2023",
            "abstractText": "This is an abstract about testing.",
            "doi": "10.1000/test.2023.abc",
            "pmid": "360000001",
            "pmcid": "PMC123456",
        }
        record = client._normalize_record(raw)
        assert record["title"] == "Test Study Title"
        assert record["authorString"] == "Author One, Author Two"
        assert record["pubYear"] == "2023"
        assert record["pmid"] == "360000001"
        assert record["pmcid"] == "PMC123456"
        assert record["doi"] == "10.1000/test.2023.abc"
        assert record["url"] is not None
        assert "PMC123456" in record["url"]

    def test_normalize_record_with_pmid_only_url(self):
        client = EuropePMCClient()
        raw = {"pmid": "360000002", "title": "PMID only study"}
        record = client._normalize_record(raw)
        assert record["url"] == "https://pubmed.ncbi.nlm.nih.gov/360000002/"

    def test_normalize_record_with_doi_only_url(self):
        client = EuropePMCClient()
        raw = {"doi": "10.1000/test.doi", "title": "DOI only study"}
        record = client._normalize_record(raw)
        assert record["url"] == "https://doi.org/10.1000/test.doi"

    def test_normalize_record_missing_fields_none(self):
        client = EuropePMCClient()
        raw = {"title": "Minimal study"}
        record = client._normalize_record(raw)
        assert record["title"] == "Minimal study"
        assert record["authorString"] is None
        assert record["abstractText"] is None
        assert record["url"] is None

    def test_normalize_record_truncates_long_abstract(self):
        client = EuropePMCClient()
        long_abstract = "x" * 2000
        raw = {"title": "Long abstract study", "abstractText": long_abstract}
        record = client._normalize_record(raw)
        assert record["abstractText"] is not None
        assert len(record["abstractText"]) <= 1003  # 1000 + "..."


class TestSearchErrors:
    """Tests for search error handling (mocked network)."""

    def test_empty_query_returns_error(self):
        client = EuropePMCClient()
        result = client.search("")
        assert result["status"] == "error"
        assert result["count"] == 0
        assert result["results"] == []
        assert "error" in result

    def test_whitespace_query_returns_error(self):
        client = EuropePMCClient()
        result = client.search("   ")
        assert result["status"] == "error"

    @patch("services.research_search.requests.Session.get")
    def test_network_timeout_returns_error(self, mock_get):
        import requests

        mock_get.side_effect = requests.Timeout("timed out")
        client = EuropePMCClient(timeout=1)
        result = client.search("coffee")
        assert result["status"] == "error"
        assert result["count"] == 0
        assert result["results"] == []
        # User-facing message should be friendly, not a raw exception repr.
        error_msg = result.get("error", "")
        assert "timed out" in error_msg.lower()
        assert "Timeout" not in error_msg  # no raw exception class name
        assert "traceback" not in error_msg.lower()

    @patch("services.research_search.requests.Session.get")
    def test_connection_error_returns_error(self, mock_get):
        import requests

        mock_get.side_effect = requests.ConnectionError("conn failed")
        client = EuropePMCClient()
        result = client.search("coffee")
        assert result["status"] == "error"
        assert result["count"] == 0

    def test_search_publications_convenience_function(self):
        """Convenience function should return a dict with expected keys."""
        # We can't easily mock this without patching the client,
        # but we can verify it returns a properly shaped dict on error.
        with patch(
            "services.research_search.EuropePMCClient.search"
        ) as mock:
            mock.return_value = {
                "status": "empty",
                "count": 0,
                "results": [],
                "error": None,
            }
            result = search_publications("anything")
            assert result["status"] == "empty"
            assert result["count"] == 0