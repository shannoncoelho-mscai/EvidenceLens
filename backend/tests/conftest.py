"""Pytest fixtures for EvidenceLens backend tests."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

# Ensure the backend/ directory is on sys.path so `app` and `services` are importable
# when pytest is invoked from the project root.
_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

import pytest

from app import create_app


@pytest.fixture
def app_client():
    """Flask test client."""
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def mock_search_ok():
    """Mock a successful Europe PMC search returning two records."""
    with patch(
        "services.research_search.EuropePMCClient.search"
    ) as mock:
        mock.return_value = {
            "status": "ok",
            "count": 2,
            "error": None,
            "results": [
                {
                    "id": "12345",
                    "source": "MED",
                    "title": "Coffee consumption and longevity: a cohort study",
                    "authorString": "Smith J, Doe A",
                    "journalTitle": "Journal of Nutrition",
                    "pubYear": "2022",
                    "abstractText": (
                        "We examined coffee drinking and mortality in a large cohort. "
                        "Higher coffee consumption was associated with lower mortality."
                    ),
                    "doi": "10.1000/example.2022.12345",
                    "pmid": "350000001",
                    "pmcid": None,
                    "url": "https://pubmed.ncbi.nlm.nih.gov/350000001/",
                },
                {
                    "id": "67890",
                    "source": "MED",
                    "title": "Tea drinking and health outcomes",
                    "authorString": "Lee S, Kim H",
                    "journalTitle": "British Medical Journal",
                    "pubYear": "2021",
                    "abstractText": (
                        "Tea consumption was studied in relation to cardiovascular health."
                    ),
                    "doi": None,
                    "pmid": "350000002",
                    "pmcid": None,
                    "url": "https://pubmed.ncbi.nlm.nih.gov/350000002/",
                },
            ],
        }
        yield mock


@pytest.fixture
def mock_search_empty():
    """Mock an empty Europe PMC search result."""
    with patch(
        "services.research_search.EuropePMCClient.search"
    ) as mock:
        mock.return_value = {
            "status": "empty",
            "count": 0,
            "error": None,
            "results": [],
        }
        yield mock


@pytest.fixture
def mock_search_error():
    """Mock a Europe PMC search failure."""
    with patch(
        "services.research_search.EuropePMCClient.search"
    ) as mock:
        mock.return_value = {
            "status": "error",
            "count": 0,
            "error": "The research database returned an error (HTTP 500).",
            "results": [],
        }
        yield mock


@pytest.fixture
def mock_nli_available():
    """Mock ClaimAnalyzer as available with fake NLI results."""
    with patch("app.ClaimAnalyzer") as mock_class:
        instance = mock_class
        instance.is_available.return_value = True
        instance.load_error.return_value = None

        def _analyze_publications(claim, publications):
            """Return publications annotated with fake NLI results."""
            annotated = []
            for pub in publications:
                pub = dict(pub)
                pub["nli"] = {
                    "label": "entailment",
                    "category": "supports",
                    "scores": {
                        "contradiction": 0.1,
                        "entailment": 0.8,
                        "neutral": 0.1,
                    },
                    "raw_logits": [-2.0, 3.0, -1.0],
                    "evidence_text": (pub.get("title", "") + " " + (pub.get("abstractText") or "")).strip(),
                    "error": None,
                }
                annotated.append(pub)
            return annotated

        instance.analyze_publications.side_effect = _analyze_publications
        yield instance


@pytest.fixture
def mock_nli_unavailable():
    """Mock ClaimAnalyzer as unavailable."""
    with patch("app.ClaimAnalyzer") as mock_class:
        instance = mock_class
        instance.is_available.return_value = False
        instance.load_error.return_value = "Model download failed"
        yield instance