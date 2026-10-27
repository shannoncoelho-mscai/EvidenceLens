"""Tests for EvidenceLens API endpoints."""

from __future__ import annotations

import json
from typing import Any

import pytest


class TestHealthEndpoint:
    """Tests for GET /api/health."""

    def test_health_returns_ok(self, app_client):
        response = app_client.get("/api/health")
        assert response.status_code == 200

        data = response.get_json()
        assert data is not None
        assert data["status"] == "ok"
        assert data["service"] == "evidencelens-backend"
        assert "components" in data
        assert data["components"]["api"] == "running"


class TestModelInfoEndpoint:
    """Tests for GET /api/model-info."""

    def test_model_info_returns_configuration(self, app_client):
        response = app_client.get("/api/model-info")
        assert response.status_code == 200

        data = response.get_json()
        assert data is not None
        assert "model_id" in data
        assert "model_loaded" in data
        assert "labels" in data
        assert isinstance(data["model_loaded"], bool)
        assert isinstance(data["labels"], list)
        assert data["model_id"] == "cross-encoder/nli-deberta-v3-small"


class TestAnalyzeEndpoint:
    """Tests for POST /api/analyze."""

    def test_analyze_valid_claim_returns_report(
        self, app_client, mock_search_ok, mock_nli_available
    ):
        response = app_client.post(
            "/api/analyze",
            data=json.dumps({"claim": "Coffee helps people live longer"}),
            content_type="application/json",
        )
        assert response.status_code == 200

        data = response.get_json()
        assert data is not None
        assert data["claim"] == "Coffee helps people live longer"
        assert data["search_status"] == "ok"
        assert data["publications_retrieved"] == 2
        assert data["publications_analyzed"] == 2
        assert len(data["results"]) == 2
        assert data["nli_analysis"]["available"] is True
        assert data["overall_assessment"]["conclusion"] == "inconclusive"
        assert "limitations" in data
        assert "evidence_summary" in data

    def test_analyze_nli_supports_evidence(
        self, app_client, mock_search_ok, mock_nli_available
    ):
        """Entailment NLI result maps to 'supports' evidence category."""
        response = app_client.post(
            "/api/analyze",
            data=json.dumps({"claim": "Coffee helps people live longer"}),
            content_type="application/json",
        )
        assert response.status_code == 200

        data = response.get_json()
        assert data is not None
        # The mocked NLI returns entailment -> supports
        supports = data["evidence_summary"]["supports"]
        assert len(supports) >= 1
        assert supports[0]["nli_label"] == "entailment"
        assert supports[0]["nli_category"] == "supports"

    def test_analyze_relevance_scores_present(
        self, app_client, mock_search_ok, mock_nli_available
    ):
        """Relevance scores must be present on each result."""
        response = app_client.post(
            "/api/analyze",
            data=json.dumps({"claim": "coffee longevity"}),
            content_type="application/json",
        )
        assert response.status_code == 200
        data = response.get_json()
        assert data is not None
        for result in data["results"]:
            assert "relevance_score" in result
            assert "relevance_note" in result

    def test_analyze_missing_claim_returns_400(self, app_client):
        response = app_client.post(
            "/api/analyze",
            data=json.dumps({}),
            content_type="application/json",
        )
        assert response.status_code == 400
        data = response.get_json()
        assert data is not None
        assert "error" in data

    def test_analyze_empty_claim_returns_400(self, app_client):
        response = app_client.post(
            "/api/analyze",
            data=json.dumps({"claim": "   "}),
            content_type="application/json",
        )
        assert response.status_code == 400

    def test_analyze_missing_body_returns_400(self, app_client):
        response = app_client.post(
            "/api/analyze",
            data="not json",
            content_type="application/json",
        )
        assert response.status_code == 400

    def test_analyze_excessively_long_claim_returns_400(self, app_client):
        long_claim = "x" * 501  # exceeds MAX_CLAIM_LENGTH of 500
        response = app_client.post(
            "/api/analyze",
            data=json.dumps({"claim": long_claim}),
            content_type="application/json",
        )
        assert response.status_code == 400

    def test_analyze_claim_at_max_length_accepted(
        self, app_client, mock_search_ok, mock_nli_available
    ):
        claim = "x" * 500  # exactly MAX_CLAIM_LENGTH
        response = app_client.post(
            "/api/analyze",
            data=json.dumps({"claim": claim}),
            content_type="application/json",
        )
        assert response.status_code == 200

    def test_analyze_search_error_returns_502(
        self, app_client, mock_search_error, mock_nli_available
    ):
        response = app_client.post(
            "/api/analyze",
            data=json.dumps({"claim": "any claim"}),
            content_type="application/json",
        )
        assert response.status_code == 502
        data = response.get_json()
        assert data is not None
        assert "error" in data

    def test_analyze_empty_search_results(
        self, app_client, mock_search_empty, mock_nli_available
    ):
        response = app_client.post(
            "/api/analyze",
            data=json.dumps({"claim": "something very obscure xyz123"}),
            content_type="application/json",
        )
        assert response.status_code == 200
        data = response.get_json()
        assert data is not None
        assert data["search_status"] == "empty"
        assert data["publications_retrieved"] == 0
        assert data["publications_analyzed"] == 0
        assert data["results"] == []
        assert data["overall_assessment"]["conclusion"] == "inconclusive"

    def test_analyze_nli_unavailable_returns_inconclusive(
        self, app_client, mock_search_ok, mock_nli_unavailable
    ):
        """When NLI model is unavailable, report is inconclusive with error."""
        response = app_client.post(
            "/api/analyze",
            data=json.dumps({"claim": "Coffee helps people live longer"}),
            content_type="application/json",
        )
        assert response.status_code == 200

        data = response.get_json()
        assert data is not None
        assert data["nli_analysis"]["available"] is False
        assert data["nli_analysis"]["error"] is not None
        assert data["overall_assessment"]["conclusion"] == "inconclusive"
        assert "model is unavailable" in data["overall_assessment"]["reason"].lower()