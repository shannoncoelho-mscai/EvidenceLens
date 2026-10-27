"""Tests for NLI claim-evidence analysis service."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from services.claim_analysis import ClaimAnalyzer


class TestClaimAnalyzerEntailment:
    """Tests for entailment (supports) NLI output."""

    @patch("services.claim_analysis.AutoTokenizer")
    @patch("services.claim_analysis.AutoModelForSequenceClassification")
    def test_entailment_maps_to_supports(
        self, mock_model_class, mock_tokenizer_class
    ):
        """Entailment label should map to 'supports' evidence category."""
        import torch

        mock_model = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_model.eval.return_value = None

        mock_logits = torch.tensor([[-2.0, 3.0, -1.0]])
        mock_outputs = MagicMock()
        mock_outputs.logits = mock_logits
        mock_model.return_value = mock_outputs

        mock_tokenizer = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_tokenizer.return_value = {}

        ClaimAnalyzer._model = None
        ClaimAnalyzer._tokenizer = None
        ClaimAnalyzer._load_attempted = False
        ClaimAnalyzer._load_error = None

        result = ClaimAnalyzer.analyze_pair(
            "Coffee helps people live longer",
            "Higher coffee consumption was associated with lower mortality.",
        )

        assert result is not None
        assert result["label"] == "entailment"
        assert result["category"] == "supports"
        assert result["scores"]["entailment"] > result["scores"]["contradiction"]
        assert result["scores"]["entailment"] > result["scores"]["neutral"]


class TestClaimAnalyzerContradiction:
    """Tests for contradiction (contradicts) NLI output."""

    @patch("services.claim_analysis.AutoTokenizer")
    @patch("services.claim_analysis.AutoModelForSequenceClassification")
    def test_contradiction_maps_to_contradicts(
        self, mock_model_class, mock_tokenizer_class
    ):
        """Contradiction label should map to 'contradicts' evidence category."""
        import torch

        mock_model = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_model.eval.return_value = None

        mock_logits = torch.tensor([[3.0, -2.0, -1.0]])
        mock_outputs = MagicMock()
        mock_outputs.logits = mock_logits
        mock_model.return_value = mock_outputs

        mock_tokenizer = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_tokenizer.return_value = {}

        ClaimAnalyzer._model = None
        ClaimAnalyzer._tokenizer = None
        ClaimAnalyzer._load_attempted = False
        ClaimAnalyzer._load_error = None

        result = ClaimAnalyzer.analyze_pair(
            "Coffee prevents all disease",
            "Coffee consumption has no effect on disease prevention.",
        )

        assert result is not None
        assert result["label"] == "contradiction"
        assert result["category"] == "contradicts"


class TestClaimAnalyzerNeutral:
    """Tests for neutral (inconclusive) NLI output."""

    @patch("services.claim_analysis.AutoTokenizer")
    @patch("services.claim_analysis.AutoModelForSequenceClassification")
    def test_neutral_maps_to_inconclusive(
        self, mock_model_class, mock_tokenizer_class
    ):
        """Neutral label should map to 'inconclusive' evidence category."""
        import torch

        mock_model = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_model.eval.return_value = None

        mock_logits = torch.tensor([[-1.0, -1.0, 3.0]])
        mock_outputs = MagicMock()
        mock_outputs.logits = mock_logits
        mock_model.return_value = mock_outputs

        mock_tokenizer = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_tokenizer.return_value = {}

        ClaimAnalyzer._model = None
        ClaimAnalyzer._tokenizer = None
        ClaimAnalyzer._load_attempted = False
        ClaimAnalyzer._load_error = None

        result = ClaimAnalyzer.analyze_pair(
            "Coffee helps people live longer",
            "The weather was sunny today.",
        )

        assert result is not None
        assert result["label"] == "neutral"
        assert result["category"] == "inconclusive"


class TestClaimAnalyzerMissingAbstract:
    """Tests for handling missing or empty abstracts."""

    @patch("services.claim_analysis.ClaimAnalyzer._load_model")
    def test_publication_with_no_text_is_inconclusive(self, mock_load):
        """Publication with no title or abstract is inconclusive, not skipped."""
        publications = [
            {
                "id": "1",
                "title": None,
                "abstractText": None,
                "url": None,
            }
        ]

        ClaimAnalyzer._model = None
        ClaimAnalyzer._tokenizer = None

        results = ClaimAnalyzer.analyze_publications("any claim", publications)

        assert len(results) == 1
        assert results[0]["nli"]["category"] == "inconclusive"
        assert results[0]["nli"]["label"] is None
        assert results[0]["nli"]["scores"] is None
        assert "error" in results[0]["nli"]

    @patch("services.claim_analysis.ClaimAnalyzer._load_model")
    def test_publication_with_only_title_is_analyzed(self, mock_load):
        """Publication with only a title (no abstract) should still be analyzed."""
        import torch

        publications = [
            {
                "id": "1",
                "title": "Coffee and longevity study",
                "abstractText": None,
                "url": "https://example.com",
            }
        ]

        mock_model = MagicMock()
        mock_model.eval.return_value = None
        mock_logits = torch.tensor([[0.0, 2.0, 0.0]])
        mock_outputs = MagicMock()
        mock_outputs.logits = mock_logits
        mock_model.return_value = mock_outputs

        ClaimAnalyzer._model = mock_model
        ClaimAnalyzer._tokenizer = MagicMock()
        ClaimAnalyzer._tokenizer.return_value = {}

        results = ClaimAnalyzer.analyze_publications("any claim", publications)

        assert len(results) == 1
        assert results[0]["nli"]["category"] == "supports"


class TestClaimAnalyzerModelUnavailable:
    """Tests for handling unavailable NLI model."""

    @patch("services.claim_analysis.ClaimAnalyzer._load_model")
    def test_analyze_pair_returns_none_when_model_unavailable(self, mock_load):
        """analyze_pair returns None when model is not loaded."""
        ClaimAnalyzer._model = None
        ClaimAnalyzer._tokenizer = None

        result = ClaimAnalyzer.analyze_pair("claim", "evidence")
        assert result is None


class TestClaimAnalyzerConflictingPredictions:
    """Tests for conflicting NLI predictions across publications."""

    @patch("services.claim_analysis.ClaimAnalyzer._load_model")
    def test_conflicting_predictions_kept_separate(self, mock_load):
        """Supporting and contradicting publications are kept in separate lists."""
        import torch

        publications = [
            {
                "id": "1",
                "title": "Study supporting coffee",
                "abstractText": "Coffee helps longevity.",
                "url": "https://example.com/1",
            },
            {
                "id": "2",
                "title": "Study contradicting coffee",
                "abstractText": "Coffee has no effect on longevity.",
                "url": "https://example.com/2",
            },
        ]

        call_count = [0]

        def mock_analyze_pair(claim, evidence_text):
            call_count[0] += 1
            if call_count[0] == 1:
                return {
                    "label": "entailment",
                    "category": "supports",
                    "scores": {"contradiction": 0.1, "entailment": 0.8, "neutral": 0.1},
                    "raw_logits": [-2.0, 3.0, -1.0],
                    "evidence_text": evidence_text,
                }
            else:
                return {
                    "label": "contradiction",
                    "category": "contradicts",
                    "scores": {"contradiction": 0.8, "entailment": 0.1, "neutral": 0.1},
                    "raw_logits": [3.0, -2.0, -1.0],
                    "evidence_text": evidence_text,
                }

        ClaimAnalyzer._model = MagicMock()
        ClaimAnalyzer._tokenizer = MagicMock()
        ClaimAnalyzer.analyze_pair = staticmethod(mock_analyze_pair)

        results = ClaimAnalyzer.analyze_publications(
            "Coffee helps longevity", publications
        )

        assert len(results) == 2
        assert results[0]["nli"]["category"] == "supports"
        assert results[1]["nli"]["category"] == "contradicts"