"""Configuration for EvidenceLens backend.

All settings are read from environment variables with sensible defaults.
No secrets or API keys are stored here.
"""

from __future__ import annotations

import os
from typing import Final

# ---------------------------------------------------------------------------
# Flask / API settings
# ---------------------------------------------------------------------------
FLASK_HOST: Final[str] = os.getenv("EVIDENCELENS_HOST", "127.0.0.1")
FLASK_PORT: Final[int] = int(os.getenv("EVIDENCELENS_PORT", "5000"))
FLASK_DEBUG: Final[bool] = os.getenv("EVIDENCELENS_DEBUG", "false").lower() == "true"

# Frontend origin allowed for CORS. Keep narrow -- never "*".
FRONTEND_ORIGIN: Final[str] = os.getenv(
    "EVIDENCELENS_FRONTEND_ORIGIN", "http://127.0.0.1:5500"
)

# ---------------------------------------------------------------------------
# Claim validation
# ---------------------------------------------------------------------------
MAX_CLAIM_LENGTH: Final[int] = int(os.getenv("EVIDENCELENS_MAX_CLAIM_LENGTH", "500"))
MIN_CLAIM_LENGTH: Final[int] = 1

# ---------------------------------------------------------------------------
# Europe PMC research retrieval
# ---------------------------------------------------------------------------
EUROPE_PMC_BASE_URL: Final[str] = os.getenv(
    "EVIDENCELENS_EUROPE_PMC_URL",
    "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
)
EUROPE_PMC_TIMEOUT: Final[float] = float(os.getenv("EVIDENCELENS_EUROPE_PMC_TIMEOUT", "10"))
EUROPE_PMC_MAX_RESULTS: Final[int] = int(
    os.getenv("EVIDENCELENS_EUROPE_PMC_MAX_RESULTS", "10")
)
EUROPE_PMC_PAGE_SIZE: Final[int] = int(
    os.getenv("EVIDENCELENS_EUROPE_PMC_PAGE_SIZE", "10")
)

# ---------------------------------------------------------------------------
# Relevance ranking (TF-IDF + cosine similarity)
# ---------------------------------------------------------------------------
MAX_RANKED_RESULTS: Final[int] = int(os.getenv("EVIDENCELENS_MAX_RANKED_RESULTS", "10"))
TFIDF_MAX_FEATURES: Final[int] = int(os.getenv("EVIDENCELENS_TFIDF_MAX_FEATURES", "5000"))

# ---------------------------------------------------------------------------
# NLI model (Stage 3)
# ---------------------------------------------------------------------------
# Selected model: cross-encoder/nli-deberta-v3-small
# License: Apache-2.0 (verified from Hugging Face model card, tag "license:apache-2.0")
# Size: 0.1B parameters (~100 MB download)
# Labels (verified order from model card): contradiction, entailment, neutral
# Compatible with: AutoModelForSequenceClassification, transformers, sentence-transformers
# Performance: 91.65% accuracy on SNLI-test, 87.55% on MNLI mismatched
NLI_MODEL_ID: Final[str] = os.getenv(
    "EVIDENCELENS_NLI_MODEL_ID",
    "cross-encoder/nli-deberta-v3-small",
)
# Label order as returned by the model's logits (verified from model card).
NLI_LABELS: Final[list[str]] = ["contradiction", "entailment", "neutral"]
NLI_MODEL_LOADED: Final[bool] = False  # Updated to True in claim_analysis.py after load

# Evidence category mapping from NLI labels
# entailment  -> supports
# contradiction -> contradicts
# neutral     -> inconclusive
NLI_TO_EVIDENCE_CATEGORY: Final[dict[str, str]] = {
    "entailment": "supports",
    "contradiction": "contradicts",
    "neutral": "inconclusive",
}