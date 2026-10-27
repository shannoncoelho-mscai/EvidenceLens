
"""EvidenceLens Flask backend.

Thin REST API layer. Business logic lives in the services package.
Routes are intentionally minimal; all retrieval, ranking, and reporting logic
is delegated to service modules.
"""

from __future__ import annotations

import logging
from typing import Any

from flask import Flask, jsonify, request
from flask_cors import CORS

from config import (
    FLASK_DEBUG,
    FLASK_HOST,
    FLASK_PORT,
    FRONTEND_ORIGIN,
    MAX_CLAIM_LENGTH,
    MIN_CLAIM_LENGTH,
    NLI_LABELS,
    NLI_MODEL_ID,
)
from services.claim_analysis import ClaimAnalyzer
from services.evidence_report import build_report
from services.evidence_ranking import rank_publications
from services.research_search import search_publications

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

logger = logging.getLogger(__name__)

app = Flask(__name__)

# Allow the local frontend from either hostname.
# Keep the permitted origins restricted to the development frontend.
allowed_origins = [
    "http://localhost:5500",
    "http://127.0.0.1:5500",
]

# Also retain the configured frontend origin if it is set.
if FRONTEND_ORIGIN and FRONTEND_ORIGIN not in allowed_origins:
    allowed_origins.append(FRONTEND_ORIGIN)

CORS(
    app,
    resources={
        r"/api/*": {
            "origins": allowed_origins,
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type", "Accept"],
        }
    },
)


def _build_error_response(
    message: str,
    status_code: int,
) -> tuple[Any, int]:
    """Return a standardized JSON error response."""
    return jsonify(
        {
            "error": message,
            "status_code": status_code,
        }
    ), status_code


def _validate_claim(claim: Any) -> str | None:
    """Validate the submitted claim.

    Returns the cleaned claim string if valid, or None if invalid.
    """
    if claim is None or not isinstance(claim, str):
        return None

    cleaned = claim.strip()

    if len(cleaned) < MIN_CLAIM_LENGTH:
        return None

    if len(cleaned) > MAX_CLAIM_LENGTH:
        return None

    return cleaned


@app.get("/api/health")
def health() -> Any:
    """Report application and component status."""
    return jsonify(
        {
            "status": "ok",
            "service": "evidencelens-backend",
            "components": {
                "api": "running",
                "research_search": "configured",
                "evidence_ranking": "configured",
                "nli_model": "configured",
            },
        }
    )


@app.get("/api/model-info")
def model_info() -> Any:
    """Report the configured inference model and its load status."""
    nli_error = (
        ClaimAnalyzer.load_error()
        if ClaimAnalyzer._load_attempted
        else None
    )

    return jsonify(
        {
            "model_id": NLI_MODEL_ID,
            "model_loaded": ClaimAnalyzer._model is not None,
            "stage": "stage_3_nli_integrated",
            "note": (
                "NLI model integration is complete. "
                "Model is loaded lazily on the first /api/analyze request."
            ),
            "labels": NLI_LABELS,
            "load_error": nli_error,
            "limitations": (
                "MNLI-trained NLI models predict textual entailment between "
                "a claim and evidence text. They are not calibrated scientific "
                "truth verifiers and must not be used to prove or disprove "
                "scientific claims."
            ),
        }
    )


@app.post("/api/analyze")
def analyze() -> Any:
    """Analyze a scientific claim against retrieved research.

    Steps:
        1. Validate the input claim.
        2. Retrieve publications from Europe PMC.
        3. Rank publications by TF-IDF relevance.
        4. Analyze publications using the NLI model.
        5. Return a structured evidence report.

    NLI predictions are model outputs, not scientific truth.
    """
    data = request.get_json(silent=True)

    if data is None or not isinstance(data, dict):
        return _build_error_response(
            "Request body must be a JSON object with a 'claim' field.",
            400,
        )

    claim = data.get("claim")
    cleaned_claim = _validate_claim(claim)

    if cleaned_claim is None:
        return _build_error_response(
            f"Claim is required and must be between "
            f"{MIN_CLAIM_LENGTH} and {MAX_CLAIM_LENGTH} characters.",
            400,
        )

    # Step 1: Retrieve research.
    search_result = search_publications(cleaned_claim)

    if search_result["status"] == "error":
        return _build_error_response(
            search_result.get("error") or "Search failed.",
            502,
        )

    publications = search_result.get("results", [])
    search_status = search_result["status"]

    # Step 2: Rank publications by relevance.
    ranked = rank_publications(cleaned_claim, publications)

    # Step 3: Analyze evidence using the NLI model.
    nli_available = ClaimAnalyzer.is_available()

    nli_error = (
        ClaimAnalyzer.load_error()
        if not nli_available
        else None
    )

    if nli_available:
        analyzed = ClaimAnalyzer.analyze_publications(
            cleaned_claim,
            ranked,
        )
    else:
        # If the model is unavailable, do not fabricate classifications.
        analyzed = []

        for pub in ranked:
            pub = dict(pub)
            pub["nli"] = {
                "label": None,
                "category": "inconclusive",
                "scores": None,
                "raw_logits": None,
                "evidence_text": None,
                "error": nli_error or "NLI model is unavailable.",
            }
            analyzed.append(pub)

    # Step 4: Build the structured evidence report.
    report = build_report(
        claim=cleaned_claim,
        search_status=search_status,
        publications=analyzed,
        publications_retrieved=search_result.get("count", 0),
        nli_available=nli_available,
        nli_error=nli_error,
    )

    return jsonify(report), 200


def create_app() -> Flask:
    """Application factory for testing and WSGI deployment."""
    return app


if __name__ == "__main__":
    logger.info(
        "Starting EvidenceLens backend on %s:%d",
        FLASK_HOST,
        FLASK_PORT,
    )

    app.run(
        host=FLASK_HOST,
        port=FLASK_PORT,
        debug=FLASK_DEBUG,
    )
