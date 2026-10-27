"""NLI claim-evidence analysis service.

Loads a pretrained natural-language inference (NLI) transformer once per
application process and compares a user-submitted claim against each retrieved
publication's title and abstract.

IMPORTANT: NLI outputs are model predictions of textual entailment between two
texts. They are NOT calibrated scientific certainty, do not establish whether a
scientific claim is true, and do not infer causation, study quality, or
consensus. A prediction of "entailment" means the evidence text is consistent
with the claim -- not that the claim is scientifically verified.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import NLI_LABELS, NLI_MODEL_ID, NLI_TO_EVIDENCE_CATEGORY

logger = logging.getLogger(__name__)

# Label order as returned by the model's logits (verified from model card):
# ["contradiction", "entailment", "neutral"]
NLI_LABEL_ORDER: list[str] = NLI_LABELS

# Maximum sequence length for the tokenizer.
MAX_SEQ_LENGTH: int = 512

class NLIModelNotAvailableError(Exception):
    """Raised when the NLI model cannot be loaded or is unavailable."""


class ClaimAnalyzer:
    """NLI-based claim-evidence analyzer.

    Loads the model and tokenizer once on first use and caches them as class
    attributes so the model is not re-downloaded or re-loaded per request.
    """

    _model: AutoModelForSequenceClassification | None = None
    _tokenizer: AutoTokenizer | None = None
    _load_attempted: bool = False
    _load_error: str | None = None

    @classmethod
    def _load_model(cls) -> None:
        """Load the NLI model and tokenizer once per process."""
        if cls._load_attempted:
            return
        cls._load_attempted = True

        try:
            logger.info("Loading NLI model: %s", NLI_MODEL_ID)
            cls._tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_ID)
            cls._model = AutoModelForSequenceClassification.from_pretrained(
                NLI_MODEL_ID
            )
            cls._model.eval()
            logger.info("NLI model loaded successfully: %s", NLI_MODEL_ID)
        except Exception as exc:  # noqa: BLE001 - any load failure is surfaced
            cls._load_error = str(exc)
            cls._model = None
            cls._tokenizer = None
            logger.warning("NLI model failed to load: %s", exc)
    @classmethod
    def analyze_pair(
        cls, claim: str, evidence_text: str
    ) -> dict[str, Any] | None:
        """Run NLI inference on a single claim-evidence pair.

        Returns a dict with:
            label: predicted NLI label (contradiction/entailment/neutral)
            category: evidence category (supports/contradicts/inconclusive)
            scores: dict mapping each NLI label to its softmax probability
            raw_logits: list of raw logit values (for technical inspection)
            evidence_text: the evidence text that was compared

        Returns None if the model is unavailable or inference fails.
        """
        cls._load_model()
        if cls._model is None or cls._tokenizer is None:
            return None

        if not claim or not claim.strip():
            return None
        if not evidence_text or not evidence_text.strip():
            return None

        try:
            inputs = cls._tokenizer(
                claim.strip(),
                evidence_text.strip(),
                truncation=True,
                max_length=MAX_SEQ_LENGTH,
                padding=True,
                return_tensors="pt",
            )

            with torch.no_grad():
                outputs = cls._model(**inputs)
                logits = outputs.logits[0]
                probabilities = torch.softmax(logits, dim=-1)

            # Map logits to label scores using the verified label order.
            scores: dict[str, float] = {}
            for idx, label in enumerate(NLI_LABEL_ORDER):
                scores[label] = round(float(probabilities[idx].item()), 4)

            raw_logits: list[float] = [
                round(float(v), 4) for v in logits.tolist()
            ]

            predicted_idx = int(probabilities.argmax().item())
            predicted_label = NLI_LABEL_ORDER[predicted_idx]
            category = NLI_TO_EVIDENCE_CATEGORY.get(
                predicted_label, "inconclusive"
            )

            return {
                "label": predicted_label,
                "category": category,
                "scores": scores,
                "raw_logits": raw_logits,
                "evidence_text": evidence_text.strip(),
            }
        except Exception as exc:  # noqa: BLE001 - inference failure is surfaced
            logger.warning("NLI inference failed: %s", exc)
            return None

    @classmethod
    def analyze_publications(
        cls, claim: str, publications: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Analyze each publication against the claim using NLI.

        Publications with no usable text (no title and no abstract) are
        annotated as inconclusive with no model scores, rather than skipped
        or fabricated.

        Each returned publication dict is augmented with an ``nli`` field
        containing the analysis result, or None if analysis could not be
        performed.
        """
        cls._load_model()

        results: list[dict[str, Any]] = []
        for pub in publications:
            pub = dict(pub)  # avoid mutating caller's dict

            title = pub.get("title") or ""
            abstract = pub.get("abstractText") or ""

            # Build evidence text from title + abstract. Never fabricate.
            evidence_parts: list[str] = []
            if title.strip():
                evidence_parts.append(title.strip())
            if abstract.strip():
                evidence_parts.append(abstract.strip())
            evidence_text = " ".join(evidence_parts)

            if not evidence_text:
                pub["nli"] = {
                    "label": None,
                    "category": "inconclusive",
                    "scores": None,
                    "raw_logits": None,
                    "evidence_text": None,
                    "error": "No usable title or abstract text available for NLI analysis.",
                }
                results.append(pub)
                continue

            analysis = cls.analyze_pair(claim, evidence_text)
            pub["nli"] = analysis
            results.append(pub)

        return results

    @classmethod
    def is_available(cls) -> bool:
        """Return True if the model is loaded and ready for inference."""
        cls._load_model()
        return cls._model is not None and cls._tokenizer is not None

    @classmethod
    def load_error(cls) -> str | None:
        """Return the load error message, if any, without triggering load."""
        return cls._load_error

    @classmethod
    def is_loaded(cls) -> bool:
        """Return True if the model has been loaded, without triggering load."""
        return cls._model is not None and cls._tokenizer is not None