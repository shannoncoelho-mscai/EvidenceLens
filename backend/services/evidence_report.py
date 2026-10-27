"""Structured evidence report construction.

Integrates NLI outputs with research retrieval and ranking results to produce
a transparent, traceable evidence report. Keeps lexical relevance scores
separate from NLI scores. Never presents a relevance score or NLI prediction
as scientific certainty.
"""

from __future__ import annotations

from typing import Any


def build_report(
    claim: str,
    search_status: str,
    publications: list[dict[str, Any]],
    nli_available: bool,
    publications_retrieved: int | None = None,
    nli_error: str | None = None,
) -> dict[str, Any]:
    """Build a structured evidence report."""
    supporting: list[dict[str, Any]] = []
    contradicting: list[dict[str, Any]] = []
    inconclusive: list[dict[str, Any]] = []

    for pub in publications:
        nli = pub.get("nli") or {}
        category = nli.get("category", "inconclusive")
        entry = _publication_summary(pub)

        if category == "supports":
            supporting.append(entry)
        elif category == "contradicts":
            contradicting.append(entry)
        else:
            inconclusive.append(entry)

    overall = _assess_overall(
        supporting, contradicting, inconclusive, search_status, nli_available
    )

    # Use provided retrieved count, or fall back to len(publications).
    retrieved_count = publications_retrieved if publications_retrieved is not None else len(publications)

    report: dict[str, Any] = {
        "claim": claim,
        "search_status": search_status,
        "publications_retrieved": retrieved_count,
        "publications_analyzed": len(publications),
        "results": publications,
        "evidence_summary": {
            "supports": supporting,
            "contradicts": contradicting,
            "inconclusive": inconclusive,
        },
        "nli_analysis": {
            "available": nli_available,
            "error": nli_error,
            "note": (
                "NLI predictions are model outputs, not calibrated scientific "
                "certainty. They do not verify a claim or establish causation."
            ),
        },
        "limitations": _build_limitations(
            search_status, nli_available, supporting, contradicting, inconclusive
        ),
        "overall_assessment": overall,
    }

    return report
def _publication_summary(pub: dict[str, Any]) -> dict[str, Any]:
    """Create a concise, safe summary of a publication for the report."""
    nli = pub.get("nli") or {}
    return {
        "title": pub.get("title"),
        "authors": pub.get("authorString"),
        "journal": pub.get("journalTitle"),
        "year": pub.get("pubYear"),
        "url": pub.get("url"),
        "doi": pub.get("doi"),
        "pmid": pub.get("pmid"),
        "pmcid": pub.get("pmcid"),
        "relevance_score": pub.get("relevance_score"),
        "relevance_note": pub.get("relevance_note"),
        "nli_label": nli.get("label"),
        "nli_category": nli.get("category"),
        "nli_scores": nli.get("scores"),
        "nli_raw_logits": nli.get("raw_logits"),
        "nli_error": nli.get("error"),
        "abstract": pub.get("abstractText"),
    }


def _assess_overall(
    supporting: list[dict[str, Any]],
    contradicting: list[dict[str, Any]],
    inconclusive: list[dict[str, Any]],
    search_status: str,
    nli_available: bool,
) -> dict[str, Any]:
    """Determine a cautious overall assessment.

    Never uses majority vote. Never declares a claim true. The assessment can
    explicitly be inconclusive when evidence is insufficient or conflicting.
    """
    total = len(supporting) + len(contradicting) + len(inconclusive)

    if search_status == "error":
        return {
            "conclusion": "inconclusive",
            "reason": "Research retrieval failed. No publications were analyzed.",
            "basis": "no_data",
        }

    if total == 0:
        return {
            "conclusion": "inconclusive",
            "reason": "No publications were found for this claim.",
            "basis": "no_results",
        }

    if not nli_available:
        return {
            "conclusion": "inconclusive",
            "reason": "NLI model is unavailable. Publications were retrieved "
            "but could not be classified.",
            "basis": "model_unavailable",
        }

    if total > 0 and len(supporting) == 0 and len(contradicting) == 0:
        return {
            "conclusion": "inconclusive",
            "reason": "All retrieved publications were rated inconclusive by "
            "the NLI model. No evidence supports or contradicts the claim.",
            "basis": "all_inconclusive",
        }

    if len(supporting) > 0 and len(contradicting) > 0:
        return {
            "conclusion": "inconclusive",
            "reason": "Retrieved publications contain conflicting NLI "
            "predictions (some support, some contradict). This does not "
            "establish scientific consensus or truth.",
            "basis": "conflicting",
        }

    if len(supporting) > 0 and len(contradicting) == 0:
        return {
            "conclusion": "inconclusive",
            "reason": (
                "Some retrieved publications were rated as supporting the "
                "claim by the NLI model. NLI predictions are not calibrated "
                "scientific certainty and do not verify the claim."
            ),
            "basis": "supports_only",
        }

    if len(contradicting) > 0 and len(supporting) == 0:
        return {
            "conclusion": "inconclusive",
            "reason": (
                "Some retrieved publications were rated as contradicting the "
                "claim by the NLI model. NLI predictions are not calibrated "
                "scientific certainty and do not disprove the claim."
            ),
            "basis": "contradicts_only",
        }

    return {
        "conclusion": "inconclusive",
        "reason": "Evidence assessment is inconclusive.",
        "basis": "unknown",
    }


def _build_limitations(
    search_status: str,
    nli_available: bool,
    supporting: list[dict[str, Any]],
    contradicting: list[dict[str, Any]],
    inconclusive: list[dict[str, Any]],
) -> list[str]:
    """Build a list of plain-language limitations for the report."""
    limitations: list[str] = [
        "TF-IDF relevance scores measure lexical overlap only -- not scientific truth or study quality.",
        "NLI predictions are model outputs, not calibrated scientific certainty.",
        "NLI does not establish causation, consensus, or whether a claim is true.",
        "Single abstracts are not sufficient proof of a scientific conclusion.",
        "Association does not imply causation.",
    ]

    if search_status == "empty":
        limitations.append(
            "No publications were found for this claim. The absence of results "
            "does not mean the claim is false -- it means no matching records "
            "were found in the searched database."
        )

    if search_status == "error":
        limitations.append(
            "Research retrieval failed. Results may be incomplete or unavailable."
        )

    if not nli_available:
        limitations.append(
            "NLI model is unavailable. Evidence classifications cannot be "
            "generated at this time."
        )

    if inconclusive and not supporting and not contradicting:
        limitations.append(
            "All retrieved publications were rated inconclusive. The NLI model "
            "did not find sufficient textual entailment in either direction."
        )

    if supporting and contradicting:
        limitations.append(
            "Retrieved publications contain conflicting NLI predictions. "
            "Conflicting evidence does not establish scientific consensus."
        )

    total = len(supporting) + len(contradicting) + len(inconclusive)
    if total > 0 and total < 5:
        limitations.append(
            f"Only {total} publication(s) were analyzed. A small number of "
            "sources limits the reliability of any overall assessment."
        )

    return limitations