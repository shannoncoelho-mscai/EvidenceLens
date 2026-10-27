"""Evidence relevance ranking service.

Uses TF-IDF vectorization with cosine similarity to rank retrieved publications
by their textual relevance to a user-submitted claim.

IMPORTANT: The similarity score measures lexical overlap between the claim and
the publication's title/abstract. It is a retrieval-relevance score ONLY. It
must never be presented as a measure of scientific truth, study quality, or
evidence strength. A high score means the publication's text is similar to the
claim -- not that the claim is correct.
"""

from __future__ import annotations

import logging
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class EvidenceRanker:
    """Rank publications by TF-IDF cosine similarity to a claim."""

    def __init__(self, max_features: int = 5000) -> None:
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            ngram_range=(1, 2),
        )

    def rank(
        self, claim: str, publications: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Rank ``publications`` by relevance to ``claim``.

        Each returned publication dict includes an additional
        ``relevance_score`` field (float between 0 and 1) and
        ``relevance_note`` explaining that this is not a truth score.

        Publications with no usable text (no title and no abstract) are
        assigned a score of 0.0 and retained at the end of the list.
        """
        if not publications:
            return []

        claim = claim.strip()
        documents: list[str] = []
        usable_indices: list[int] = []

        for idx, pub in enumerate(publications):
            text_parts: list[str] = []
            title = pub.get("title")
            abstract = pub.get("abstractText")
            if title:
                text_parts.append(str(title))
            if abstract:
                text_parts.append(str(abstract))

            if text_parts:
                documents.append(" ".join(text_parts))
                usable_indices.append(idx)
            else:
                documents.append("")

        if not usable_indices:
            # No publication has any text to compare against.
            for pub in publications:
                pub["relevance_score"] = 0.0
                pub["relevance_note"] = (
                    "No usable title or abstract text available for relevance scoring."
                )
            return publications

        # Fit TF-IDF on all documents, then compute cosine similarity vs claim.
        self.vectorizer.fit(documents)
        claim_vector = self.vectorizer.transform([claim])
        doc_vectors = self.vectorizer.transform(documents)

        similarities = cosine_similarity(claim_vector, doc_vectors).flatten()

        for idx, pub in enumerate(publications):
            score = float(similarities[idx])
            pub["relevance_score"] = round(score, 4)
            pub["relevance_note"] = (
                "Lexical relevance between claim and publication (TF-IDF cosine "
                "similarity). Not a measure of scientific truth or study quality."
            )

        # Sort by relevance score descending.
        ranked = sorted(publications, key=lambda p: p.get("relevance_score", 0.0), reverse=True)
        logger.info("Ranked %d publications by relevance to claim", len(ranked))
        return ranked


def rank_publications(
    claim: str, publications: list[dict[str, Any]], max_features: int = 5000
) -> list[dict[str, Any]]:
    """Convenience function to rank publications using default configuration."""
    ranker = EvidenceRanker(max_features=max_features)
    return ranker.rank(claim, publications)