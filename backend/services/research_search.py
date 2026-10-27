
"""Europe PMC research retrieval service."""

from __future__ import annotations

import logging
import re
import time
from typing import Any
from urllib.parse import quote

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import (
    EUROPE_PMC_BASE_URL,
    EUROPE_PMC_MAX_RESULTS,
    EUROPE_PMC_PAGE_SIZE,
    EUROPE_PMC_TIMEOUT,
)

logger = logging.getLogger(__name__)

# Common words that usually contribute little to literature searches.
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being",
    "by", "can", "could", "did", "do", "does", "for", "from",
    "had", "has", "have", "how", "i", "in", "into", "is", "it",
    "its", "may", "might", "more", "most", "of", "on", "or",
    "our", "people", "should", "that", "the", "their", "them",
    "there", "these", "they", "this", "those", "to", "was",
    "were", "what", "when", "which", "who", "why", "will",
    "with", "would", "you", "your", "always", "never",
}

MIN_TERM_LENGTH = 3


class ResearchSearchError(Exception):
    """Raised when the research retrieval service fails."""


class EuropePMCClient:
    """Client for the Europe PMC REST search API."""

    def __init__(
        self,
        base_url: str = EUROPE_PMC_BASE_URL,
        timeout: float = EUROPE_PMC_TIMEOUT,
        page_size: int = EUROPE_PMC_PAGE_SIZE,
        max_results: int = EUROPE_PMC_MAX_RESULTS,
    ) -> None:
        self.base_url = base_url
        self.timeout = timeout
        self.page_size = max(1, page_size)
        self.max_results = max(1, max_results)

        self.session = requests.Session()
# Configure retry strategy for transient Europe PMC errors (429, 500-503)
        retry_strategy = Retry(
            total=3,  # maximum 3 retry attempts
            backoff_factor=0.5,  # exponential backoff: 0.5s, 1s, 2s
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.session.headers.update(
            {
                "User-Agent": (
                    "EvidenceLens/1.0 "
                    "(educational scientific literature research tool)"
                ),
                "Accept": "application/json",
            }
        )

    @staticmethod
    def _extract_terms(query: str) -> list[str]:
        """Extract useful search terms from a natural-language claim."""
        words = re.findall(r"[A-Za-z0-9]+", query.lower())

        terms: list[str] = []
        seen: set[str] = set()

        for word in words:
            if (
                len(word) < MIN_TERM_LENGTH
                or word in STOP_WORDS
                or word in seen
                or word.isdigit()
            ):
                continue

            seen.add(word)
            terms.append(word)

        return terms

    @staticmethod
    def _build_search_query(query: str) -> str:
        """Build a title-and-abstract-focused Europe PMC query.

        Meaningful claim terms are searched in publication titles and
        abstracts rather than sending the entire sentence as a search query.
        """
        terms = EuropePMCClient._extract_terms(query)

        if not terms:
            return query.strip()

        # OR allows relevant publications to match one or more key concepts.
        # Europe PMC defaults to searching titles and abstracts;
        # explicit field syntax (TITLE_ABS:) has been observed to cause HTTP 503 errors.
        expressions = [term for term in terms[:8]]

        return " OR ".join(expressions)

    def search(self, query: str) -> dict[str, Any]:
        """Search Europe PMC and return normalized publication records.

        Returns:
            status: "ok", "error", or "empty"
            count: number of records returned
            results: normalized publication dictionaries
            error: error message when the search fails
        """
        if not query or not query.strip():
            return {
                "status": "error",
                "count": 0,
                "results": [],
                "error": "Search query must not be empty.",
            }

        cleaned_query = query.strip()
        search_query = self._build_search_query(cleaned_query)

        page_size = min(self.page_size, self.max_results)

        params: dict[str, str | int] = {
            "query": search_query,
            "format": "json",
            "pageSize": page_size,
            "resultType": "core",
            "sort": "RELEVANCE",
        }

        try:
            logger.info(
                "Querying Europe PMC with focused query: %s",
                search_query[:300],
            )

            response = self.session.get(
                self.base_url,
                params=params,
                timeout=self.timeout,
            )
            response.raise_for_status()

        except requests.Timeout as exc:
            logger.warning("Europe PMC request timed out: %s", exc)
            return {
                "status": "error",
                "count": 0,
                "results": [],
                "error": (
                    "The research database request timed out. "
                    "Please try again later."
                ),
            }

        except requests.ConnectionError as exc:
            logger.warning("Europe PMC connection error: %s", exc)
            return {
                "status": "error",
                "count": 0,
                "results": [],
                "error": (
                    "Unable to connect to the research database. "
                    "Please check your network connection."
                ),
            }

        except requests.HTTPError as exc:
            status_code = (
                exc.response.status_code
                if exc.response is not None
                else "unknown"
            )
            logger.warning(
                "Europe PMC HTTP error %s: %s",
                status_code,
                exc,
            )
            return {
                "status": "error",
                "count": 0,
                "results": [],
                "error": (
                    "The research database returned an error "
                    f"(HTTP {status_code})."
                ),
            }

        except requests.RequestException as exc:
            logger.warning("Europe PMC request failed: %s", exc)
            return {
                "status": "error",
                "count": 0,
                "results": [],
                "error": (
                    "An unexpected error occurred while contacting "
                    "the research database."
                ),
            }

        try:
            data = response.json()
        except ValueError as exc:
            logger.warning("Europe PMC returned invalid JSON: %s", exc)
            return {
                "status": "error",
                "count": 0,
                "results": [],
                "error": (
                    "The research database returned an invalid response."
                ),
            }

        result_list = data.get("resultList", {}).get("result", [])

        if not result_list:
            logger.info(
                "Europe PMC returned no results for query: %s",
                search_query[:200],
            )
            return {
                "status": "empty",
                "count": 0,
                "results": [],
                "error": None,
            }

        records = [
            self._normalize_record(item)
            for item in result_list
        ]

        # Rank returned records by overlap between the claim's meaningful
        # terms and each publication's title and abstract. This is a simple
        # lexical ranking signal, not a measure of scientific evidence.
        terms = self._extract_terms(cleaned_query)
        records.sort(
            key=lambda record: self._local_relevance(record, terms),
            reverse=True,
        )

        records = records[: self.max_results]

        logger.info(
            "Europe PMC returned %d publication(s) for the claim",
            len(records),
        )

        return {
            "status": "ok",
            "count": len(records),
            "results": records,
            "error": None,
        }

    @staticmethod
    def _local_relevance(
        record: dict[str, Any],
        terms: list[str],
    ) -> float:
        """Estimate lexical overlap with a publication's title and abstract."""
        title = str(record.get("title") or "").lower()
        abstract = str(record.get("abstractText") or "").lower()

        if not terms:
            return 0.0

        title_matches = sum(
            1 for term in terms if term in title
        )
        abstract_matches = sum(
            1 for term in terms if term in abstract
        )

        # Title matches receive more weight than abstract matches.
        return (
            (2.0 * title_matches) + abstract_matches
        ) / (3.0 * len(terms))

    @staticmethod
    def _normalize_record(
        item: dict[str, Any],
    ) -> dict[str, Any]:
        """Normalize a Europe PMC record without inventing metadata."""
        pmid = item.get("pmid")
        pmcid = item.get("pmcid")
        doi = item.get("doi")

        url: str | None = None

        if pmcid:
            url = (
                "https://europepmc.org/articles/"
                f"{quote(str(pmcid), safe='')}"
            )
        elif pmid:
            url = (
                "https://pubmed.ncbi.nlm.nih.gov/"
                f"{quote(str(pmid), safe='')}/"
            )
        elif doi:
            url = f"https://doi.org/{quote(str(doi), safe='/')}"

        abstract = item.get("abstractText")

        if abstract and len(abstract) > 2000:
            abstract = abstract[:2000].rstrip() + "..."

        return {
            "id": item.get("id"),
            "source": item.get("source"),
            "title": item.get("title"),
            "authorString": item.get("authorString"),
            "journalTitle": item.get("journalTitle"),
            "pubYear": item.get("pubYear"),
            "abstractText": abstract,
            "doi": doi,
            "pmid": pmid,
            "pmcid": pmcid,
            "url": url,
        }


def search_publications(query: str) -> dict[str, Any]:
    """Search Europe PMC using the default project configuration."""
    client = EuropePMCClient()
    return client.search(query)
