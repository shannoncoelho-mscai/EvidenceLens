
"""Europe PMC research retrieval service."""

from __future__ import annotations

import logging
import re
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

        retry_strategy = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        self.session.headers.update(
            {
                "User-Agent": "EvidenceLens/1.0",
                "Accept": "application/json",
            }
        )

    @staticmethod
    def _extract_terms(query: str) -> list[str]:
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
        terms = EuropePMCClient._extract_terms(query)

        if not terms:
            return query.strip()

        return " OR ".join(terms[:5])

    def search(self, query: str) -> dict[str, Any]:
        if not query or not query.strip():
            return {
                "status": "error",
                "count": 0,
                "results": [],
                "error": "Search query must not be empty.",
            }

        cleaned_query = query.strip()
        search_query = self._build_search_query(cleaned_query)

        params: dict[str, str | int] = {
            "query": search_query,
            "format": "json",
            "pageSize": min(self.page_size, self.max_results),
            "resultType": "lite",
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

        except requests.Timeout:
            logger.warning("Europe PMC request timed out.")
            return self._error_result(
                "The research database request timed out. Please try again."
            )

        except requests.ConnectionError as exc:
            logger.warning("Europe PMC connection error: %s", exc)
            return self._error_result(
                "Unable to connect to the research database. Check your network."
            )

        except requests.HTTPError as exc:
            status_code = (
                exc.response.status_code
                if exc.response is not None
                else "unknown"
            )
            logger.warning("Europe PMC HTTP error: %s", exc)
            return self._error_result(
                f"The research database returned HTTP {status_code}. "
                "Please try again later."
            )

        except requests.RequestException as exc:
            logger.warning("Europe PMC request failed: %s", exc)
            return self._error_result(
                "An unexpected error occurred while contacting the database."
            )

        try:
            data = response.json()
        except ValueError:
            logger.warning("Europe PMC returned invalid JSON.")
            return self._error_result(
                "The research database returned an invalid response."
            )

        result_list = data.get("resultList", {}).get("result", [])

        if not result_list:
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

        terms = self._extract_terms(cleaned_query)
        records.sort(
            key=lambda record: self._local_relevance(record, terms),
            reverse=True,
        )
        records = records[: self.max_results]

        logger.info("Europe PMC returned %d publication(s).", len(records))

        return {
            "status": "ok",
            "count": len(records),
            "results": records,
            "error": None,
        }

    @staticmethod
    def _error_result(message: str) -> dict[str, Any]:
        return {
            "status": "error",
            "count": 0,
            "results": [],
            "error": message,
        }

    @staticmethod
    def _local_relevance(
        record: dict[str, Any],
        terms: list[str],
    ) -> float:
        title = str(record.get("title") or "").lower()
        abstract = str(record.get("abstractText") or "").lower()

        if not terms:
            return 0.0

        title_matches = sum(term in title for term in terms)
        abstract_matches = sum(term in abstract for term in terms)

        return (
            (2.0 * title_matches) + abstract_matches
        ) / (3.0 * len(terms))

    @staticmethod
    def _normalize_record(item: dict[str, Any]) -> dict[str, Any]:
        pmid = item.get("pmid")
        pmcid = item.get("pmcid")
        doi = item.get("doi")

        url = None

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

        abstract = item.get("abstractText") or item.get("abstract")

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
