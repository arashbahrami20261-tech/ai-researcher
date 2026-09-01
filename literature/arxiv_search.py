"""
Literature search — arXiv backend.

Why arXiv first: it's free, needs no API key, and covers almost all AI/ML
research. This keeps the "Literature Engine" from the spec unblocked while
signup-gated sources (Semantic Scholar with a key, publisher APIs, etc.)
can be added later as additional backends behind the same `search_papers`
shape, without changing any calling code.
"""

from __future__ import annotations

from dataclasses import dataclass
from xml.etree import ElementTree

import requests

ARXIV_API_URL = "http://export.arxiv.org/api/query"

# arXiv's Atom feed uses these XML namespaces; declaring them once here
# keeps the parsing code below readable instead of repeating full URLs.
_NAMESPACES = {
    "atom": "http://www.w3.org/2005/Atom",
}


@dataclass
class Paper:
    """
    One search result, with provenance kept explicit (per the spec's rule
    that every source must retain where it came from).
    """

    title: str
    authors: list[str]
    abstract: str
    url: str
    published: str
    source: str = "arxiv"


def search_papers(
    query: str,
    max_results: int = 10,
    sort_by_newest: bool = True,
    min_year: int | None = None,
) -> list[Paper]:
    """
    Search arXiv for `query` and return up to `max_results` papers.

    `sort_by_newest`: ask arXiv to sort results by submission date instead
    of its default relevance ranking. For a research agent, "what's the
    latest work on this" is usually more useful than raw keyword relevance.

    `min_year`: if given, drop any paper published before this year. This
    filters client-side (arXiv's API doesn't support a year cutoff
    directly), so we over-fetch a bit internally to still return close to
    `max_results` papers after filtering.

    Results are de-duplicated by arXiv ID (the numeric part of the URL),
    since the same paper can otherwise appear twice across revisions.

    Raises `requests.HTTPError` if the arXiv API itself fails — callers
    should not silently swallow that, since a failed search should never
    be reported to the research loop as "zero relevant papers found".
    """
    # Over-fetch when a year filter is active, since some results will be
    # dropped after the fact. Capped at 50 to keep the request reasonable.
    fetch_count = min(max_results * 3, 50) if min_year else max_results

    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": fetch_count,
    }
    if sort_by_newest:
        params["sortBy"] = "submittedDate"
        params["sortOrder"] = "descending"

    response = requests.get(ARXIV_API_URL, params=params, timeout=15)
    response.raise_for_status()

    root = ElementTree.fromstring(response.text)
    papers: list[Paper] = []
    seen_ids: set[str] = set()

    for entry in root.findall("atom:entry", _NAMESPACES):
        title = entry.findtext("atom:title", default="", namespaces=_NAMESPACES).strip()
        abstract = entry.findtext("atom:summary", default="", namespaces=_NAMESPACES).strip()
        url = entry.findtext("atom:id", default="", namespaces=_NAMESPACES).strip()
        published = entry.findtext("atom:published", default="", namespaces=_NAMESPACES).strip()
        authors = [
            author.findtext("atom:name", default="", namespaces=_NAMESPACES)
            for author in entry.findall("atom:author", _NAMESPACES)
        ]

        # arXiv IDs look like ".../abs/2602.02159v1" — strip the version
        # suffix so v1 and v2 of the same paper count as one duplicate,
        # not two separate results.
        arxiv_id = url.rsplit("/", 1)[-1].split("v")[0]
        if arxiv_id in seen_ids:
            continue
        seen_ids.add(arxiv_id)

        if min_year and published[:4].isdigit() and int(published[:4]) < min_year:
            continue

        papers.append(
            Paper(
                title=title,
                authors=authors,
                abstract=abstract,
                url=url,
                published=published,
            )
        )

        if len(papers) >= max_results:
            break

    return papers
