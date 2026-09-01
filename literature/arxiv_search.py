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


def search_papers(query: str, max_results: int = 10) -> list[Paper]:
    """
    Search arXiv for `query` and return up to `max_results` papers.

    Raises `requests.HTTPError` if the arXiv API itself fails — callers
    should not silently swallow that, since a failed search should never
    be reported to the research loop as "zero relevant papers found".
    """
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
    }
    response = requests.get(ARXIV_API_URL, params=params, timeout=15)
    response.raise_for_status()

    root = ElementTree.fromstring(response.text)
    papers: list[Paper] = []

    for entry in root.findall("atom:entry", _NAMESPACES):
        title = entry.findtext("atom:title", default="", namespaces=_NAMESPACES).strip()
        abstract = entry.findtext("atom:summary", default="", namespaces=_NAMESPACES).strip()
        url = entry.findtext("atom:id", default="", namespaces=_NAMESPACES).strip()
        published = entry.findtext("atom:published", default="", namespaces=_NAMESPACES).strip()
        authors = [
            author.findtext("atom:name", default="", namespaces=_NAMESPACES)
            for author in entry.findall("atom:author", _NAMESPACES)
        ]

        papers.append(
            Paper(
                title=title,
                authors=authors,
                abstract=abstract,
                url=url,
                published=published,
            )
        )

    return papers
