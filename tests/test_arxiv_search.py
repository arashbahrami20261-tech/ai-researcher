"""
Tests for literature/arxiv_search.py.

These tests mock the HTTP call to arXiv (via `unittest.mock`) instead of
hitting the real API. That means they run instantly, work offline, and
never fail because arXiv happened to be slow or down — exactly what you
want from a unit test. A separate, optional test further down does hit
the real API, for when you want to confirm the live integration still
works.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from literature.arxiv_search import search_papers

# A minimal but realistic arXiv Atom feed with two entries, one of them a
# second version (v2) of the other — used to test de-duplication.
FAKE_ARXIV_RESPONSE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2602.02159v1</id>
    <title>Focus-dLLM: Accelerating Long-Context Diffusion LLM Inference</title>
    <summary>We propose a method for accelerating inference.</summary>
    <published>2026-02-15T00:00:00Z</published>
    <author><name>Lingkun Long</name></author>
    <author><name>Yushi Huang</name></author>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2602.02159v2</id>
    <title>Focus-dLLM: Accelerating Long-Context Diffusion LLM Inference (v2)</title>
    <summary>We propose a method for accelerating inference.</summary>
    <published>2026-02-20T00:00:00Z</published>
    <author><name>Lingkun Long</name></author>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2019.99999v1</id>
    <title>An Older Paper on the Same Topic</title>
    <summary>An earlier approach to the same problem.</summary>
    <published>2019-05-01T00:00:00Z</published>
    <author><name>Someone Else</name></author>
  </entry>
</feed>
"""


def _mock_response():
    mock_resp = Mock()
    mock_resp.text = FAKE_ARXIV_RESPONSE
    mock_resp.raise_for_status = Mock()
    return mock_resp


@patch("literature.arxiv_search.requests.get")
def test_search_papers_parses_results(mock_get):
    mock_get.return_value = _mock_response()

    papers = search_papers("long context transformers", max_results=10)

    # Three entries in the feed, but two share the same arXiv ID
    # (2602.02159, versions v1/v2) so de-duplication should leave 2.
    assert len(papers) == 2
    assert papers[0].title.startswith("Focus-dLLM")
    assert "Lingkun Long" in papers[0].authors
    assert papers[0].source == "arxiv"


@patch("literature.arxiv_search.requests.get")
def test_search_papers_deduplicates_versions(mock_get):
    mock_get.return_value = _mock_response()

    papers = search_papers("test query", max_results=10)
    urls = [p.url for p in papers]

    # Only the first-seen version of 2602.02159 should be kept.
    assert sum("2602.02159" in u for u in urls) == 1


@patch("literature.arxiv_search.requests.get")
def test_search_papers_min_year_filters_old_results(mock_get):
    mock_get.return_value = _mock_response()

    papers = search_papers("test query", max_results=10, min_year=2025)

    # The 2019 paper should be filtered out; only the 2026 one remains.
    assert len(papers) == 1
    assert all(p.published.startswith("2026") for p in papers)


@patch("literature.arxiv_search.requests.get")
def test_search_papers_respects_max_results(mock_get):
    mock_get.return_value = _mock_response()

    papers = search_papers("test query", max_results=1)

    assert len(papers) == 1


@pytest.mark.live
def test_search_papers_live_network_call():
    """
    Optional real-network test. Skipped by default (see pytest.ini /
    the -m flag) since it depends on arXiv being reachable. Run it
    explicitly with: pytest -m live
    """
    papers = search_papers("transformer", max_results=3)
    assert len(papers) > 0
    assert papers[0].url.startswith("http")
