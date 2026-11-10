from __future__ import annotations

import pytest

from relay.retrieval.citations import CitationError, render_claims, validate_citation_ids


@pytest.fixture
def passage():
    return {"chunk_id": "chunk-1", "paper_id": "1706.03762", "title": "Attention Is All You Need",
            "arxiv_version": "7", "pdf_page_number": 3, "chunk_text": "Self-attention relates positions."}


def test_valid_citation_resolves_to_real_page(passage):
    resolved = validate_citation_ids(["chunk-1"], [passage], {"1706.03762": 15})
    assert resolved[0]["pdf_page_number"] == 3


@pytest.mark.parametrize("citation", ["invented", ""])
def test_unknown_or_empty_citation_is_rejected(passage, citation):
    with pytest.raises(CitationError):
        validate_citation_ids([citation], [passage], {"1706.03762": 15})


def test_missing_or_out_of_bounds_page_is_rejected(passage):
    invalid = {**passage, "pdf_page_number": 16}
    with pytest.raises(CitationError):
        validate_citation_ids(["chunk-1"], [invalid], {"1706.03762": 15})


def test_missing_page_and_wrong_version_are_rejected(passage):
    invalid_page = {key: value for key, value in passage.items() if key != "pdf_page_number"}
    with pytest.raises(CitationError):
        validate_citation_ids(["chunk-1"], [invalid_page], {"1706.03762": 15})
    wrong_version = {**passage, "arxiv_version": "6"}
    with pytest.raises(CitationError):
        validate_citation_ids(["chunk-1"], [wrong_version], {"1706.03762": 15})


def test_citation_format_uses_source_metadata(passage):
    answer = render_claims([{"text": "Self-attention relates positions.", "citation_ids": ["chunk-1"]}],
                          [passage], {"1706.03762": 15})
    assert "arXiv:1706.03762v7" in answer
    assert "PDF p. 3" in answer
