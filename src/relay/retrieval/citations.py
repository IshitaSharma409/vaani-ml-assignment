"""Citation rendering and source-of-truth validation."""

from __future__ import annotations

from typing import Any


class CitationError(ValueError):
    """Raised when an answer references anything outside retrieved evidence."""


EXPECTED_VERSIONS = {"1706.03762": "7", "2410.00037": "1", "2507.07955": "2"}


def validate_citation_ids(citation_ids: list[str], passages: list[dict[str, Any]], page_counts: dict[str, int]) -> list[dict[str, Any]]:
    """Resolve citation IDs only against retrieved chunks and verified PDF bounds."""
    available = {str(item["chunk_id"]): item for item in passages}
    if not citation_ids:
        raise CitationError("Every factual claim needs at least one citation")
    resolved = []
    for citation_id in citation_ids:
        passage = available.get(citation_id)
        if passage is None:
            raise CitationError(f"Citation does not identify a retrieved chunk: {citation_id}")
        paper_id = str(passage["paper_id"])
        if paper_id not in EXPECTED_VERSIONS or str(passage.get("arxiv_version")) != EXPECTED_VERSIONS[paper_id]:
            raise CitationError(f"Citation has an unknown or wrong paper version: {citation_id}")
        page = passage.get("pdf_page_number")
        if not isinstance(page, int) or page < 1 or page > page_counts.get(paper_id, 0):
            raise CitationError(f"Invalid PDF page for citation {citation_id}")
        if not passage.get("arxiv_version") or not passage.get("title"):
            raise CitationError(f"Citation source metadata is incomplete: {citation_id}")
        resolved.append(passage)
    return resolved


def render_claims(claims: list[dict[str, Any]], passages: list[dict[str, Any]], page_counts: dict[str, int]) -> str:
    """Render claim-level citations from trusted metadata, never model-supplied pages."""
    blocks = []
    for claim in claims:
        sources = validate_citation_ids(claim.get("citation_ids", []), passages, page_counts)
        refs = []
        seen = set()
        for source in sources:
            identity = source["chunk_id"]
            if identity not in seen:
                refs.append(
                    f"{source['title']} (arXiv:{source['paper_id']}v{source['arxiv_version']}, "
                    f"PDF p. {source['pdf_page_number']})"
                )
                seen.add(identity)
        text = str(claim.get("text", "")).strip()
        if not text:
            raise CitationError("Claim text is empty")
        blocks.append(f"{text} [{'; '.join(refs)}]")
    if not blocks:
        raise CitationError("No answer claims were provided")
    return "\n\n".join(blocks)
