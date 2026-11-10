"""Page-bounded chunking helpers."""

from __future__ import annotations

import hashlib
import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def stable_chunk_id(metadata: dict[str, object], index: int) -> str:
    """Return an ID derived only from paper/version/page/chunk position."""
    identity = f"{metadata['paper_id']}:{metadata['arxiv_version']}:{metadata['pdf_page_number']}:{index}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


def split_pages(pages: list[Document], chunk_size: int, overlap: int) -> list[Document]:
    """Split each page independently, retaining source page metadata."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > overlap >= 0")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    chunks: list[Document] = []
    for page in pages:
        normalized = re.sub(r"[\t\xa0 ]+", " ", page.page_content)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized).strip()
        if not normalized:
            continue
        parts = splitter.create_documents([normalized], metadatas=[dict(page.metadata)])
        for index, part in enumerate(parts):
            metadata = dict(part.metadata)
            metadata["chunk_index"] = index
            metadata["chunk_id"] = stable_chunk_id(metadata, index)
            metadata["chunk_start"] = metadata.pop("start_index", -1)
            chunks.append(Document(page_content=part.page_content, metadata=metadata))
    return chunks
