"""Page-aware extraction of the fixed, versioned PDF library."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = PROJECT_ROOT / "data/papers/manifest.json"
PAPERS_DIR = PROJECT_ROOT / "data/papers/original"


def load_paper_pages(manifest_path: Path = MANIFEST_PATH) -> list[Document]:
    """Extract one LangChain document per PDF page, with stable source metadata."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    timestamp = datetime.now(timezone.utc).isoformat()
    pages: list[Document] = []
    for paper in manifest["papers"]:
        path = manifest_path.parent / "original" / paper["filename"]
        if not path.is_file():
            raise FileNotFoundError(f"Required source PDF is missing: {path}")
        reader = PdfReader(str(path))
        if len(reader.pages) != paper["page_count"]:
            raise ValueError(f"Page count changed for {paper['filename']}")
        for page_index, pdf_page in enumerate(reader.pages, start=1):
            content = pdf_page.extract_text(extraction_mode="plain") or ""
            content = re.sub(r"[\t\xa0 ]+", " ", content)
            content = re.sub(r"\n{3,}", "\n\n", content).strip()
            pages.append(
                Document(
                    page_content=content,
                    metadata={
                        "paper_id": paper["arxiv_id"],
                        "title": paper["title"],
                        "arxiv_version": paper["version"],
                        "pdf_page_number": page_index,
                        "source_filename": paper["filename"],
                        "extracted_at": timestamp,
                    },
                )
            )
    return pages
