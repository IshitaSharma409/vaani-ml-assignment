"""Download and verify only the three arXiv versions required by the brief."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "data" / "papers" / "original"
MANIFEST = ROOT / "data" / "papers" / "manifest.json"
PAPERS = (
    ("1706.03762", "7", "Attention Is All You Need", "https://arxiv.org/pdf/1706.03762v7"),
    (
        "2410.00037",
        "1",
        "Moshi: a speech-text foundation model for real-time dialogue",
        "https://arxiv.org/pdf/2410.00037v1",
    ),
    (
        "2507.07955",
        "2",
        "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling",
        "https://arxiv.org/pdf/2507.07955v2",
    ),
)


def _verify_pdf(path: Path, title: str) -> int:
    """Require a parseable PDF whose first page contains the expected title."""
    content = path.read_bytes()
    if not content.startswith(b"%PDF-"):
        raise ValueError(f"Downloaded file is not a PDF: {path.name}")
    reader = PdfReader(str(path))
    first_page = " ".join((reader.pages[0].extract_text() or "").split()).casefold()
    if " ".join(title.split()).casefold() not in first_page:
        raise ValueError(f"Expected title was not found in {path.name}")
    return len(reader.pages)


def main() -> None:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for arxiv_id, version, title, url in PAPERS:
        filename = f"{arxiv_id}v{version}.pdf"
        destination = PAPER_DIR / filename
        if not destination.exists():
            request = urllib.request.Request(
                url, headers={"User-Agent": "TheRelayInternshipProject/0.1"}
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read()
            temporary = destination.with_suffix(".pdf.part")
            temporary.write_bytes(payload)
            _verify_pdf(temporary, title)
            temporary.replace(destination)

        page_count = _verify_pdf(destination, title)
        rows.append(
            {
                "arxiv_id": arxiv_id,
                "version": version,
                "title": title,
                "filename": filename,
                "source_url": url,
                "downloaded_at_utc": datetime.fromtimestamp(
                    destination.stat().st_mtime, timezone.utc
                ).isoformat(timespec="seconds").replace("+00:00", "Z"),
                "verified_at_utc": datetime.now(timezone.utc)
                .isoformat(timespec="seconds")
                .replace("+00:00", "Z"),
                "page_count": page_count,
                "file_size_bytes": destination.stat().st_size,
                "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                "verification": "PDF parses and expected title appears on its first page",
            }
        )

    actual = sorted(path.name for path in PAPER_DIR.glob("*.pdf"))
    expected = sorted(f"{arxiv_id}v{version}.pdf" for arxiv_id, version, _, _ in PAPERS)
    if actual != expected:
        raise ValueError(f"Corpus must contain exactly {expected}; found {actual}")

    manifest = {
        "corpus_scope": "Exactly the three assignment-specified arXiv versions",
        "manifest_created_at_utc": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "papers": rows,
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {len(rows)} versioned papers; manifest: {MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
