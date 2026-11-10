"""Rebuild local retrieval collections from the three authoritative PDFs."""

from __future__ import annotations

import json

from relay.retrieval.index import build_indexes


def main() -> None:
    manifest = build_indexes()
    print(json.dumps({"status": "built", "settings": manifest["stats"]}, indent=2))


if __name__ == "__main__":
    main()
