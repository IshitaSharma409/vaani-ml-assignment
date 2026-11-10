"""Build and query reproducible local Chroma collections."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from relay.retrieval.chunking import split_pages
from relay.retrieval.ingest import PROJECT_ROOT, load_paper_pages


CONFIG_PATH = PROJECT_ROOT / "config/retrieval.yaml"
# Keep generated Chroma files below the planned directory; on some OneDrive
# checkouts the parent itself is a directory reparse point and must not be removed.
INDEX_DIR = PROJECT_ROOT / "data/vectorstore/relay_chroma"


def load_retrieval_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def make_embeddings(config: dict[str, Any]):
    emb = config["embedding"]
    return HuggingFaceEmbeddings(
        model_name=emb["model_id"],
        model_kwargs={"device": "cpu", "revision": emb["revision"]},
        encode_kwargs={"normalize_embeddings": True},
    )


def _collection_name(setting: str) -> str:
    return f"relay_{setting}"


def build_indexes(config_path: Path = CONFIG_PATH, index_dir: Path = INDEX_DIR) -> dict[str, Any]:
    """Rebuild all configured collections; refuses an unpinned embedding revision."""
    config = load_retrieval_config(config_path)
    embedding = config["embedding"]
    if not embedding.get("revision"):
        raise ValueError("Pin embedding.revision before building the index")
    pages = load_paper_pages()
    embeddings = make_embeddings(config)
    if index_dir.exists():
        shutil.rmtree(index_dir)
    index_dir.mkdir(parents=True)
    stats: dict[str, Any] = {}
    for name, settings in config["settings"].items():
        chunks = split_pages(pages, settings["chunk_size_chars"], settings["chunk_overlap_chars"])
        Chroma.from_documents(
            chunks,
            embedding=embeddings,
            ids=[doc.metadata["chunk_id"] for doc in chunks],
            collection_name=_collection_name(name),
            persist_directory=str(index_dir),
        )
        by_paper: dict[str, list[int]] = {}
        for chunk in chunks:
            by_paper.setdefault(chunk.metadata["paper_id"], []).append(len(chunk.page_content))
        stats[name] = {
            "chunk_count": len(chunks),
            "average_chunk_chars": round(sum(map(len, (d.page_content for d in chunks))) / max(len(chunks), 1), 1),
            "empty_or_short_chunks": sum(len(d.page_content.strip()) < 80 for d in chunks),
            "per_paper": {paper: {"chunks": len(lengths), "average_chars": round(sum(lengths) / len(lengths), 1)} for paper, lengths in by_paper.items()},
        }
    manifest = {
        "schema_version": 1,
        "embedding_model_id": embedding["model_id"],
        "embedding_revision": embedding["revision"],
        "source_manifest_sha256": hashlib.sha256((PROJECT_ROOT / "data/papers/manifest.json").read_bytes()).hexdigest(),
        "settings": config["settings"],
        "stats": stats,
    }
    (index_dir / "index_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


class RetrievalService:
    """Local vector retrieval with an explicit bounded BM25 rerank option."""

    def __init__(self, config_path: Path = CONFIG_PATH, index_dir: Path = INDEX_DIR):
        self.config = load_retrieval_config(config_path)
        self.index_dir = index_dir
        manifest_path = index_dir / "index_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError("Local index is missing; run scripts/build_index.py first")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_hash = hashlib.sha256((PROJECT_ROOT / "data/papers/manifest.json").read_bytes()).hexdigest()
        if self.manifest["source_manifest_sha256"] != source_hash:
            raise ValueError("Index was built from a different corpus manifest; rebuild it")
        if self.manifest["settings"] != self.config["settings"]:
            raise ValueError("Retrieval config differs from index manifest; rebuild the index")
        self.embeddings = make_embeddings(self.config)
        self._stores: dict[str, Chroma] = {}
        self._search_lock = threading.RLock()

    def search(self, query: str, setting: str | None = None, top_k: int | None = None) -> tuple[list[dict[str, Any]], dict[str, float]]:
        """Return metadata-rich passages and separately measured retrieval timings."""
        # Chroma's embedded Rust client is not safe for concurrent collection/query
        # initialization in this Windows environment; serialize only the local search.
        with self._search_lock:
            return self._search_unlocked(query, setting, top_k)

    def _search_unlocked(self, query: str, setting: str | None = None,
                         top_k: int | None = None) -> tuple[list[dict[str, Any]], dict[str, float]]:
        setting = setting or self.config["default_setting"]
        if setting not in self.config["settings"]:
            raise ValueError(f"Unknown retrieval setting: {setting}")
        options = self.config["settings"][setting]
        requested_k = top_k or options.get("candidate_k", options["top_k"])
        store = self._stores.get(setting)
        if store is None:
            store = Chroma(
                collection_name=_collection_name(setting),
                persist_directory=str(self.index_dir),
                embedding_function=self.embeddings,
            )
            self._stores[setting] = store
        start = time.perf_counter()
        distance_results = store.similarity_search_with_score(query, k=requested_k)
        # Normalize Chroma's nonnegative distance to a documented [0, 1] monotonic score.
        results = [(doc, 1.0 / (1.0 + max(float(distance), 0.0)), None) for doc, distance in distance_results]
        vector_seconds = time.perf_counter() - start
        rerank_seconds = 0.0
        if options.get("rerank"):
            start = time.perf_counter()
            query_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
            lexical_scores = self._bm25_candidates(query_terms, [doc.page_content for doc, _, _ in results])
            docs_scores = [(doc, score, lexical) for (doc, score, _), lexical in zip(results, lexical_scores)]
            docs_scores.sort(key=lambda row: (row[2], row[1]), reverse=True)
            results = docs_scores[: options["top_k"]]
            rerank_seconds = time.perf_counter() - start
        else:
            results = results[: options["top_k"]]
        passages = []
        for doc, score, rerank_score in results:
            passages.append({"chunk_text": doc.page_content, "score": float(score),
                             "retrieval_score": float(score),
                             "rerank_score": float(rerank_score) if rerank_score is not None else None,
                             **doc.metadata})
        return passages, {"vector_search_seconds": vector_seconds, "rerank_seconds": rerank_seconds}

    @staticmethod
    def _bm25_score(query: str, text: str) -> float:
        """BM25 lexical score over a bounded candidate set."""
        tokens = lambda s: re.findall(r"[a-z0-9]+", s.lower())
        qterms = set(tokens(query))
        return RetrievalService._bm25_candidates(qterms, [text])[0]

    @staticmethod
    def _bm25_candidates(query_terms: set[str], texts: list[str]) -> list[float]:
        """Score candidates with standard BM25 term frequency, IDF, and length normalization."""
        tokenized = [re.findall(r"[a-z0-9]+", text.lower()) for text in texts]
        average_length = sum(map(len, tokenized)) / max(len(tokenized), 1)
        document_frequency = {term: sum(term in terms for terms in map(set, tokenized)) for term in query_terms}
        count = len(tokenized)
        k1, b = 1.5, 0.75
        scores = []
        for terms in tokenized:
            counts = Counter(terms)
            score = 0.0
            for term in query_terms:
                frequency = counts[term]
                if frequency:
                    idf = math.log(1 + (count - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5))
                    denominator = frequency + k1 * (1 - b + b * len(terms) / max(average_length, 1))
                    score += idf * frequency * (k1 + 1) / denominator
            scores.append(score)
        return scores
