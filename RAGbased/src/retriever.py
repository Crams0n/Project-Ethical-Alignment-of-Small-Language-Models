"""Retrieval backends for the ethical corpus.

Two interchangeable backends behind one interface:
  - DenseRetriever: sentence-transformer encoder + cosine similarity (numpy).
  - BM25Retriever:  rank-bm25 lexical baseline.

The dense index is saved as two files in the index directory:
  - chunks.json   list of Chunk dicts (so retrieved hits can be reattached)
  - vectors.npy   (N, D) float32, L2-normalized embeddings

The BM25 backend is built in-memory each time (cheap on this corpus size).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.corpus import Chunk, chunks_to_dicts, dicts_to_chunks, load_corpus
from src.utils import get_logger

log = get_logger(__name__)


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


class BaseRetriever:
    name: str = "base"

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        raise NotImplementedError

    def search_batch(self, queries: list[str], k: int) -> list[list[RetrievedChunk]]:
        return [self.search(q, k) for q in queries]


# ---------------------------------------------------------------------------
# Dense retriever — sentence-transformers + numpy cosine
# ---------------------------------------------------------------------------

class DenseRetriever(BaseRetriever):
    name = "dense"

    def __init__(
        self,
        chunks: list[Chunk],
        vectors: np.ndarray,
        encoder_name: str,
        encoder=None,
    ):
        if vectors.shape[0] != len(chunks):
            raise ValueError(
                f"vectors/chunks length mismatch: {vectors.shape[0]} vs {len(chunks)}"
            )
        self.chunks = chunks
        self.vectors = vectors.astype(np.float32, copy=False)
        self.encoder_name = encoder_name
        self._encoder = encoder  # may be None until first query

    # ----- factory methods -------------------------------------------------

    @classmethod
    def build(
        cls,
        corpus_dir: str | Path,
        encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
    ) -> "DenseRetriever":
        from sentence_transformers import SentenceTransformer

        chunks = load_corpus(corpus_dir)
        log.info(f"Loading encoder '{encoder_name}' on {device}")
        encoder = SentenceTransformer(encoder_name, device=device)
        texts = [c.to_retrieval_text() for c in chunks]
        log.info(f"Encoding {len(texts)} chunks")
        vectors = encoder.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype(np.float32)
        return cls(chunks, vectors, encoder_name, encoder)

    def save(self, index_dir: str | Path) -> None:
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        np.save(index_dir / "vectors.npy", self.vectors)
        with open(index_dir / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(
                {"encoder_name": self.encoder_name, "chunks": chunks_to_dicts(self.chunks)},
                f,
                ensure_ascii=False,
                indent=2,
            )
        log.info(f"Saved dense index ({len(self.chunks)} chunks) to {index_dir}")

    @classmethod
    def load(cls, index_dir: str | Path, device: str = "cpu") -> "DenseRetriever":
        index_dir = Path(index_dir)
        vectors = np.load(index_dir / "vectors.npy")
        with open(index_dir / "chunks.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        chunks = dicts_to_chunks(meta["chunks"])
        encoder_name = meta["encoder_name"]
        # Encoder loaded lazily on first query to keep load() cheap.
        ret = cls(chunks, vectors, encoder_name, encoder=None)
        ret._device = device
        return ret

    # ----- search ----------------------------------------------------------

    def _ensure_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            dev = getattr(self, "_device", "cpu")
            log.info(f"Lazy-loading encoder '{self.encoder_name}' on {dev}")
            self._encoder = SentenceTransformer(self.encoder_name, device=dev)
        return self._encoder

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        return self.search_batch([query], k)[0]

    def search_batch(self, queries: list[str], k: int) -> list[list[RetrievedChunk]]:
        enc = self._ensure_encoder()
        q_vec = enc.encode(
            queries,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)
        # cosine sim = dot product since both sides are L2-normalized.
        scores = q_vec @ self.vectors.T  # (Q, N)
        results: list[list[RetrievedChunk]] = []
        for row in scores:
            top_idx = np.argsort(-row)[:k]
            results.append([RetrievedChunk(self.chunks[i], float(row[i])) for i in top_idx])
        return results


# ---------------------------------------------------------------------------
# BM25 retriever — lexical baseline
# ---------------------------------------------------------------------------

class BM25Retriever(BaseRetriever):
    name = "bm25"

    def __init__(self, chunks: list[Chunk]):
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as e:
            raise ImportError(
                "rank_bm25 is required for the BM25 retriever. Install with "
                "`pip install rank-bm25`."
            ) from e
        self.chunks = chunks
        tokenized = [self._tokenize(c.to_retrieval_text()) for c in chunks]
        self._bm25 = BM25Okapi(tokenized)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [t for t in text.lower().replace("/", " ").split() if t]

    @classmethod
    def build(cls, corpus_dir: str | Path) -> "BM25Retriever":
        return cls(load_corpus(corpus_dir))

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        scores = self._bm25.get_scores(self._tokenize(query))
        top_idx = np.argsort(-scores)[:k]
        return [RetrievedChunk(self.chunks[i], float(scores[i])) for i in top_idx]


# ---------------------------------------------------------------------------
# Convenience: build the retriever named in a config dict
# ---------------------------------------------------------------------------

def build_retriever(retrieval_cfg: dict, corpus_dir: str | Path, device: str = "cpu") -> BaseRetriever:
    backend = retrieval_cfg.get("backend", "dense")
    if backend == "dense":
        return DenseRetriever.build(
            corpus_dir=corpus_dir,
            encoder_name=retrieval_cfg.get(
                "encoder_name", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            device=device,
        )
    if backend == "bm25":
        return BM25Retriever.build(corpus_dir)
    raise ValueError(f"Unknown retrieval backend: {backend}")


def load_retriever(retrieval_cfg: dict, index_dir: str | Path, corpus_dir: str | Path, device: str = "cpu") -> BaseRetriever:
    backend = retrieval_cfg.get("backend", "dense")
    if backend == "dense":
        return DenseRetriever.load(index_dir, device=device)
    if backend == "bm25":
        # BM25 has no persisted index — rebuild from corpus.
        return BM25Retriever.build(corpus_dir)
    raise ValueError(f"Unknown retrieval backend: {backend}")
