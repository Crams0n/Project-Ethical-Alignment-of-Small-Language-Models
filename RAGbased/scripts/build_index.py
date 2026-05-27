"""Build (or rebuild) the dense retrieval index from the ethical corpus.

For BM25 there's nothing to persist — this script is a no-op in that case.
"""
from __future__ import annotations

import argparse

from src.retriever import DenseRetriever
from src.utils import get_logger, load_config

log = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rag.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    backend = cfg["retrieval"].get("backend", "dense")
    if backend != "dense":
        log.info(f"Backend is '{backend}'; nothing to persist. Done.")
        return

    retriever = DenseRetriever.build(
        corpus_dir=cfg["corpus"]["dir"],
        encoder_name=cfg["retrieval"]["encoder_name"],
        device=cfg["retrieval"].get("encoder_device", "cpu"),
    )
    retriever.save(cfg["retrieval"]["index_dir"])
    log.info(
        f"Index built: {len(retriever.chunks)} chunks, "
        f"dim={retriever.vectors.shape[1]}, saved to {cfg['retrieval']['index_dir']}"
    )


if __name__ == "__main__":
    main()
