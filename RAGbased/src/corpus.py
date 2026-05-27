"""Load and chunk the ethical-principles corpus from markdown files.

Each markdown file in `corpus/` is treated as one source. The file is split
into chunks at every `## ` heading; each chunk takes the heading as a title
and everything until the next `## ` (or EOF) as its body. Heading-less
preamble is attached to the first chunk if present.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path

from src.utils import get_logger

log = get_logger(__name__)


@dataclass
class Chunk:
    id: str          # stable id "source#title"
    source: str      # filename without extension
    title: str       # heading text (or "preamble")
    text: str        # body, trimmed

    def to_retrieval_text(self) -> str:
        """The string the retriever will embed / index."""
        return f"{self.title}. {self.text}"

    def to_prompt_block(self) -> str:
        """The string that will be injected into the LLM prompt."""
        return f"- [{self.source} — {self.title}] {self.text}"


_HEADING_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


def _split_markdown(text: str) -> list[tuple[str, str]]:
    """Return [(title, body), ...]. Body has its leading whitespace stripped."""
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return [("preamble", text.strip())]
    chunks: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        title = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            chunks.append((title, body))
    return chunks


def load_corpus(corpus_dir: str | Path) -> list[Chunk]:
    corpus_dir = Path(corpus_dir)
    if not corpus_dir.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    chunks: list[Chunk] = []
    for path in sorted(corpus_dir.glob("*.md")):
        source = path.stem
        text = path.read_text(encoding="utf-8")
        for title, body in _split_markdown(text):
            cid = f"{source}#{title}"
            chunks.append(Chunk(id=cid, source=source, title=title, text=body))

    log.info(f"Loaded {len(chunks)} chunks from {corpus_dir}")
    return chunks


def chunks_to_dicts(chunks: list[Chunk]) -> list[dict]:
    return [asdict(c) for c in chunks]


def dicts_to_chunks(items: list[dict]) -> list[Chunk]:
    return [Chunk(**d) for d in items]
