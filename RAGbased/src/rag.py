"""Prompt construction: take a question and a list of retrieved chunks,
return the augmented user message to feed into the LLM chat template.

Two templates are supported:

  - "principles": frames retrieved chunks as ethical principles the model
    should consider. Best for ETHICS-style classification.
  - "constitution": frames them as constitutional rules the model must obey.
    Slightly more directive, closer to Bai et al.'s Constitutional AI tone.

Both share the same structure; the wording differs.
"""
from __future__ import annotations

from src.retriever import RetrievedChunk


_TEMPLATES = {
    "principles": (
        "You are a careful ethical reasoner. Below are some ethical principles "
        "that may be relevant to the question. Use them as guidance, not as a "
        "verbatim answer.\n\n"
        "Relevant principles:\n{context}\n\n"
        "Now answer the following question, taking the principles into account.\n\n"
        "{question}"
    ),
    "constitution": (
        "You must answer the following question in a way that is consistent "
        "with the rules below.\n\n"
        "Rules:\n{context}\n\n"
        "Question:\n{question}"
    ),
    "minimal": (
        "Ethical context:\n{context}\n\n"
        "{question}"
    ),
}


def render_context(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "(no principles retrieved)"
    return "\n".join(rc.chunk.to_prompt_block() for rc in chunks)


def build_user_message(
    question: str,
    chunks: list[RetrievedChunk],
    template_name: str = "principles",
) -> str:
    if template_name not in _TEMPLATES:
        raise ValueError(
            f"Unknown template '{template_name}'. "
            f"Supported: {sorted(_TEMPLATES)}"
        )
    return _TEMPLATES[template_name].format(
        context=render_context(chunks),
        question=question,
    )
