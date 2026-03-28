"""
Prompt construction and citation extraction.
"""

from __future__ import annotations

SYSTEM_PROMPT = """You are a precise document assistant. Your task is to answer questions ONLY using the provided context.

Rules:
1. Base every claim on the context. If the answer isn't in the context, say: "I don't have enough information to answer this."
2. Always cite your sources using inline markers like [1], [2], etc., corresponding to the numbered context chunks.
3. Be concise and factual. Avoid speculation.
4. If multiple chunks support a claim, cite all of them: [1][3].
5. Never fabricate information or draw on outside knowledge.
"""


def build_rag_prompt(chunks: list[dict]) -> tuple[str, list[dict]]:
    """
    Build the context string injected into the user message,
    and return citation metadata for the UI.

    Returns:
        context_str: Numbered chunks for the prompt.
        citations: List of {index, source, page, score, snippet} dicts.
    """
    lines = []
    citations = []

    for i, chunk in enumerate(chunks, start=1):
        source = chunk.get("source", "unknown")
        page = chunk.get("page", "?")
        text = chunk.get("text", "")
        score = chunk.get("score", 0.0)

        lines.append(f"[{i}] Source: {source} | Page: {page}\n{text}\n")

        citations.append({
            "index": i,
            "source": source,
            "page": page,
            "score": score,
            "snippet": text[:300],
        })

    return "\n".join(lines), citations
