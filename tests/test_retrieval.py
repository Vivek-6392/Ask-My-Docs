"""
Unit tests for retrieval components.
Run with:  pytest tests/ -v
"""

import pytest
from unittest.mock import MagicMock, patch
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.prompt import build_rag_prompt, SYSTEM_PROMPT
from retrieval.bm25_store import _tokenize


# ── RRF tests ────────────────────────────────────────────────────────────────

class TestRRF:
    def _make_doc(self, text: str, score: float) -> dict:
        return {"text": text, "score": score, "source": "test", "page": 1}

    def test_single_list_passthrough(self):
        docs = [self._make_doc(f"doc {i}", float(i)) for i in range(5)]
        result = reciprocal_rank_fusion([docs])
        assert len(result) == 5

    def test_deduplication(self):
        doc = self._make_doc("shared document text here", 0.9)
        list1 = [doc, self._make_doc("only in list1", 0.7)]
        list2 = [doc, self._make_doc("only in list2", 0.8)]
        result = reciprocal_rank_fusion([list1, list2])
        texts = [r["text"] for r in result]
        # shared doc should appear once
        shared = [t for t in texts if t.startswith("shared")]
        assert len(shared) == 1

    def test_higher_ranked_gets_better_score(self):
        doc_a = self._make_doc("document alpha", 1.0)
        doc_b = self._make_doc("document beta only here", 0.5)
        list1 = [doc_a, doc_b]
        list2 = [doc_a]
        result = reciprocal_rank_fusion([list1, list2])
        # doc_a appears in both lists so should score higher
        assert result[0]["text"] == doc_a["text"]

    def test_empty_lists(self):
        result = reciprocal_rank_fusion([[], []])
        assert result == []

    def test_rrf_score_key_present(self):
        docs = [self._make_doc("some text document", 0.5)]
        result = reciprocal_rank_fusion([docs])
        assert "rrf_score" in result[0]


# ── Tokenizer tests ───────────────────────────────────────────────────────────

class TestTokenizer:
    def test_lowercases(self):
        tokens = _tokenize("Hello World")
        assert tokens == ["hello", "world"]

    def test_strips_punctuation(self):
        tokens = _tokenize("Hello, World!")
        assert "hello" in tokens
        assert "world" in tokens
        assert "," not in tokens

    def test_empty_string(self):
        assert _tokenize("") == []

    def test_numbers_kept(self):
        tokens = _tokenize("version 3 2 1")
        assert "3" in tokens


# ── Prompt builder tests ──────────────────────────────────────────────────────

class TestPromptBuilder:
    def _make_chunk(self, i: int) -> dict:
        return {
            "text": f"This is chunk number {i} with some content.",
            "source": f"doc_{i}.pdf",
            "page": i,
            "score": 0.9 - i * 0.1,
        }

    def test_returns_tuple(self):
        chunks = [self._make_chunk(i) for i in range(3)]
        result = build_rag_prompt(chunks)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_citations_length(self):
        chunks = [self._make_chunk(i) for i in range(4)]
        _, citations = build_rag_prompt(chunks)
        assert len(citations) == 4

    def test_citation_fields(self):
        chunks = [self._make_chunk(0)]
        _, citations = build_rag_prompt(chunks)
        c = citations[0]
        assert "source" in c
        assert "page" in c
        assert "score" in c
        assert "snippet" in c
        assert "index" in c

    def test_context_contains_source(self):
        chunks = [self._make_chunk(0)]
        context_str, _ = build_rag_prompt(chunks)
        assert "doc_0.pdf" in context_str

    def test_numbered_markers(self):
        chunks = [self._make_chunk(i) for i in range(3)]
        context_str, _ = build_rag_prompt(chunks)
        assert "[1]" in context_str
        assert "[2]" in context_str
        assert "[3]" in context_str

    def test_system_prompt_has_rules(self):
        assert "cite" in SYSTEM_PROMPT.lower()
        assert "context" in SYSTEM_PROMPT.lower()

    def test_empty_chunks(self):
        context_str, citations = build_rag_prompt([])
        assert context_str == ""
        assert citations == []
