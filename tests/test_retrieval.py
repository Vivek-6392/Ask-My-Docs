"""
Unit tests for retrieval components.
Run with:  pytest tests/ -v
"""

import json
from types import SimpleNamespace

from retrieval.bm25_store import BM25Store, _tokenize
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.prompt import SYSTEM_PROMPT, build_rag_prompt

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


# ── BM25Store tests ───────────────────────────────────────────────────────────


class TestBM25Store:
    def test_bm25_session_isolation(self, tmp_path):
        config = SimpleNamespace(chroma_persist_dir=str(tmp_path))
        store_user1 = BM25Store(config, session_id="user_1")
        store_user2 = BM25Store(config, session_id="user_2")

        store_user1.add_chunks(
            [
                {
                    "text": "python programming language code and guide",
                    "metadata": {"source": "u1.txt"},
                },
                {
                    "text": "general architecture overview without any matching terms",
                    "metadata": {"source": "u1_other1.txt"},
                },
                {
                    "text": "sample documentation details for testing indexing",
                    "metadata": {"source": "u1_other2.txt"},
                },
            ]
        )
        store_user2.add_chunks(
            [
                {
                    "text": "rust systems programming code and memory safety",
                    "metadata": {"source": "u2.txt"},
                },
                {
                    "text": "general architecture overview without any matching terms",
                    "metadata": {"source": "u2_other1.txt"},
                },
                {
                    "text": "sample documentation details for testing indexing",
                    "metadata": {"source": "u2_other2.txt"},
                },
            ]
        )

        # User 1 searches python -> found
        hits_u1 = store_user1.search("python")
        assert len(hits_u1) == 1
        assert hits_u1[0]["source"] == "u1.txt"

        # User 2 searches python -> nothing
        hits_u2 = store_user2.search("python")
        assert len(hits_u2) == 0

        # User 2 searches rust -> found
        hits_u2_rust = store_user2.search("rust")
        assert len(hits_u2_rust) == 1
        assert hits_u2_rust[0]["source"] == "u2.txt"

    def test_bm25_json_persistence(self, tmp_path):
        config = SimpleNamespace(chroma_persist_dir=str(tmp_path))
        store = BM25Store(config, session_id="test_session")
        chunks = [
            {
                "text": "artificial intelligence and machine learning models",
                "metadata": {"source": "ai.txt", "page": 1},
            },
            {
                "text": "recipes for cooking Italian dishes and pasta",
                "metadata": {"source": "food.txt", "page": 2},
            },
            {
                "text": "astronomy and space exploration missions",
                "metadata": {"source": "space.txt", "page": 3},
            },
        ]
        store.add_chunks(chunks)

        index_file = tmp_path / "bm25_index_test_session.json"
        assert index_file.exists()
        with open(index_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) == 3
        assert data[0]["text"] == "artificial intelligence and machine learning models"

        # Reload store from disk
        reloaded_store = BM25Store(config, session_id="test_session")
        hits = reloaded_store.search("intelligence")
        assert len(hits) == 1
        assert hits[0]["source"] == "ai.txt"


# ── Session helper tests ──────────────────────────────────────────────────────


class TestSessionHelpers:
    def test_add_message_trims_to_custom_max_history(self, monkeypatch):
        import streamlit as st

        from app.session import add_message, clear_history, init_session

        class MockSessionState(dict):
            def __getattr__(self, item):
                return self[item]

            def __setattr__(self, key, value):
                self[key] = value

        mock_state = MockSessionState()
        monkeypatch.setattr(st, "session_state", mock_state)

        init_session()
        assert "session_id" in st.session_state
        assert "rerank_top_k" in st.session_state

        for i in range(10):
            add_message("user", f"msg {i}", max_history=5)

        assert len(st.session_state.messages) == 5
        assert st.session_state.messages[-1]["content"] == "msg 9"
        assert st.session_state.messages[0]["content"] == "msg 5"

        clear_history()
        assert st.session_state.messages == []
