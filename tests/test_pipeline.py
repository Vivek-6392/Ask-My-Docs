"""
Integration tests — mock out OpenAI calls so no API key needed.
Run with:  pytest tests/ -v
"""

from unittest.mock import MagicMock, patch


def _make_config():
    from app.config import AppConfig
    return AppConfig(
        groq_api_key="gsk-test",
        llm_model="qwen/qwen3.6-27b",
        chroma_persist_dir="/tmp/test_chroma",
        bm25_top_k=5,
        vector_top_k=5,
        rerank_top_k=3,
    )


@patch("retrieval.pipeline.ChatGroq")
@patch("retrieval.pipeline.VectorStore")
@patch("retrieval.pipeline.BM25Store")
@patch("retrieval.pipeline.CrossEncoderReranker")
class TestRAGPipeline:
    def test_query_returns_expected_keys(
        self, mock_reranker_cls, mock_bm25_cls, mock_vector_cls, mock_llm_cls
    ):
        from retrieval.pipeline import RAGPipeline

        # Setup mocks
        mock_vector = MagicMock()
        mock_vector.similarity_search.return_value = [
            {"text": "chunk a", "score": 0.9, "source": "a.pdf", "page": 1},
        ]
        mock_vector_cls.return_value = mock_vector

        mock_bm25 = MagicMock()
        mock_bm25.search.return_value = [
            {"text": "chunk b", "score": 5.0, "source": "b.pdf", "page": 2},
        ]
        mock_bm25_cls.return_value = mock_bm25

        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [
            {"text": "chunk a", "score": 0.8, "source": "a.pdf", "page": 1},
        ]
        mock_reranker_cls.return_value = mock_reranker

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Based on [1], the answer is X."
        mock_llm.invoke.return_value = mock_response
        mock_llm_cls.return_value = mock_llm

        config = _make_config()
        pipeline = RAGPipeline(config)
        result = pipeline.query("What is X?")

        assert "answer" in result
        assert "citations" in result
        assert "chunks" in result
        assert "latency_ms" in result
        assert result["answer"] == "Based on [1], the answer is X."

    def test_add_documents_calls_both_stores(
        self, mock_reranker_cls, mock_bm25_cls, mock_vector_cls, mock_llm_cls
    ):
        from retrieval.pipeline import RAGPipeline

        mock_vector = MagicMock()
        mock_vector_cls.return_value = mock_vector
        mock_bm25 = MagicMock()
        mock_bm25_cls.return_value = mock_bm25
        mock_reranker_cls.return_value = MagicMock()
        mock_llm_cls.return_value = MagicMock()

        config = _make_config()
        pipeline = RAGPipeline(config)

        chunks = [{"text": "hello", "metadata": {"source": "x.pdf"}}]
        n = pipeline.add_documents(chunks)

        mock_vector.add_chunks.assert_called_once_with(chunks)
        mock_bm25.add_chunks.assert_called_once_with(chunks)
        assert n == 1

    def test_query_removes_model_thinking_text(
        self, mock_reranker_cls, mock_bm25_cls, mock_vector_cls, mock_llm_cls
    ):
        from retrieval.pipeline import RAGPipeline

        mock_vector_cls.return_value.similarity_search.return_value = []
        mock_bm25_cls.return_value.search.return_value = []
        mock_reranker_cls.return_value.rerank.return_value = []
        mock_llm_cls.return_value.invoke.return_value.content = (
            "<think>Reason through the answer privately.</think>\nFinal answer [1]."
        )

        result = RAGPipeline(_make_config()).query("What is the answer?")

        assert result["answer"] == "Final answer [1]."

    def test_query_hides_unclosed_model_thinking_text(
        self, mock_reranker_cls, mock_bm25_cls, mock_vector_cls, mock_llm_cls
    ):
        from retrieval.pipeline import RAGPipeline

        mock_vector_cls.return_value.similarity_search.return_value = []
        mock_bm25_cls.return_value.search.return_value = []
        mock_reranker_cls.return_value.rerank.return_value = []
        mock_llm_cls.return_value.invoke.return_value.content = (
            "<think>Reason through the answer privately."
        )

        result = RAGPipeline(_make_config()).query("What is the answer?")

        assert result["answer"] == ""
