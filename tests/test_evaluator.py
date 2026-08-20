"""Dependency compatibility checks for the optional RAG evaluation job."""


def test_evaluator_imports_with_pinned_ragas_stack():
    """Catch RAGAS/LangChain import incompatibilities before the CI eval job."""
    import evaluation.evaluator as evaluator

    assert evaluator is not None
