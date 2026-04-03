"""
Evaluation pipeline using RAGAS metrics.
Run standalone:  python -m evaluation.evaluator --dataset eval_dataset.json
"""

from __future__ import annotations
import json
import argparse
import sys
import os
from pathlib import Path
from typing import Any

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from app.config import AppConfig
from retrieval.pipeline import RAGPipeline


def load_eval_dataset(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def run_evaluation(dataset_path: str, config: AppConfig) -> dict[str, float]:
    records = load_eval_dataset(dataset_path)
    pipeline = RAGPipeline(config)

    questions, answers, contexts, ground_truths = [], [], [], []

    # 1. Collect results (This part is fine)
    for rec in records:
        q = rec["question"]
        gt = rec["ground_truth"]
        result = pipeline.query(q)

        questions.append(q)
        answers.append(result["answer"])
        contexts.append([c["text"] for c in result["chunks"]])
        ground_truths.append(gt)

    ds = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    # 2. Modern LLM Initialization with Groq safety
    # We use model_kwargs={"n": 1} to prevent the BadRequestError
    eval_chat_model = ChatGroq(
        model=config.ragas_llm_model, 
        api_key=config.groq_api_key,
        temperature=0,
        max_retries=3,
        n=1,
    )
    
    llm = LangchainLLMWrapper(eval_chat_model)
    
    emb = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name=config.embedding_model,
            model_kwargs={"device": "cpu"}
        )
    )

    # 3. CRITICAL: Disable Async and Concurrency for Groq Free Tier
    # This prevents the TimeoutErrors and 429 Rate Limit hits
    result = evaluate(
        dataset=ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=emb,
        is_async=False,   # <--- STOP parallel requests
    )

    # Convert Result object to dictionary of means
    return result.scores

def check_thresholds(scores: dict[str, float], thresholds: dict[str, float]) -> bool:
    """Returns True if all scores meet their thresholds."""
    passed = True
    for metric, threshold in thresholds.items():
        val = scores.get(metric, 0.0)
        status = "✅" if val >= threshold else "❌"
        print(f"  {status} {metric}: {val:.4f} (threshold {threshold})")
        if val < threshold:
            passed = False
    return passed


# ── CLI entry point ──────────────────────────────────────────────────────────

THRESHOLDS = {
    "faithfulness": 0.80,
    "answer_relevancy": 0.75,
    "context_precision": 0.70,
    "context_recall": 0.70,
}


def main():
    parser = argparse.ArgumentParser(description="RAG Evaluation Pipeline")
    parser.add_argument("--dataset", default="evaluation/eval_dataset.json")
    parser.add_argument("--output", default="evaluation/results.json")
    args = parser.parse_args()

    config = AppConfig.from_env()
    errors = config.validate()
    if errors:
        print("Config errors:", errors)
        sys.exit(1)

    print(f"\n🔍 Running evaluation on: {args.dataset}")
    scores = run_evaluation(args.dataset, config)

    print("\n📊 RAGAS Scores:")
    passed = check_thresholds(scores, THRESHOLDS)

    # Save results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"scores": scores, "passed": passed}, f, indent=2)
    print(f"\n💾 Results saved to {args.output}")

    if not passed:
        print("\n❌ Evaluation FAILED — thresholds not met.")
        sys.exit(1)
    else:
        print("\n✅ Evaluation PASSED.")


if __name__ == "__main__":
    main()
