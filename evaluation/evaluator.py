"""
Evaluation pipeline using RAGAS metrics.
Run standalone: python -m evaluation.evaluator --dataset eval_dataset.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from datasets import Dataset
from groq import Groq
from langchain_groq import ChatGroq
from ragas import evaluate
from ragas.embeddings import HuggingFaceEmbeddings as RagasHFEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from ragas.run_config import RunConfig

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import AppConfig
from retrieval.pipeline import RAGPipeline


def load_eval_dataset(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_evaluation(dataset_path: str, config: AppConfig) -> dict[str, float]:
    records = load_eval_dataset(dataset_path)
    pipeline = RAGPipeline(config)

    questions: list[str] = []
    answers: list[str] = []
    contexts: list[list[str]] = []
    ground_truths: list[str] = []

    run_config = RunConfig(
        max_workers=1,
        timeout=60,
        max_retries=3,
    )

    for rec in records:
        q = rec["question"]
        gt = rec["ground_truth"]

        result = pipeline.query(q)

        questions.append(q)
        answers.append(result["answer"])
        contexts.append([c["text"] for c in result["chunks"]])
        ground_truths.append(gt)

    ds = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )

    # ✅ Use Langchain ChatGroq wrapper - RAGAS expects OpenAI-compatible chat interface
    groq_client = ChatGroq(
        api_key=config.groq_api_key,
        model_name=config.ragas_llm_model,  # e.g., "llama3-8b-8192" or "llama-3.1-70b-versatile"
        temperature=0,
    )
    llm = LangchainLLMWrapper(groq_client)

    emb = RagasHFEmbeddings(model=config.embedding_model)

    # ✅ Use collections import (removes deprecation warning)
    metrics = [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=emb, strictness=1),
        ContextPrecision(llm=llm),
        ContextRecall(llm=llm),
    ]

    result = evaluate(
        dataset=ds,
        metrics=metrics,
        llm=llm,
        embeddings=emb,
        run_config=run_config,
    )

    if not result.scores:
        raise ValueError("RAGAS returned empty scores")

    scores: dict[str, float] = {}
    for metric in result.scores[0].keys():
        valid = [
            row[metric]
            for row in result.scores
            if row.get(metric) is not None and not math.isnan(row[metric])
        ]
        scores[metric] = sum(valid) / len(valid) if valid else float("nan")

    return scores


def check_thresholds(scores: dict[str, float], thresholds: dict[str, float]) -> bool:
    passed = True
    for metric, threshold in thresholds.items():
        val = scores.get(metric, 0.0)
        if math.isnan(val):
            print(f"  ⚠️  {metric}: NaN — all evaluation requests failed for this metric")
            passed = False
        else:
            status = "✅" if val >= threshold else "❌"
            print(f"  {status} {metric}: {val:.4f} (threshold {threshold})")
            if val < threshold:
                passed = False
    return passed


THRESHOLDS = {
    "faithfulness": 0.80,
    "answer_relevancy": 0.75,
    "context_precision": 0.70,
    "context_recall": 0.70,
}


def main() -> None:
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

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"scores": scores, "passed": passed}, f, indent=2)

    print(f"\n💾 Results saved to {args.output}")

    if not passed:
        print("\n❌ Evaluation FAILED — thresholds not met.")
        sys.exit(1)

    print("\n✅ Evaluation PASSED.")


if __name__ == "__main__":
    main()