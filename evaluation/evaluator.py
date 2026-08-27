"""
Evaluation pipeline using RAGAS metrics.
Run standalone:
    python -m evaluation.evaluator --dataset evaluation/eval_dataset.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import warnings
from pathlib import Path

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)


from datasets import Dataset  # noqa: E402
from langchain_groq import ChatGroq  # noqa: E402
from langchain_huggingface import HuggingFaceEmbeddings  # noqa: E402
from ragas import evaluate  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import (  # noqa: E402
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)
from ragas.run_config import RunConfig  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import AppConfig  # noqa: E402
from retrieval.pipeline import RAGPipeline  # noqa: E402

THRESHOLDS = {
    "faithfulness": 0.85,
    "answer_relevancy": 0.80,
    "context_precision": 0.75,
    "context_recall": 0.70,
}


def load_eval_dataset(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_ragas_dataset(records: list[dict], pipeline: RAGPipeline) -> Dataset:
    questions: list[str] = []
    answers: list[str] = []
    contexts: list[list[str]] = []
    ground_truths: list[str] = []

    for rec in records:
        q = rec["question"]
        gt = rec["ground_truth"]
        result = pipeline.query(q)
        questions.append(q)
        answers.append(result["answer"])
        contexts.append([c["text"] for c in result["chunks"]])
        ground_truths.append(gt)

    return Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )


def build_llm_and_embeddings(config: AppConfig):
    chat_model = ChatGroq(
        api_key=config.groq_api_key,
        model=config.ragas_llm_model,
        temperature=0.0,
        max_tokens=4096,
        max_retries=3,
        request_timeout=120.0,
    )
    llm = LangchainLLMWrapper(chat_model)
    hf_embeddings = HuggingFaceEmbeddings(model_name=config.embedding_model)
    embeddings = LangchainEmbeddingsWrapper(hf_embeddings)
    return llm, embeddings


def run_evaluation(dataset_path: str, config: AppConfig) -> dict[str, float]:
    records = load_eval_dataset(dataset_path)
    pipeline = RAGPipeline(config)
    ds = build_ragas_dataset(records, pipeline)
    llm, embeddings = build_llm_and_embeddings(config)

    run_config = RunConfig(
        max_workers=1,
        timeout=120,
        max_retries=3,
    )

    metrics = [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=embeddings, strictness=1),
        ContextPrecision(llm=llm),
        ContextRecall(llm=llm),
    ]

    result = evaluate(
        dataset=ds,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        run_config=run_config,
    )

    if not getattr(result, "scores", None):
        raise ValueError("RAGAS returned empty scores")

    scores: dict[str, float] = {}
    metric_names = result.scores[0].keys()
    for metric in metric_names:
        valid_values: list[float] = []
        for row in result.scores:
            value = row.get(metric)
            if value is None:
                continue
            if isinstance(value, float) and math.isnan(value):
                continue
            valid_values.append(float(value))
        scores[metric] = sum(valid_values) / len(valid_values) if valid_values else float("nan")

    return scores


def check_thresholds(scores: dict[str, float], thresholds: dict[str, float]) -> bool:
    passed = True
    for metric, threshold in thresholds.items():
        val = scores.get(metric, float("nan"))
        if math.isnan(val):
            print(f"  ⚠️  {metric}: NaN — all evaluation requests failed for this metric")
            passed = False
        else:
            status = "✅" if val >= threshold else "❌"
            print(f"  {status} {metric}: {val:.4f} (threshold {threshold})")
            if val < threshold:
                passed = False
    return passed


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

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
