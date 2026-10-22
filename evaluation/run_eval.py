from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import Settings
from app.models.provider import AnswerProvider, ExtractiveProvider, GenerationError, no_information
from app.rag.language import detect_language
from app.rag.retriever import search
from app.service import AssistantService


def _cases(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_evaluation(
    service: AssistantService | None = None,
    use_configured_model: bool = False,
) -> tuple[dict[str, Any], Path, Path]:
    if service is None:
        settings = Settings.from_env()
        service = AssistantService.create(settings)
    settings = service.settings
    case_path = settings.project_root / "evaluation" / "cases.jsonl"
    cases = _cases(case_path)
    provider: AnswerProvider = service.provider if use_configured_model else ExtractiveProvider()
    external_provider = use_configured_model and provider.name != "extractive"
    model_calls = 0
    rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    reciprocal_ranks: list[float] = []
    source_hits: list[float] = []
    evidence_scores: list[float] = []
    abstention_correct: list[float] = []
    groundedness: list[float] = []
    stable: list[float] = []

    for case in cases:
        question = str(case["question"])
        started = time.perf_counter()
        result = service.answer("__evaluation__", question, provider_override=provider)
        if external_provider and result["sources"]:
            model_calls += 1
        latency = (time.perf_counter() - started) * 1000
        latencies.append(latency)
        expected = case["source"]
        titles = [str(source["title"]) for source in result["sources"]]
        rank = next((idx + 1 for idx, title in enumerate(titles) if title == expected), None) if expected else None
        if expected:
            reciprocal_ranks.append(1 / rank if rank else 0.0)
            source_hit = float(rank is not None)
            source_hits.append(source_hit)
        else:
            source_hit = None
        retrieved_text = " ".join(str(source["snippet"]).casefold() for source in result["sources"])
        terms = [str(term).casefold() for term in case["evidence"]]
        evidence_score = sum(term in retrieved_text for term in terms) / len(terms) if terms else None
        if evidence_score is not None:
            evidence_scores.append(evidence_score)
        should_abstain = expected is None
        abstention_correct.append(float(bool(result["abstained"]) == should_abstain))
        cited = result["sources"]
        answer = str(result["answer"]).casefold()
        groundedness_value = float(bool(cited) and any(str(source["snippet"]).casefold()[:120] in answer for source in cited))
        if expected:
            groundedness.append(groundedness_value)
        repeat = search(
            service.store, "__evaluation__", question, service.embedder,
            service.settings.top_k, service.settings.min_retrieval_score,
            language=str(result["language"]),
        )
        repeat_ids = [f"{item['document_id']}#chunk-{int(item['ordinal']) + 1}" for item in repeat]
        stable.append(float(repeat_ids == [str(item["source_id"]) for item in cited]))
        rows.append({
            "id": case["id"], "question": question,
            "expected_evidence": "|".join(str(term) for term in case["evidence"]),
            "language_expected": case["language"], "language_detected": result["language"],
            "expected_source": expected or "", "retrieved_sources": "|".join(titles),
            "source_hit": int(source_hit) if source_hit is not None else "",
            "reciprocal_rank": round(1 / rank if rank else 0.0, 4) if expected else "",
            "expected_evidence_recall_proxy": round(evidence_score, 4) if evidence_score is not None else "",
            "abstained": int(bool(result["abstained"])), "abstention_correct": int(abstention_correct[-1]),
            "groundedness_proxy": int(groundedness_value), "stable_retrieval": int(stable[-1]),
            "latency_ms": round(latency, 2),
            "answer": str(result["answer"]),
            "top_source_snippet": str(cited[0]["snippet"]) if cited else "",
        })

    no_rag: list[str] = []
    for case in cases:
        if external_provider:
            model_calls += 1
        try:
            no_rag.append(provider.generate(str(case["question"]), str(case["language"]), [], ""))
        except GenerationError:
            no_rag.append(no_information(str(case["language"])))

    def looks_like_abstention(answer: str) -> bool:
        lowered = answer.casefold()
        return any(marker in lowered for marker in (
            "недостаточно информации", "не буду придумывать", "жеткілікті ақпарат жоқ", "ойдан шығармаймын",
        ))

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary: dict[str, Any] = {
        "run_id": run_id,
        "cases": len(cases),
        "language_counts": {lang: sum(case["language"] == lang for case in cases) for lang in ("ru", "kk")},
        "embedding_backend": service.embedder.name,
        "generation_backend": provider.name,
        "retrieval_source_hit_at_expected_source": round(statistics.mean(source_hits), 4),
        "mean_reciprocal_rank": round(statistics.mean(reciprocal_ranks), 4),
        "expected_evidence_recall_proxy": round(statistics.mean(evidence_scores), 4),
        "answerable_case_groundedness_proxy": round(statistics.mean(groundedness), 4),
        "abstention_accuracy": round(statistics.mean(abstention_correct), 4),
        "no_answer_abstention_rate": round(sum(bool(rows[i]["abstained"]) for i, case in enumerate(cases) if case["source"] is None) / max(1, sum(case["source"] is None for case in cases)), 4),
        "language_accuracy": round(sum(row["language_detected"] == row["language_expected"] for row in rows) / len(rows), 4),
        "stable_retrieval_rate": round(statistics.mean(stable), 4),
        "average_latency_ms": round(statistics.mean(latencies), 2),
        "p95_latency_ms": round(sorted(latencies)[min(len(latencies) - 1, int(0.95 * len(latencies)))], 2),
        "rag_answerable_case_coverage": round(sum(not bool(rows[i]["abstained"]) for i, case in enumerate(cases) if case["source"] is not None) / max(1, sum(case["source"] is not None for case in cases)), 4),
        "no_rag_answer_coverage": round(sum(not looks_like_abstention(text) for text in no_rag) / len(cases), 4),
        "no_rag_source_citation_coverage": 0.0,
        "model_calls": model_calls,
        "evaluation_mode": "configured provider with/no context" if use_configured_model else "deterministic extractive reference",
        "limitations": "Character-ngram/hash embeddings and automated proxies do not establish answer quality. Human review is still required.",
    }
    output_dir = settings.project_root / "evaluation"
    json_path = output_dir / "results.json"
    csv_path = output_dir / "results.csv"
    payload = {"summary": summary, "cases": rows, "manual_review": {"required": True, "rubric": "See evaluation/MANUAL_REVIEW.md"}}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return summary, json_path, csv_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run deterministic bilingual RAG retrieval evaluation")
    parser.add_argument(
        "--with-configured-model", action="store_true",
        help="Compare configured generator with and without retrieval; may use paid inference",
    )
    args = parser.parse_args()
    result, json_file, csv_file = run_evaluation(use_configured_model=args.with_configured_model)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"JSON: {json_file}\nCSV: {csv_file}")
