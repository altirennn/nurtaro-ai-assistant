from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .prompting import render_prompt


def _prompt(row: dict[str, Any], tokenizer: Any) -> str:
    return render_prompt(tokenizer, row["instruction"], row["input"], row["language"])


def _generate(model: Any, tokenizer: Any, torch: Any, prompt: str, max_new_tokens: int) -> str:
    tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.inference_mode():
        output = model.generate(**tokens, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(output[0][tokens["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def _score(rows: list[dict[str, Any]], answers: list[str]) -> dict[str, float]:
    term_recall: list[float] = []
    abstention_correct: list[float] = []
    for row, answer in zip(rows, answers):
        lowered = answer.casefold()
        terms = [str(term).casefold() for term in row["expected_terms"]]
        term_recall.append(sum(term in lowered for term in terms) / max(1, len(terms)))
        if row["should_abstain"]:
            markers = ("не буду придумывать", "нет медицинской информации", "ақпарат жоқ", "ойдан шығармаймын")
            abstention_correct.append(float(any(marker in lowered for marker in markers)))
        else:
            abstention_correct.append(float(not any(marker in lowered for marker in ("не буду придумывать", "ойдан шығармаймын"))))
    return {
        "exact_expected_term_recall_proxy": round(statistics.mean(term_recall), 4),
        "abstention_behavior_accuracy_proxy": round(statistics.mean(abstention_correct), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare base and LoRA responses on a synthetic holdout set")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--dataset", type=Path, default=Path(__file__).with_name("prepared.jsonl"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("artifacts") / "adapter-evaluation.json")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    args = parser.parse_args()
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install fine-tuning dependencies first: pip install -r finetuning/requirements-finetuning.txt") from exc

    rows = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any("split" in row for row in rows):
        rows = [row for row in rows if row.get("split") == "test"]
    if not rows:
        raise SystemExit("Test split is empty. Run python -m finetuning.prepare_data first.")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype="auto", device_map="auto")
    base_answers = [_generate(base_model, tokenizer, torch, _prompt(row, tokenizer), args.max_new_tokens) for row in rows]
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    adapted_base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype="auto", device_map="auto")
    adapted_model = PeftModel.from_pretrained(adapted_base, args.adapter)
    adapter_answers = [
        _generate(adapted_model, tokenizer, torch, _prompt(row, tokenizer), args.max_new_tokens) for row in rows
    ]
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_model": args.base_model,
        "adapter": args.adapter,
        "cases": len(rows),
        "metrics_are_proxies": True,
        "metrics_note": "Exact keyword coverage and abstention markers are reproducible checks, not a human answer-quality score.",
        "base_metrics": _score(rows, base_answers),
        "adapter_metrics": _score(rows, adapter_answers),
        "results": [
            {
                "id": row.get("id", f"test_{index + 1}"),
                "language": row["language"],
                "expected_terms": row["expected_terms"],
                "should_abstain": row["should_abstain"],
                "base_answer": base,
                "adapter_answer": adapted,
            }
            for index, (row, base, adapted) in enumerate(zip(rows, base_answers, adapter_answers))
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"base_metrics": report["base_metrics"], "adapter_metrics": report["adapter_metrics"], "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
