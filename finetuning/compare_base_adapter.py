from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .prompting import render_prompt


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a side-by-side base/adapter comparison JSON")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--prompts", type=Path, default=Path(__file__).with_name("comparison_prompts.json"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("artifacts") / "comparison.json")
    args = parser.parse_args()
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install finetuning dependencies first") from exc
    prompts = json.loads(args.prompts.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    def generate(model, prompt: str) -> str:
        formatted_prompt = render_prompt(tokenizer, prompt)
        inputs = tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=192, do_sample=False)
        return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype="auto", device_map="auto")
    base_outputs = [generate(base, prompt) for prompt in prompts]
    del base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    adapted_base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype="auto", device_map="auto")
    adapted = PeftModel.from_pretrained(adapted_base, args.adapter)
    tuned_outputs = [generate(adapted, prompt) for prompt in prompts]
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_model": args.base_model, "adapter": args.adapter,
        "prompts": prompts,
        "outputs": [{"prompt": p, "base": b, "adapter": a} for p, b, a in zip(prompts, base_outputs, tuned_outputs)],
        "metrics": "No automatic quality score is calculated; review each pair using evaluation/MANUAL_REVIEW.md.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved side-by-side comparison to {args.output}")


if __name__ == "__main__":
    main()
