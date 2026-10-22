from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Load a saved PEFT adapter for a single prompt")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install finetuning dependencies first") from exc
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype="auto", device_map="auto")
    model = PeftModel.from_pretrained(base, args.adapter)
    inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    print(tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))


if __name__ == "__main__":
    main()
