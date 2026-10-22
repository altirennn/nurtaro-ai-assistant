from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small LoRA/QLoRA adapter; this script does not run at import time")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument(
        "--model-id",
        help="Canonical Hugging Face model ID to record when --base-model is a local cache path",
    )
    parser.add_argument("--dataset", type=Path, default=Path(__file__).with_name("prepared.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("artifacts") / "qwen-lora")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--qlora", action="store_true", help="Use bitsandbytes 4-bit loading on a supported CUDA GPU")
    args = parser.parse_args()
    if not args.dataset.exists():
        raise SystemExit("Dataset missing. Run: python -m finetuning.prepare_data")

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments
        from finetuning.prompting import build_messages
    except ImportError as exc:
        raise SystemExit("Install finetuning dependencies: pip install -r requirements-finetuning.txt") from exc

    quantization_config = None
    if args.qlora:
        if not torch.cuda.is_available():
            raise SystemExit("QLoRA requires a supported CUDA GPU; no training was started")
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise SystemExit("Install bitsandbytes to use --qlora") from exc
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quantization_config,
        device_map="auto" if args.qlora else None,
        torch_dtype="auto",
    )
    print(f"Loaded base model: {args.base_model}", flush=True)
    model.config.use_cache = False
    if args.qlora:
        print("Preparing 4-bit model for adapter training", flush=True)
        model = prepare_model_for_kbit_training(model)
        print("4-bit model is ready", flush=True)
    adapter_config = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, adapter_config)
    print("LoRA adapter attached", flush=True)
    rows = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    dataset = Dataset.from_list(rows)
    print(f"Loaded {len(dataset)} examples from {args.dataset}", flush=True)
    split_counts = {
        name: len(dataset.filter(lambda row, split=name: row["split"] == split, load_from_cache_file=False))
        for name in ("train", "validation", "test")
    }
    if not split_counts["train"]:
        raise SystemExit("No train examples found in dataset split")
    train_dataset = dataset.filter(lambda row: row["split"] == "train", load_from_cache_file=False)
    validation_dataset = dataset.filter(lambda row: row["split"] == "validation", load_from_cache_file=False)
    if not len(validation_dataset):
        raise SystemExit("No validation examples found in dataset split")

    def tokenize(row):
        prompt_ids = tokenizer.apply_chat_template(
            build_messages(row["instruction"], row["input"], row["language"]),
            tokenize=True,
            add_generation_prompt=True,
        )
        answer_ids = tokenizer(row["output"].strip() + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
        if len(prompt_ids) >= args.max_length:
            prompt_ids = prompt_ids[-(args.max_length - 1):]
        answer_ids = answer_ids[:max(1, args.max_length - len(prompt_ids))]
        input_ids = prompt_ids + answer_ids
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": [-100] * len(prompt_ids) + answer_ids,
        }

    tokenized = train_dataset.map(tokenize, remove_columns=train_dataset.column_names, load_from_cache_file=False)
    tokenized_validation = validation_dataset.map(
        tokenize, remove_columns=validation_dataset.column_names, load_from_cache_file=False
    )
    print(
        f"Prepared {len(tokenized)} training and {len(tokenized_validation)} validation examples; "
        f"starting {args.epochs} epoch(s)",
        flush=True,
    )
    training = TrainingArguments(
        output_dir=str(args.output_dir), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=8,
        learning_rate=2e-4, logging_steps=1, save_strategy="epoch",
        report_to="none", fp16=bool(torch.cuda.is_available() and not torch.cuda.is_bf16_supported()),
        bf16=bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        gradient_checkpointing=args.qlora,
        seed=42,
    )
    trainer = Trainer(
        model=model, args=training, train_dataset=tokenized, eval_dataset=tokenized_validation,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True, label_pad_token_id=-100),
    )
    trainer.train()
    print("Training finished; computing validation metrics", flush=True)
    validation_metrics = trainer.evaluate()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    if args.model_id:
        adapter_config_path = args.output_dir / "adapter_config.json"
        adapter_config_json = json.loads(adapter_config_path.read_text(encoding="utf-8"))
        adapter_config_json["base_model_name_or_path"] = args.model_id
        adapter_config_path.write_text(
            json.dumps(adapter_config_json, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    try:
        dataset_label = args.dataset.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        dataset_label = args.dataset.name
    metadata = {
        "base_model": args.model_id or args.base_model,
        "adapter_type": "QLoRA 4-bit NF4" if args.qlora else "LoRA",
        "prompt_format": "base tokenizer chat template with bilingual safety system prompt",
        "dataset": dataset_label, "dataset_splits": split_counts, "training_examples": len(train_dataset),
        "training_ran": True, "validation_loss_metrics": validation_metrics,
        "training_log_history": trainer.state.log_history,
    }
    (args.output_dir / "training_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
