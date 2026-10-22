from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter


def prepare(source: Path, destination: Path) -> int:
    count = 0
    splits: Counter[str] = Counter()
    seen: set[tuple[str, str]] = set()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open(encoding="utf-8") as input_file, destination.open("w", encoding="utf-8") as output_file:
        for line_number, line in enumerate(input_file, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            for key in ("instruction", "input", "output"):
                if not isinstance(row.get(key), str) or not row[key].strip():
                    raise ValueError(f"Line {line_number}: expected non-empty string field {key}")
            if row.get("split") not in {"train", "validation", "test"}:
                raise ValueError(f"Line {line_number}: split must be train, validation, or test")
            if row.get("language") not in {"ru", "kk"}:
                raise ValueError(f"Line {line_number}: language must be ru or kk")
            if row["split"] == "test":
                terms = row.get("expected_terms")
                if not isinstance(terms, list) or not terms or not all(isinstance(term, str) and term.strip() for term in terms):
                    raise ValueError(f"Line {line_number}: test rows need non-empty expected_terms")
                if not isinstance(row.get("should_abstain"), bool):
                    raise ValueError(f"Line {line_number}: test rows need boolean should_abstain")
            identity = (row["language"], row["instruction"].strip().casefold())
            if identity in seen:
                raise ValueError(f"Line {line_number}: duplicate instruction within language")
            seen.add(identity)
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
            splits[row["split"]] += 1
    if not count:
        raise ValueError("Training dataset is empty")
    if not all(splits.get(name, 0) for name in ("train", "validation", "test")):
        raise ValueError("Dataset must have non-empty train, validation, and test splits")
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate and normalize the synthetic instruction dataset")
    parser.add_argument("--input", type=Path, default=Path(__file__).with_name("examples.jsonl"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("prepared.jsonl"))
    args = parser.parse_args()
    count = prepare(args.input, args.output)
    rows = [json.loads(line) for line in args.output.read_text(encoding="utf-8").splitlines() if line.strip()]
    splits = Counter(row["split"] for row in rows)
    languages = Counter(row["language"] for row in rows)
    print(f"Prepared {count} synthetic examples at {args.output}; splits={dict(splits)}; languages={dict(languages)}")
