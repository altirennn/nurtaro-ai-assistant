import json
from collections import Counter
from pathlib import Path
import uuid

from app.vision.transcripts import transcript_to_document
from finetuning.prepare_data import prepare


ROOT = Path(__file__).resolve().parents[1]


def test_finetuning_dataset_keeps_train_validation_and_test_splits():
    prepared = ROOT / "finetuning" / f".prepared-test-{uuid.uuid4().hex}.jsonl"
    try:
        count = prepare(ROOT / "finetuning" / "examples.jsonl", prepared)
        rows = [json.loads(line) for line in prepared.read_text(encoding="utf-8").splitlines()]
    finally:
        prepared.unlink(missing_ok=True)
    assert count == 28
    assert Counter(row["split"] for row in rows) == {"train": 16, "validation": 4, "test": 8}
    assert Counter(row["language"] for row in rows) == {"ru": 14, "kk": 14}


def test_timestamped_transcript_is_normalized_for_rag():
    document = transcript_to_document(
        "Урок игры",
        "kk",
        "WEBVTT\n\n00:00:03.000 --> 00:00:07.500\n"
        "Үлкен текшені қорапқа бірге салыңыз.",
    )
    assert "Язык: kk" in document
    assert "[00:00:03–00:00:07]" in document
    assert "Үлкен текшені қорапқа бірге салыңыз." in document
