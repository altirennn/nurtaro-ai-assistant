from dataclasses import replace
from pathlib import Path
import uuid

import pytest

from app.config import Settings
from app.db import Store
from app.models.provider import ExtractiveProvider
from app.rag.embedder import HashingEmbedder
from app.service import AssistantService

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def service():
    database_path = PROJECT_ROOT / "data" / f"test-{uuid.uuid4().hex}.sqlite3"
    settings = replace(
        Settings.from_env(),
        project_root=PROJECT_ROOT,
        database_path=database_path,
        api_keys={"alice-secret": "alice", "bob-secret": "bob", "bot-secret": "telegram-bot"},
        max_upload_bytes=2048,
        min_retrieval_score=0.08,
    )
    instance = AssistantService(settings, Store(settings.database_path), HashingEmbedder(), ExtractiveProvider())
    try:
        yield instance
    finally:
        for suffix in ("", "-wal", "-shm"):
            path = Path(f"{database_path}{suffix}")
            if path.exists():
                path.unlink()
