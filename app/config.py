from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # Environment variables can also be supplied by Docker/systemd.
    pass


def _parse_api_keys(raw: str) -> dict[str, str]:
    pairs: dict[str, str] = {}
    for item in raw.split(";"):
        if "=" not in item:
            continue
        principal, secret = item.split("=", 1)
        if principal.strip() and secret.strip():
            pairs[secret.strip()] = principal.strip()
    return pairs


@dataclass(frozen=True)
class Settings:
    project_root: Path
    database_path: Path
    api_keys: dict[str, str]
    bot_principal: str
    llm_provider: str
    ollama_base_url: str
    ollama_model: str
    openai_base_url: str
    openai_model: str
    openai_api_key: str
    temperature: float
    max_tokens: int
    llm_timeout: float
    embedding_backend: str
    sentence_transformer_model: str
    top_k: int
    min_retrieval_score: float
    max_upload_bytes: int
    max_pdf_pages: int
    vision_model: str
    vision_confidence: float
    vision_max_upload_bytes: int
    vision_max_duration_seconds: int
    vision_sample_interval_seconds: float
    vision_max_frames: int
    vision_max_dimension: int

    @classmethod
    def from_env(cls) -> "Settings":
        root = Path(__file__).resolve().parents[1]
        db_path = Path(os.getenv("DATABASE_PATH", str(root / "data" / "nurtaro.sqlite3")))
        if not db_path.is_absolute():
            db_path = (root / db_path).resolve()
        return cls(
            project_root=root,
            database_path=db_path,
            api_keys=_parse_api_keys(os.getenv("API_KEYS", "")),
            bot_principal=os.getenv("BOT_PRINCIPAL", "telegram-bot"),
            llm_provider=os.getenv("LLM_PROVIDER", "extractive").lower(),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"),
            ollama_model=os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct-q4_K_M"),
            openai_base_url=os.getenv("OPENAI_COMPATIBLE_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
            openai_model=os.getenv("OPENAI_COMPATIBLE_MODEL", "qwen2.5"),
            openai_api_key=os.getenv("OPENAI_COMPATIBLE_API_KEY", ""),
            temperature=max(0.0, min(2.0, float(os.getenv("LLM_TEMPERATURE", "0.2")))),
            max_tokens=max(32, min(4096, int(os.getenv("LLM_MAX_TOKENS", "512")))),
            llm_timeout=max(1.0, min(180.0, float(os.getenv("LLM_TIMEOUT_SECONDS", "45")))),
            embedding_backend=os.getenv("EMBEDDING_BACKEND", "hashing").lower(),
            sentence_transformer_model=os.getenv("SENTENCE_TRANSFORMER_MODEL", "intfloat/multilingual-e5-small"),
            top_k=max(1, min(10, int(os.getenv("TOP_K", "4")))),
            min_retrieval_score=max(0.0, min(1.0, float(os.getenv("MIN_RETRIEVAL_SCORE", "0.08")))),
            max_upload_bytes=max(1024, min(20 * 1024 * 1024, int(os.getenv("MAX_UPLOAD_BYTES", "5242880")))),
            max_pdf_pages=max(1, min(200, int(os.getenv("MAX_PDF_PAGES", "50")))),
            vision_model=os.getenv("VISION_MODEL", "yolo11n.pt"),
            vision_confidence=max(0.05, min(0.95, float(os.getenv("VISION_CONFIDENCE", "0.35")))),
            vision_max_upload_bytes=max(1024, min(200 * 1024 * 1024, int(os.getenv("VISION_MAX_UPLOAD_BYTES", str(50 * 1024 * 1024))))),
            vision_max_duration_seconds=max(1, min(300, int(os.getenv("VISION_MAX_DURATION_SECONDS", "30")))),
            vision_sample_interval_seconds=max(0.25, min(10.0, float(os.getenv("VISION_SAMPLE_INTERVAL_SECONDS", "1.0")))),
            vision_max_frames=max(1, min(120, int(os.getenv("VISION_MAX_FRAMES", "30")))),
            vision_max_dimension=max(320, min(4096, int(os.getenv("VISION_MAX_DIMENSION", "1920")))),
        )
