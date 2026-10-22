from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from app.config import Settings
from app.db import Store
from app.models.provider import AnswerProvider, GenerationError, create_provider, no_information
from app.rag.embedder import Embedder, create_embedder
from app.rag.ingest import index_pending
from app.rag.language import detect_language
from app.rag.retriever import search

logger = logging.getLogger(__name__)


class AssistantService:
    def __init__(self, settings: Settings, store: Store, embedder: Embedder, provider: AnswerProvider) -> None:
        self.settings = settings
        self.store = store
        self.embedder = embedder
        self.provider = provider
        self.seed_and_index()

    @classmethod
    def create(cls, settings: Settings) -> "AssistantService":
        return cls(settings, Store(settings.database_path), create_embedder(settings.embedding_backend, settings.sentence_transformer_model), create_provider(settings))

    def seed_and_index(self) -> None:
        sample_dir = self.settings.project_root / "data" / "sample"
        for filename in ("play-guide-ru.md", "play-guide-kk.md"):
            source = sample_dir / filename
            document_id = self.store.seed_document(filename, filename, "markdown", source.read_text(encoding="utf-8"))
            index_pending(self.store, "__shared__", self.embedder, document_id, force=True)

    def answer(
        self,
        owner_id: str,
        question: str,
        profile_id: str | None = None,
        provider_override: AnswerProvider | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        profile = self.store.get_profile(profile_id, owner_id) if profile_id else None
        if profile_id and not profile:
            raise LookupError("Профиль не найден")
        preferred = str(profile["preferred_language"]) if profile else "ru"
        language = detect_language(question, preferred)
        profile_parts: list[str] = []
        if profile:
            profile_parts.extend([
                f"возраст: {profile['age_months']} месяцев",
                f"интересы: {', '.join(profile['interests'])}",
                f"доступные материалы: {', '.join(profile['available_materials'])}",
                f"предпочтения родителей: {profile['parent_preferences']}",
            ])
        profile_text = "; ".join(part for part in profile_parts if not part.endswith(": "))
        matches = search(
            self.store, owner_id, question, self.embedder, self.settings.top_k,
            self.settings.min_retrieval_score, profile_text, language,
        )
        abstained = not matches
        sources = [
            {
                "document_id": str(item["document_id"]),
                "source_id": f"{item['document_id']}#chunk-{item['ordinal'] + 1}",
                "title": str(item["title"]),
                "snippet": str(item["text"])[:320],
                "score": float(item["score"]),
            }
            for item in matches
        ]
        if abstained:
            answer = no_information(language)
            mode = "abstention"
        else:
            context = [str(item["text"]) for item in matches]
            provider = provider_override or self.provider
            try:
                answer = provider.generate(question, language, context, profile_text)
                mode = provider.name
            except GenerationError:
                answer = create_provider_fallback(language, question, context, profile_text)
                mode = "extractive-fallback"
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        # Character-based estimates are useful for rough local observability, not billing.
        estimated_input_tokens = max(1, sum(len(str(item["text"])) for item in matches) // 4 + len(question) // 4)
        estimated_output_tokens = max(1, len(answer) // 4)
        logger.info(
            "chat completed",
            extra={"event": "chat_completed", "provider": mode, "duration_ms": elapsed_ms,
                   "input_tokens": estimated_input_tokens, "output_tokens": estimated_output_tokens},
        )
        return {
            "answer": answer,
            "language": language,
            "sources": sources,
            "abstained": abstained,
            "generation_mode": mode,
            "latency_ms": elapsed_ms,
            "personalization": {
                "profile_id": profile_id,
                "age_months": profile["age_months"] if profile else None,
                "preferred_language": profile["preferred_language"] if profile else None,
                "used_for_retrieval_and_prompt": bool(profile),
            } if profile else None,
            "estimated_input_tokens": estimated_input_tokens,
            "estimated_output_tokens": estimated_output_tokens,
        }


def create_provider_fallback(language: str, question: str, context: list[str], profile_text: str) -> str:
    from app.models.provider import ExtractiveProvider

    return ExtractiveProvider().generate(question, language, context, profile_text)
