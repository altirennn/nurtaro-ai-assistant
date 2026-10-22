from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse

from app.config import Settings
from app.logging_config import configure_logging
from app.rag.ingest import IngestError, extract_text, index_pending
from app.schemas import ChatRequest, ChatResponse, DocumentResponse, EvaluationResponse, IndexRequest, ProfileCreate, ProfileResponse
from app.security import Principal, principal_dependency
from app.service import AssistantService
from app.vision.routes import create_video_router

logger = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    service: AssistantService | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    if service is None:
        service = AssistantService.create(settings)
    auth = principal_dependency(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield

    application = FastAPI(
        title="Nurtaro AI — Bilingual Parenting Assistant",
        version="0.1.0",
        description="Демонстрационный RU/KK RAG-ассистент с синтетическими материалами.",
        lifespan=lifespan,
    )
    application.state.settings = settings
    application.state.service = service
    application.state.started_at = time.time()
    application.include_router(create_video_router(auth))

    @application.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        return response

    @application.exception_handler(Exception)
    async def unhandled_error(_: Request, exc: Exception):
        logger.error("request failed", extra={"event": "request_error", "status_code": 500}, exc_info=exc)
        return JSONResponse(status_code=500, content={"detail": "Внутренняя ошибка сервера"})

    @application.get("/health", tags=["health"])
    def health() -> dict[str, Any]:
        try:
            with service.store.connect() as conn:
                conn.execute("SELECT 1").fetchone()
            database_status = "ok"
        except Exception:
            database_status = "error"
        return {
            "status": "ok" if database_status == "ok" else "degraded",
            "database": database_status,
            "embedding_backend": service.embedder.name,
            "generation_backend": service.provider.name,
            "uptime_seconds": round(time.time() - application.state.started_at, 1),
        }

    @application.post("/chat", response_model=ChatResponse, tags=["chat"])
    def chat(payload: ChatRequest, principal: Principal = Depends(auth)):
        try:
            return service.answer(principal.owner_id, payload.question, payload.profile_id)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @application.post("/documents/upload", response_model=DocumentResponse, status_code=201, tags=["documents"])
    async def upload_document(request: Request, file: UploadFile = File(...), principal: Principal = Depends(auth)):
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > settings.max_upload_bytes + 65536:
                    raise HTTPException(status_code=413, detail="Запрос превышает лимит размера")
            except ValueError as exc:
                raise HTTPException(status_code=400, detail="Некорректный Content-Length") from exc
        if file.size is not None and file.size > settings.max_upload_bytes:
            raise HTTPException(status_code=413, detail="Файл превышает настроенный лимит размера")
        raw_name = (file.filename or "document").replace("\\", "/").split("/")[-1]
        if not raw_name or len(raw_name) > 180:
            raise HTTPException(status_code=400, detail="Некорректное имя файла")
        data = await file.read(settings.max_upload_bytes + 1)
        await file.close()
        try:
            text, source_type, removed_lines = extract_text(raw_name, data, settings)
        except IngestError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        try:
            doc_id = service.store.insert_document(
                principal.owner_id, raw_name[:180], source_type, text,
                removed_lines=removed_lines,
            )
        except Exception as exc:
            logger.error("document save failed", extra={"event": "document_save_error"})
            raise HTTPException(status_code=500, detail="Не удалось сохранить документ") from exc
        logger.info("document uploaded", extra={"event": "document_uploaded", "document_id": doc_id})
        return {
            "id": doc_id, "title": raw_name[:180], "source_type": source_type,
            "indexed_chunks": 0, "is_shared": False,
            "created_at": service.store.get_document(doc_id, principal.owner_id)["created_at"],
        }

    @application.post("/documents/index", tags=["documents"])
    def index_documents(payload: IndexRequest, principal: Principal = Depends(auth)):
        if payload.document_id and not service.store.get_document(payload.document_id, principal.owner_id):
            raise HTTPException(status_code=404, detail="Документ не найден")
        embedder = service.embedder
        indexed = index_pending(service.store, principal.owner_id, embedder, payload.document_id, force=payload.force)
        return {"indexed": indexed, "embedding_backend": embedder.name}

    @application.get("/documents", response_model=list[DocumentResponse], tags=["documents"])
    def list_documents(principal: Principal = Depends(auth)):
        return service.store.list_documents(principal.owner_id)

    @application.post("/profiles", response_model=ProfileResponse, status_code=201, tags=["profiles"])
    def create_profile(payload: ProfileCreate, principal: Principal = Depends(auth)):
        return service.store.create_profile(principal.owner_id, payload.model_dump())

    @application.get("/profiles", response_model=list[ProfileResponse], tags=["profiles"])
    def list_profiles(principal: Principal = Depends(auth)):
        return service.store.list_profiles(principal.owner_id)

    @application.get("/profiles/{profile_id}", response_model=ProfileResponse, tags=["profiles"])
    def get_profile(profile_id: str, principal: Principal = Depends(auth)):
        profile = service.store.get_profile(profile_id, principal.owner_id)
        if not profile:
            raise HTTPException(status_code=404, detail="Профиль не найден")
        return profile

    @application.post("/evaluate", response_model=EvaluationResponse, tags=["evaluation"])
    def evaluate(principal: Principal = Depends(auth)):
        del principal
        try:
            from evaluation.run_eval import run_evaluation

            summary, json_path, csv_path = run_evaluation(service)
        except Exception as exc:
            logger.error("evaluation failed", extra={"event": "evaluation_error"}, exc_info=exc)
            raise HTTPException(status_code=500, detail="Оценивание не удалось выполнить") from exc
        return {
            "run_id": Path(json_path).stem,
            "summary": summary,
            "results_json": "evaluation/results.json",
            "results_csv": "evaluation/results.csv",
        }

    return application


configure_logging()
app = create_app()
