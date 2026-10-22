from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Callable

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from app.rag.ingest import index_pending
from app.rag.sanitize import sanitize_document
from app.schemas import DocumentResponse, VideoAnalysisResponse
from app.security import Principal, principal_dependency
from app.vision.transcripts import transcript_to_document
from app.vision.video import VideoInputError, VisionUnavailable, analyze_video

logger = logging.getLogger(__name__)
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
TRANSCRIPT_SUFFIXES = {".txt", ".vtt", ".srt"}


def _safe_name(upload: UploadFile) -> tuple[str, str]:
    name = (upload.filename or "").replace("\\", "/").split("/")[-1]
    suffix = Path(name).suffix.lower()
    return name[:180], suffix


def create_video_router(auth_dependency: Callable[..., Principal]) -> APIRouter:
    router = APIRouter(tags=["video"])

    @router.post("/vision/analyze", response_model=VideoAnalysisResponse, tags=["vision"])
    async def analyze_uploaded_video(
        request: Request,
        file: UploadFile = File(...),
        principal: Principal = Depends(auth_dependency),
    ):
        del principal  # Authentication is required; this endpoint stores no user-specific result.
        settings = request.app.state.settings
        _, suffix = _safe_name(file)
        if suffix not in VIDEO_SUFFIXES:
            await file.close()
            raise HTTPException(status_code=422, detail="Поддерживаются MP4, MOV, MKV, WEBM и AVI")
        content_length = request.headers.get("content-length")
        if content_length and content_length.isdigit() and int(content_length) > settings.vision_max_upload_bytes + 65536:
            await file.close()
            raise HTTPException(status_code=413, detail="Запрос превышает лимит размера")
        if file.size is not None and file.size > settings.vision_max_upload_bytes:
            await file.close()
            raise HTTPException(status_code=413, detail="Видео превышает настроенный лимит размера")
        raw = await file.read(settings.vision_max_upload_bytes + 1)
        await file.close()
        if len(raw) > settings.vision_max_upload_bytes:
            raise HTTPException(status_code=413, detail="Видео превышает настроенный лимит размера")
        if not raw:
            raise HTTPException(status_code=422, detail="Видео пустое")
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(prefix="nurtaro-video-", suffix=suffix, delete=False) as handle:
                handle.write(raw)
                temp_path = Path(handle.name)
            raw = b""
            result = await run_in_threadpool(analyze_video, temp_path, settings)
        except VideoInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except VisionUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            logger.error("video analysis failed", extra={"event": "video_analysis_error"})
            raise HTTPException(status_code=500, detail="Не удалось обработать видео") from exc
        finally:
            if temp_path and temp_path.exists():
                temp_path.unlink(missing_ok=True)
        logger.info(
            "video analyzed",
            extra={"event": "video_analyzed", "duration_ms": result["processing_ms"], "sampled_frames": result["sampled_frames"]},
        )
        result.pop("processing_ms", None)
        return result

    @router.post("/videos/transcripts/upload", response_model=DocumentResponse, status_code=201)
    async def upload_video_transcript(
        request: Request,
        language: str,
        file: UploadFile = File(...),
        principal: Principal = Depends(auth_dependency),
    ):
        settings = request.app.state.settings
        name, suffix = _safe_name(file)
        if language not in {"ru", "kk"}:
            await file.close()
            raise HTTPException(status_code=422, detail="Язык должен быть ru или kk")
        if suffix not in TRANSCRIPT_SUFFIXES:
            await file.close()
            raise HTTPException(status_code=422, detail="Загрузите текстовые субтитры TXT, VTT или SRT")
        if file.size is not None and file.size > settings.max_upload_bytes:
            await file.close()
            raise HTTPException(status_code=413, detail="Файл транскрипта превышает настроенный лимит")
        raw = await file.read(settings.max_upload_bytes + 1)
        await file.close()
        if len(raw) > settings.max_upload_bytes:
            raise HTTPException(status_code=413, detail="Файл транскрипта превышает настроенный лимит")
        try:
            source_text = raw.decode("utf-8-sig")
            content = transcript_to_document(Path(name).stem, language, source_text)
        except UnicodeDecodeError as exc:
            raise HTTPException(status_code=422, detail="Субтитры должны быть в UTF-8") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        content, removed_lines = sanitize_document(content)
        if len(content) > 100_000:
            raise HTTPException(status_code=413, detail="Транскрипт превышает лимит 100 000 символов")
        service = request.app.state.service
        document_id = service.store.insert_document(
            principal.owner_id, Path(name).stem[:180], "video_transcript", content, removed_lines=removed_lines,
        )
        indexed = index_pending(service.store, principal.owner_id, service.embedder, document_id)
        stored = service.store.get_document(document_id, principal.owner_id)
        logger.info("video transcript indexed", extra={"event": "video_transcript_indexed", "document_id": document_id})
        return {
            "id": document_id,
            "title": Path(name).stem[:180],
            "source_type": "video_transcript",
            "indexed_chunks": indexed[0]["indexed_chunks"] if indexed else 0,
            "is_shared": False,
            "created_at": stored["created_at"],
        }

    return router
