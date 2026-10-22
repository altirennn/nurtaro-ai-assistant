from __future__ import annotations

from io import BytesIO
from pathlib import Path

from pypdf import PdfReader

from app.config import Settings
from app.db import Store
from app.rag.chunking import split_into_chunks
from app.rag.embedder import Embedder
from app.rag.sanitize import sanitize_document

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}


class IngestError(ValueError):
    pass


def extract_text(filename: str, content: bytes, settings: Settings) -> tuple[str, str, int]:
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise IngestError("Поддерживаются только PDF, TXT и Markdown")
    if len(content) > settings.max_upload_bytes:
        raise IngestError(f"Файл превышает лимит {settings.max_upload_bytes} байт")
    if suffix == ".pdf":
        try:
            reader = PdfReader(BytesIO(content), strict=True)
            if len(reader.pages) > settings.max_pdf_pages:
                raise IngestError(f"PDF содержит больше {settings.max_pdf_pages} страниц")
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except IngestError:
            raise
        except Exception as exc:
            raise IngestError("PDF не удалось прочитать") from exc
        source_type = "pdf"
    else:
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise IngestError("Текстовый файл должен быть в UTF-8") from exc
        source_type = "markdown" if suffix in {".md", ".markdown"} else "text"
    cleaned, removed = sanitize_document(text)
    if len(cleaned.strip()) < 20:
        raise IngestError("В файле мало извлекаемого текста")
    return cleaned, source_type, removed


def index_pending(
    store: Store, owner_id: str, embedder: Embedder,
    document_id: str | None = None, force: bool = False,
) -> list[dict[str, object]]:
    indexed: list[dict[str, object]] = []
    documents = store.pending_documents(owner_id, document_id)
    if force:
        if document_id:
            selected = store.get_document(document_id, owner_id)
            documents = [selected] if selected else []
        else:
            visible = store.list_documents(owner_id)
            documents = [doc for row in visible if (doc := store.get_document(str(row["id"]), owner_id))]
    for document in documents:
        pieces = split_into_chunks(str(document["content"]))
        vectors = [(piece, embedder.embed(piece), embedder.name) for piece in pieces]
        count = store.replace_chunks(str(document["id"]), vectors)
        indexed.append({"document_id": document["id"], "title": document["title"], "indexed_chunks": count})
    return indexed
