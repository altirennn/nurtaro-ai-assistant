from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    is_shared INTEGER NOT NULL DEFAULT 0,
                    seed_key TEXT UNIQUE,
                    injection_lines_removed INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_documents_owner ON documents(owner_id);
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    ordinal INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    vector_json TEXT NOT NULL,
                    embedding_backend TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(document_id, ordinal)
                );
                CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
                CREATE TABLE IF NOT EXISTS profiles (
                    id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    age_months INTEGER NOT NULL CHECK(age_months BETWEEN 0 AND 35),
                    preferred_language TEXT NOT NULL CHECK(preferred_language IN ('ru','kk')),
                    interests_json TEXT NOT NULL,
                    materials_json TEXT NOT NULL,
                    parent_preferences TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_profiles_owner ON profiles(owner_id);
                """
            )

    def insert_document(
        self, owner_id: str, title: str, source_type: str, content: str,
        is_shared: bool = False, seed_key: str | None = None, removed_lines: int = 0,
    ) -> str:
        doc_id = str(uuid.uuid4())
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO documents(id,owner_id,title,source_type,content,is_shared,seed_key,injection_lines_removed,created_at) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (doc_id, owner_id, title, source_type, content, int(is_shared), seed_key, removed_lines, now_iso()),
            )
        return doc_id

    def seed_document(self, seed_key: str, title: str, source_type: str, content: str) -> str:
        with self.connect() as conn:
            row = conn.execute("SELECT id FROM documents WHERE seed_key=?", (seed_key,)).fetchone()
            if row:
                return str(row["id"])
        return self.insert_document("__shared__", title, source_type, content, True, seed_key)

    def list_documents(self, owner_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT d.id,d.title,d.source_type,d.is_shared,d.created_at,COUNT(c.id) AS indexed_chunks "
                "FROM documents d LEFT JOIN chunks c ON c.document_id=d.id "
                "WHERE d.owner_id=? OR d.is_shared=1 GROUP BY d.id ORDER BY d.created_at DESC",
                (owner_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_document(self, document_id: str, owner_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE id=? AND (owner_id=? OR is_shared=1)",
                (document_id, owner_id),
            ).fetchone()
        return dict(row) if row else None

    def pending_documents(self, owner_id: str, document_id: str | None = None) -> list[dict[str, Any]]:
        query = (
            "SELECT d.* FROM documents d LEFT JOIN chunks c ON c.document_id=d.id "
            "WHERE (d.owner_id=? OR d.is_shared=1) AND c.id IS NULL"
        )
        params: list[Any] = [owner_id]
        if document_id:
            query += " AND d.id=?"
            params.append(document_id)
        query += " GROUP BY d.id ORDER BY d.created_at"
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def replace_chunks(self, document_id: str, chunks: list[tuple[str, list[float], str]]) -> int:
        with self.connect() as conn:
            conn.execute("DELETE FROM chunks WHERE document_id=?", (document_id,))
            conn.executemany(
                "INSERT INTO chunks(id,document_id,ordinal,text,vector_json,embedding_backend,created_at) VALUES(?,?,?,?,?,?,?)",
                [
                    (str(uuid.uuid4()), document_id, ordinal, text, json.dumps(vector), backend, now_iso())
                    for ordinal, (text, vector, backend) in enumerate(chunks)
                ],
            )
        return len(chunks)

    def retrieve_chunks(self, owner_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT c.id AS chunk_id,c.ordinal,c.text,c.vector_json,c.embedding_backend,d.id AS document_id,"
                "d.owner_id,d.title,d.is_shared FROM chunks c JOIN documents d ON d.id=c.document_id "
                "WHERE d.owner_id=? OR d.is_shared=1",
                (owner_id,),
            ).fetchall()
        result = []
        for row in rows:
            entry = dict(row)
            entry["vector"] = json.loads(entry.pop("vector_json"))
            result.append(entry)
        return result

    def create_profile(self, owner_id: str, values: dict[str, Any]) -> dict[str, Any]:
        profile_id = str(uuid.uuid4())
        stamp = now_iso()
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO profiles(id,owner_id,age_months,preferred_language,interests_json,materials_json,parent_preferences,created_at,updated_at) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (profile_id, owner_id, values["age_months"], values["preferred_language"],
                 json.dumps(values["interests"], ensure_ascii=False), json.dumps(values["available_materials"], ensure_ascii=False),
                 values["parent_preferences"], stamp, stamp),
            )
        return self.get_profile(profile_id, owner_id)  # type: ignore[return-value]

    def get_profile(self, profile_id: str, owner_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM profiles WHERE id=? AND owner_id=?", (profile_id, owner_id)).fetchone()
        if not row:
            return None
        result = dict(row)
        result.pop("owner_id", None)
        result["interests"] = json.loads(result.pop("interests_json"))
        result["available_materials"] = json.loads(result.pop("materials_json"))
        return result

    def list_profiles(self, owner_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT id FROM profiles WHERE owner_id=? ORDER BY created_at DESC", (owner_id,)).fetchall()
        return [profile for row in rows if (profile := self.get_profile(str(row["id"]), owner_id))]
