from __future__ import annotations

from app.db import Store
from app.rag.embedder import Embedder, cosine_similarity, lexical_terms
from app.rag.language import detect_language


def search(
    store: Store, owner_id: str, query: str, embedder: Embedder,
    top_k: int = 4, min_score: float = 0.08, profile_text: str = "",
    language: str | None = None,
) -> list[dict[str, object]]:
    query_vector = embedder.embed(query)
    query_terms = lexical_terms(query)
    profile_vector = embedder.embed(profile_text) if profile_text.strip() else None
    ranked: list[dict[str, object]] = []
    for chunk in store.retrieve_chunks(owner_id):
        if chunk["embedding_backend"] != embedder.name:
            continue
        score = cosine_similarity(query_vector, chunk["vector"])
        chunk_terms = lexical_terms(str(chunk["text"]))
        overlap = len(query_terms & chunk_terms) / max(1, len(query_terms))
        # Character n-grams tolerate inflections and typos but can create false hits;
        # damp their contribution when no lexical/concept evidence overlaps.
        if overlap:
            score = 0.55 * score + 0.45 * overlap
        elif embedder.name == "hashing-ngram":
            # The lightweight fallback is lexical, so subword-only matches are too weak
            # to justify citing a document for an unrelated query.
            score *= 0.15
        profile_score = cosine_similarity(profile_vector, chunk["vector"]) if profile_vector else 0.0
        document_language = detect_language(str(chunk["text"]))
        language_bonus = 0.04 if language and document_language == language else 0.0
        final_score = score + (0.08 * profile_score if profile_text else 0.0) + language_bonus
        if score >= min_score:
            ranked.append({**chunk, "score": round(final_score, 6), "query_score": round(score, 6)})
    ranked.sort(key=lambda item: float(item["score"]), reverse=True)
    return ranked[:top_k]
