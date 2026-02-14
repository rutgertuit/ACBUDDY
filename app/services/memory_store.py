"""Persistent memory store — learns from past research for cross-session awareness.

Stores memories in GCS at memory/memory.json. Uses Gemini embeddings for recall
with keyword fallback.
"""

import json
import logging
import math
import secrets
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

MEMORY_BLOB = "memory/memory.json"

# Module-level cache for GCS loads (5 min TTL)
_cache: dict = {"data": None, "ts": 0}
_CACHE_TTL = 300


@dataclass
class MemoryEntry:
    id: str = ""
    type: str = ""  # finding, pattern, fact, recommendation
    content: str = ""
    source_job_id: str = ""
    source_query: str = ""
    tags: list[str] = field(default_factory=list)
    created_at: str = ""


@dataclass
class MemoryStore:
    entries: list[MemoryEntry] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"entries": [asdict(e) for e in self.entries]}

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryStore":
        store = cls()
        for edata in data.get("entries", []):
            store.entries.append(MemoryEntry(**edata))
        return store


def load_memory(bucket_name: str, use_cache: bool = True) -> MemoryStore:
    """Load memory from GCS, or return empty store. Uses in-memory cache."""
    if not bucket_name:
        return MemoryStore()
    if use_cache and _cache["data"] and (time.time() - _cache["ts"]) < _CACHE_TTL:
        return _cache["data"]
    try:
        from google.cloud import storage

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(MEMORY_BLOB)
        if not blob.exists():
            return MemoryStore()
        data = json.loads(blob.download_as_text())
        store = MemoryStore.from_dict(data)
        _cache["data"] = store
        _cache["ts"] = time.time()
        return store
    except Exception:
        logger.exception("Failed to load memory store")
        return MemoryStore()


def save_memory(store: MemoryStore, bucket_name: str) -> None:
    """Save memory to GCS. Invalidates cache."""
    if not bucket_name:
        return
    try:
        from google.cloud import storage

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(MEMORY_BLOB)
        blob.upload_from_string(
            json.dumps(store.to_dict(), indent=2),
            content_type="application/json",
        )
        # Invalidate cache on save
        _cache["data"] = None
        _cache["ts"] = 0
        logger.info("Saved memory store: %d entries", len(store.entries))
    except Exception:
        logger.exception("Failed to save memory store")


def add_memories(store: MemoryStore, entries: list[dict], job_id: str, query: str) -> int:
    """Add new memory entries to the store.

    Args:
        store: MemoryStore to add to (modified in-place).
        entries: List of dicts with type, content, tags.
        job_id: Source job ID.
        query: Source research query.

    Returns:
        Number of entries added.
    """
    now = datetime.now(timezone.utc).isoformat()
    added = 0
    for entry in entries:
        content = entry.get("content", "").strip()
        if not content:
            continue
        # Dedup: skip if very similar content already exists
        if any(e.content.lower() == content.lower() for e in store.entries):
            continue
        store.entries.append(MemoryEntry(
            id=secrets.token_hex(6),
            type=entry.get("type", "finding"),
            content=content,
            source_job_id=job_id,
            source_query=query,
            tags=entry.get("tags", []),
            created_at=now,
        ))
        added += 1
    return added


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _recall_with_keywords(store: MemoryStore, query: str, top_k: int = 5) -> list[dict]:
    """Recall using keyword overlap (fallback method)."""
    if not store.entries:
        return []

    query_words = set(query.lower().split())

    scored = []
    for entry in store.entries:
        content_words = set(entry.content.lower().split())
        tag_words = set(t.lower() for t in entry.tags)
        score = len(query_words & content_words) + 2 * len(query_words & tag_words)
        if score > 0:
            scored.append((score, entry))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [asdict(entry) for _, entry in scored[:top_k]]


def _recall_with_embeddings(store: MemoryStore, query: str, top_k: int = 5) -> list[dict]:
    """Recall using Gemini text-embedding-004 for semantic similarity."""
    import os
    from google import genai

    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        raise RuntimeError("No GOOGLE_API_KEY for embeddings")

    client = genai.Client(api_key=api_key)

    # Embed query
    q_resp = client.models.embed_content(model="text-embedding-004", content=query)
    q_vec = q_resp.embeddings[0].values

    # Cap entries to avoid excessive embedding costs
    entries = store.entries[:100]

    # Batch embed all memory contents
    contents = [e.content for e in entries]
    c_resp = client.models.embed_content(model="text-embedding-004", content=contents)

    # Score by cosine similarity
    scored = []
    for i, emb in enumerate(c_resp.embeddings):
        sim = _cosine_similarity(q_vec, emb.values)
        if sim > 0.3:  # threshold
            scored.append((sim, entries[i]))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [asdict(entry) for _, entry in scored[:top_k]]


def recall(store: MemoryStore, query: str, top_k: int = 5) -> list[dict]:
    """Recall relevant memories — tries embedding similarity, falls back to keywords.

    Args:
        store: MemoryStore to search.
        query: Search query.
        top_k: Max number of results.

    Returns:
        List of memory entry dicts, most relevant first.
    """
    if not store.entries:
        return []

    try:
        results = _recall_with_embeddings(store, query, top_k)
        if results:
            logger.info("Embedding recall returned %d results", len(results))
            return results
    except Exception:
        logger.warning("Embedding recall failed, falling back to keywords")

    return _recall_with_keywords(store, query, top_k)


def delete_memory(store: MemoryStore, memory_id: str) -> bool:
    """Delete a memory entry by ID. Returns True if found and deleted."""
    for i, entry in enumerate(store.entries):
        if entry.id == memory_id:
            store.entries.pop(i)
            return True
    return False


def get_memory_stats(store: MemoryStore) -> dict:
    """Get summary stats of the memory store."""
    type_counts = {}
    for e in store.entries:
        type_counts[e.type] = type_counts.get(e.type, 0) + 1
    return {
        "total_entries": len(store.entries),
        "types": type_counts,
    }
