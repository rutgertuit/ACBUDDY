"""Watch store — manages research watches (topic monitors) in GCS."""

import hashlib
import json
import logging
import secrets
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

WATCHES_PREFIX = "watches/"


@dataclass
class WatchUpdate:
    checked_at: str = ""
    findings_hash: str = ""
    summary: str = ""
    changed: bool = False


@dataclass
class ResearchWatch:
    id: str = ""
    query: str = ""
    interval_hours: int = 24
    created_at: str = ""
    last_checked: str = ""
    last_findings_hash: str = ""
    history: list[dict] = field(default_factory=list)  # list of WatchUpdate dicts
    active: bool = True

    def is_due(self) -> bool:
        """Check if this watch is due for a check."""
        if not self.last_checked:
            return True
        try:
            last = datetime.fromisoformat(self.last_checked)
            now = datetime.now(timezone.utc)
            hours_since = (now - last).total_seconds() / 3600
            return hours_since >= self.interval_hours
        except Exception:
            return True


def _watch_blob(watch_id: str) -> str:
    return f"{WATCHES_PREFIX}{watch_id}.json"


def create_watch(query: str, interval_hours: int, bucket_name: str) -> ResearchWatch:
    """Create a new research watch and save to GCS."""
    watch = ResearchWatch(
        id=secrets.token_hex(6),
        query=query,
        interval_hours=max(1, interval_hours),
        created_at=datetime.now(timezone.utc).isoformat(),
        active=True,
    )
    _save_watch(watch, bucket_name)
    logger.info("Created watch %s: %s (every %dh)", watch.id, query[:60], interval_hours)
    return watch


def get_watch(watch_id: str, bucket_name: str) -> ResearchWatch | None:
    """Fetch a single watch from GCS."""
    if not bucket_name:
        return None
    try:
        from google.cloud import storage

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(_watch_blob(watch_id))
        if not blob.exists():
            return None
        data = json.loads(blob.download_as_text())
        return ResearchWatch(**data)
    except Exception:
        logger.exception("Failed to fetch watch %s", watch_id)
        return None


def list_watches(bucket_name: str) -> list[ResearchWatch]:
    """List all watches from GCS."""
    if not bucket_name:
        return []
    try:
        from google.cloud import storage

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blobs = list(bucket.list_blobs(prefix=WATCHES_PREFIX))

        watches = []
        for blob in blobs:
            if not blob.name.endswith(".json"):
                continue
            try:
                data = json.loads(blob.download_as_text())
                watches.append(ResearchWatch(**data))
            except Exception:
                logger.warning("Failed to parse watch blob %s", blob.name)
        watches.sort(key=lambda w: w.created_at, reverse=True)
        return watches
    except Exception:
        logger.exception("Failed to list watches")
        return []


def delete_watch(watch_id: str, bucket_name: str) -> bool:
    """Delete a watch from GCS."""
    if not bucket_name:
        return False
    try:
        from google.cloud import storage

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(_watch_blob(watch_id))
        if blob.exists():
            blob.delete()
            logger.info("Deleted watch %s", watch_id)
            return True
        return False
    except Exception:
        logger.exception("Failed to delete watch %s", watch_id)
        return False


def record_check(watch: ResearchWatch, findings: str, bucket_name: str) -> WatchUpdate:
    """Record a check result for a watch.

    Compares findings hash to detect changes. Updates watch in GCS.
    """
    now = datetime.now(timezone.utc).isoformat()
    findings_hash = hashlib.sha256(findings.encode()).hexdigest()[:16]
    changed = findings_hash != watch.last_findings_hash and watch.last_findings_hash != ""

    update = WatchUpdate(
        checked_at=now,
        findings_hash=findings_hash,
        summary=findings[:500] if changed else "No significant changes detected.",
        changed=changed,
    )

    watch.last_checked = now
    watch.last_findings_hash = findings_hash
    watch.history.append(asdict(update))
    # Keep only last 20 history entries
    watch.history = watch.history[-20:]

    _save_watch(watch, bucket_name)
    logger.info("Watch %s checked: changed=%s hash=%s", watch.id, changed, findings_hash)
    return update


def get_due_watches(bucket_name: str) -> list[ResearchWatch]:
    """Get all watches that are due for checking."""
    watches = list_watches(bucket_name)
    return [w for w in watches if w.active and w.is_due()]


def _save_watch(watch: ResearchWatch, bucket_name: str) -> None:
    """Save a watch to GCS."""
    if not bucket_name:
        return
    try:
        from google.cloud import storage

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(_watch_blob(watch.id))
        blob.upload_from_string(
            json.dumps(asdict(watch), indent=2),
            content_type="application/json",
        )
    except Exception:
        logger.exception("Failed to save watch %s", watch.id)
