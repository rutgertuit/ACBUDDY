"""In-memory thread-safe job state tracker for UI-triggered research."""

import secrets
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class JobInfo:
    job_id: str
    query: str
    depth: str
    status: JobStatus = JobStatus.PENDING
    phase: str = ""
    result_url: str = ""
    error: str = ""
    elevenlabs_doc_id: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: str = ""
    # Structured progress for DEEP pipeline
    study_plan: list = field(default_factory=list)
    study_progress: list = field(default_factory=list)  # [{title, status, rounds}]
    current_step: str = ""  # e.g. "study_2", "synthesis", "refinement"


_jobs: dict[str, JobInfo] = {}
_lock = threading.Lock()


def create_job(query: str, depth: str) -> str:
    """Create a new job and return its ID (12-char hex)."""
    job_id = secrets.token_hex(6)
    job = JobInfo(job_id=job_id, query=query, depth=depth)
    with _lock:
        _jobs[job_id] = job
    return job_id


def get_job(job_id: str) -> Optional[JobInfo]:
    """Return job info or None if not found."""
    with _lock:
        return _jobs.get(job_id)


def update_job(job_id: str, **kwargs) -> None:
    """Update fields on an existing job."""
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        for key, value in kwargs.items():
            if hasattr(job, key):
                setattr(job, key, value)


def count_active_jobs() -> int:
    """Return the number of jobs currently in PENDING or RUNNING status."""
    with _lock:
        return sum(
            1 for j in _jobs.values()
            if j.status in (JobStatus.PENDING, JobStatus.RUNNING)
        )
