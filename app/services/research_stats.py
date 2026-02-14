"""Thread-local research statistics tracking.

Tracks web searches, pages read, etc. per research job.
Tool functions call increment() which auto-pushes stats to the job tracker.
"""

import threading

_local = threading.local()


def init_stats(job_id: str = "") -> None:
    """Initialize stats counters for the current thread."""
    _local.stats = {
        "web_searches": 0,
        "urls_fetched": 0,
        "pages_read": 0,
        "news_searches": 0,
        "news_articles": 0,
        "grok_queries": 0,
        "reasoning_calls": 0,
    }
    _local.job_id = job_id


def increment(key: str, amount: int = 1) -> None:
    """Increment a stat counter and push to job tracker."""
    stats = getattr(_local, "stats", None)
    if stats is None:
        return
    stats[key] = stats.get(key, 0) + amount
    _push_stats()


def get_stats() -> dict:
    """Return current stats dict (copy)."""
    stats = getattr(_local, "stats", None)
    return dict(stats) if stats else {}


def _push_stats() -> None:
    """Push current stats snapshot to the job tracker (if job_id set)."""
    job_id = getattr(_local, "job_id", None)
    stats = getattr(_local, "stats", None)
    if job_id and stats:
        from app.services.job_tracker import update_job

        update_job(job_id, research_stats=dict(stats))


def compute_human_hours(
    stats: dict,
    num_studies: int = 0,
    num_qa_clusters: int = 0,
    depth: str = "STANDARD",
) -> dict:
    """Compute estimated human equivalent effort in minutes.

    Based on average professional research times:
    - Web search + evaluate results: ~8 min per search
    - Reading a full web page critically: ~5 min per page
    - Reading a news article: ~3 min per article
    - Deep analytical reasoning: ~15 min per analysis
    - Per-study research cycle: ~45 min per study
    - Writing a synthesis report: 30 min (standard) or 120 min (deep)
    - Q&A preparation per cluster: ~30 min
    """
    searching = (
        stats.get("web_searches", 0)
        + stats.get("news_searches", 0)
        + stats.get("grok_queries", 0)
    ) * 8
    reading = stats.get("pages_read", 0) * 5 + stats.get("news_articles", 0) * 3
    analyzing = stats.get("reasoning_calls", 0) * 15 + num_studies * 45
    writing = 120 if depth.upper() == "DEEP" else 30
    qa_prep = num_qa_clusters * 30

    total_minutes = searching + reading + analyzing + writing + qa_prep

    return {
        "searching_min": searching,
        "reading_min": reading,
        "analyzing_min": analyzing,
        "writing_min": writing,
        "qa_prep_min": qa_prep,
        "total_minutes": total_minutes,
        "total_hours": round(total_minutes / 60, 1),
    }
