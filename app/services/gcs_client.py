"""Generate an HTML results page and upload it to Google Cloud Storage."""

import html
import json
import logging
import re
from datetime import datetime, timezone

from app.models.research_result import ResearchResult

logger = logging.getLogger(__name__)


def _md_to_html(text: str) -> str:
    """Minimal markdown-to-HTML conversion for research content.

    Handles headers (#–####), bold (**), italic (*), bullet lists, and paragraphs.
    """
    if not text:
        return ""

    lines = text.split("\n")
    out: list[str] = []
    in_list = False

    for line in lines:
        stripped = line.strip()

        # Blank line closes list and adds paragraph break
        if not stripped:
            if in_list:
                out.append("</ul>")
                in_list = False
            out.append("")
            continue

        # Headers
        m = re.match(r"^(#{1,4})\s+(.*)", stripped)
        if m:
            if in_list:
                out.append("</ul>")
                in_list = False
            level = len(m.group(1))
            content = _inline_format(html.escape(m.group(2)))
            out.append(f"<h{level + 1}>{content}</h{level + 1}>")
            continue

        # Bullet list items
        if re.match(r"^[-*]\s+", stripped):
            if not in_list:
                out.append("<ul>")
                in_list = True
            content = _inline_format(html.escape(re.sub(r"^[-*]\s+", "", stripped)))
            out.append(f"  <li>{content}</li>")
            continue

        # Regular paragraph line
        if in_list:
            out.append("</ul>")
            in_list = False
        content = _inline_format(html.escape(stripped))
        out.append(f"<p>{content}</p>")

    if in_list:
        out.append("</ul>")

    return "\n".join(out)


def _inline_format(text: str) -> str:
    """Convert bold (**text**) and italic (*text*) markers to HTML."""
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
    return text


_CSS = """\
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    line-height: 1.7; color: #1a1a1a; max-width: 900px; margin: 0 auto;
    padding: 2rem 1.5rem; background: #fafafa;
}
header { border-bottom: 3px solid #2563eb; padding-bottom: 1rem; margin-bottom: 2rem; }
header h1 { font-size: 1.8rem; color: #1e40af; }
header p { color: #6b7280; font-size: 0.9rem; margin-top: 0.4rem; }
section { margin-bottom: 2.5rem; }
h2 { font-size: 1.4rem; color: #1e40af; margin-bottom: 0.8rem; border-bottom: 1px solid #e5e7eb; padding-bottom: 0.3rem; }
h3 { font-size: 1.15rem; color: #374151; margin: 1.2rem 0 0.5rem; }
h4 { font-size: 1rem; color: #4b5563; margin: 0.8rem 0 0.4rem; }
p { margin-bottom: 0.6rem; }
ul { margin: 0.4rem 0 0.8rem 1.5rem; }
li { margin-bottom: 0.3rem; }
strong { color: #111827; }
.study { background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 1.2rem 1.5rem; margin-bottom: 1.2rem; }
.cluster { background: #f0f9ff; border-left: 4px solid #3b82f6; padding: 1rem 1.2rem; margin-bottom: 1rem; border-radius: 0 6px 6px 0; }
.quality-card { background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; padding: 1.2rem 1.5rem; margin-bottom: 1.5rem; }
.quality-card h3 { color: #166534; margin-bottom: 0.6rem; }
.score-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 0.8rem; margin: 0.8rem 0; }
.score-item { background: #fff; border: 1px solid #e5e7eb; border-radius: 6px; padding: 0.6rem 0.8rem; }
.score-item .label { font-size: 0.85rem; color: #6b7280; text-transform: capitalize; }
.score-item .value { font-size: 1.3rem; font-weight: 600; color: #1e40af; }
"""


def generate_html(result: ResearchResult, query: str, depth: str) -> str:
    """Build a self-contained HTML page from a ResearchResult."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    num_studies = len(result.studies) if result.studies else 0

    meta_parts = [
        f"Query: {html.escape(query)}",
        f"Depth: {html.escape(depth.upper())}",
    ]
    if num_studies:
        meta_parts.append(f"Studies: {num_studies}")
    meta_parts.append(f"Generated: {now}")

    sections: list[str] = []

    # DEEP pipeline sections
    if depth.upper() == "DEEP":
        # Quality scores (if refinement loop ran)
        if result.synthesis_score > 0:
            score_items = ""
            for dim, val in result.synthesis_scores.items():
                score_items += (
                    f'<div class="score-item"><div class="label">{html.escape(dim)}</div>'
                    f'<div class="value">{val}/10</div></div>'
                )
            refinement_note = ""
            if result.refinement_rounds > 0:
                refinement_note = (
                    f"<p>Refined through {result.refinement_rounds} evaluation "
                    f"{'round' if result.refinement_rounds == 1 else 'rounds'} "
                    f"to address identified gaps and strengthen evidence.</p>"
                )
            sections.append(
                f'<section id="quality"><div class="quality-card">'
                f"<h3>Synthesis Quality: {result.synthesis_score:.1f}/10</h3>"
                f'{refinement_note}'
                f'<div class="score-grid">{score_items}</div>'
                f"</div></section>"
            )

        # Master synthesis
        if result.master_synthesis:
            sections.append(
                f'<section id="master">\n<h2>Executive Summary</h2>\n'
                f"{_md_to_html(result.master_synthesis)}\n</section>"
            )

        # Strategic analysis
        if result.strategic_analysis:
            sections.append(
                f'<section id="strategy">\n<h2>Strategic Analysis</h2>\n'
                f"{_md_to_html(result.strategic_analysis)}\n</section>"
            )

        # Individual studies
        if result.studies:
            study_parts: list[str] = []
            for i, study in enumerate(result.studies, 1):
                if not study.synthesis:
                    continue
                study_parts.append(
                    f'<div class="study">\n<h3>Study {i}: {html.escape(study.title)}</h3>\n'
                    f"{_md_to_html(study.synthesis)}\n</div>"
                )
            if study_parts:
                sections.append(
                    f'<section id="studies">\n<h2>Individual Studies</h2>\n'
                    + "\n".join(study_parts)
                    + "\n</section>"
                )

        # Q&A clusters
        qa_parts: list[str] = []
        for cluster in result.qa_clusters:
            if not cluster.findings:
                continue
            qa_parts.append(
                f'<div class="cluster">\n<h3>Cluster: {html.escape(cluster.theme)}</h3>\n'
                f"{_md_to_html(cluster.findings)}\n</div>"
            )
        if result.qa_summary:
            qa_parts.append(f"<h3>Q&amp;A Summary</h3>\n{_md_to_html(result.qa_summary)}")
        if qa_parts:
            sections.append(
                f'<section id="qa">\n<h2>Anticipated Q&amp;A</h2>\n'
                + "\n".join(qa_parts)
                + "\n</section>"
            )
    else:
        # QUICK / STANDARD — just the final synthesis
        if result.final_synthesis:
            sections.append(
                f'<section id="synthesis">\n<h2>Research Synthesis</h2>\n'
                f"{_md_to_html(result.final_synthesis)}\n</section>"
            )

    body = "\n".join(sections)
    title = html.escape(query[:120])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Research: {title}</title>
<style>{_CSS}</style>
</head>
<body>
<header>
<h1>Research Briefing</h1>
<p>{" | ".join(meta_parts)}</p>
</header>
{body}
</body>
</html>"""


def upload_html(html_content: str, conversation_id: str, bucket_name: str) -> str:
    """Upload HTML to GCS and return its public URL.

    Returns empty string if bucket is not configured or upload fails.
    """
    if not bucket_name:
        return ""

    try:
        from google.cloud import storage

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob_name = f"results/{conversation_id}.html"
        blob = bucket.blob(blob_name)
        blob.upload_from_string(html_content, content_type="text/html")

        return f"https://storage.googleapis.com/{bucket_name}/{blob_name}"
    except Exception:
        logger.exception("Failed to upload HTML results to GCS bucket %s", bucket_name)
        return ""


def publish_results(
    result: ResearchResult,
    query: str,
    depth: str,
    conversation_id: str,
    bucket_name: str,
) -> str:
    """Generate HTML and upload to GCS. Returns public URL or empty string."""
    html_content = generate_html(result, query, depth)
    return upload_html(html_content, conversation_id, bucket_name)


# ---------------------------------------------------------------------------
# Metadata helpers for the Web UI
# ---------------------------------------------------------------------------


def upload_metadata(metadata: dict, job_id: str, bucket_name: str) -> None:
    """Write a JSON metadata file to GCS at results/{job_id}_meta.json."""
    if not bucket_name:
        return
    try:
        from google.cloud import storage

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(f"results/{job_id}_meta.json")
        blob.upload_from_string(json.dumps(metadata), content_type="application/json")
    except Exception:
        logger.exception("Failed to upload metadata for job %s", job_id)


def list_results_metadata(bucket_name: str, limit: int = 50) -> list[dict]:
    """List metadata JSON blobs in GCS, return parsed list sorted newest-first."""
    if not bucket_name:
        return []
    try:
        from google.cloud import storage

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blobs = list(bucket.list_blobs(prefix="results/", max_results=500))

        meta_blobs = [b for b in blobs if b.name.endswith("_meta.json")]
        meta_blobs.sort(key=lambda b: b.time_created or b.updated, reverse=True)

        results = []
        for blob in meta_blobs[:limit]:
            try:
                data = json.loads(blob.download_as_text())
                results.append(data)
            except Exception:
                logger.warning("Failed to parse metadata blob %s", blob.name)
        return results
    except Exception:
        logger.exception("Failed to list results metadata from GCS")
        return []


def get_result_metadata(job_id: str, bucket_name: str) -> dict | None:
    """Fetch a single metadata JSON from GCS."""
    if not bucket_name:
        return None
    try:
        from google.cloud import storage

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(f"results/{job_id}_meta.json")
        if not blob.exists():
            return None
        return json.loads(blob.download_as_text())
    except Exception:
        logger.exception("Failed to fetch metadata for job %s", job_id)
        return None


def publish_results_with_metadata(
    result: ResearchResult,
    query: str,
    depth: str,
    job_id: str,
    bucket_name: str,
    elevenlabs_doc_id: str = "",
    phase_timings: dict | None = None,
    research_stats: dict | None = None,
) -> str:
    """Generate HTML, upload it, then write a metadata JSON alongside it.

    Returns the public URL of the HTML page, or empty string on failure.
    """
    html_content = generate_html(result, query, depth)
    result_url = upload_html(html_content, job_id, bucket_name)

    num_studies = len(result.studies) if result.studies else 0
    now = datetime.now(timezone.utc).isoformat()

    metadata = {
        "job_id": job_id,
        "query": query,
        "depth": depth.upper(),
        "status": "completed",
        "created_at": now,
        "completed_at": now,
        "result_url": result_url,
        "num_studies": num_studies,
        "elevenlabs_doc_id": elevenlabs_doc_id,
        "synthesis_score": result.synthesis_score,
        "refinement_rounds": result.refinement_rounds,
        "phase_timings": phase_timings or {},
        "research_stats": research_stats or {},
    }
    upload_metadata(metadata, job_id, bucket_name)
    return result_url


def _slugify(text: str, max_len: int = 60) -> str:
    """Convert text to a URL-friendly slug."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower().strip())
    slug = slug.strip("-")
    return slug[:max_len].rstrip("-")


def _notebooklm_html(title: str, md_content: str) -> str:
    """Wrap markdown content in a minimal HTML page for NotebookLM ingestion."""
    body = _md_to_html(md_content)
    safe_title = html.escape(title[:120])
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{safe_title}</title>
<style>{_CSS}</style>
</head>
<body>
<header><h1>{safe_title}</h1></header>
{body}
</body>
</html>"""


def publish_notebooklm_sources(
    result: ResearchResult,
    query: str,
    job_id: str,
    bucket_name: str,
) -> list[dict]:
    """Upload individual research documents as HTML files for NotebookLM.

    Each component (master synthesis, studies, Q&A, strategic analysis) becomes
    a separate .html file in GCS under notebooklm/{job_id}/.

    Returns list of dicts: [{"label": "...", "url": "https://..."}]
    """
    if not bucket_name:
        return []

    try:
        from google.cloud import storage

        client = storage.Client()
        bucket = client.bucket(bucket_name)
    except Exception:
        logger.exception("Failed to init GCS client for NotebookLM sources")
        return []

    prefix = f"notebooklm/{job_id}"
    sources: list[dict] = []

    def _upload_source(filename: str, label: str, title: str, content: str):
        if not content or not content.strip():
            return
        try:
            blob = bucket.blob(f"{prefix}/{filename}")
            html_content = _notebooklm_html(title, content)
            blob.upload_from_string(html_content, content_type="text/html")
            url = f"https://storage.googleapis.com/{bucket_name}/{prefix}/{filename}"
            sources.append({"label": label, "url": url})
        except Exception:
            logger.warning("Failed to upload NotebookLM source: %s", filename)

    # Master synthesis
    if result.master_synthesis:
        _upload_source(
            "00-executive-briefing.html", "Executive Briefing",
            f"Executive Briefing: {query}", result.master_synthesis,
        )

    # Strategic analysis
    if result.strategic_analysis:
        _upload_source(
            "01-strategic-analysis.html", "Strategic Analysis",
            f"Strategic Analysis: {query}", result.strategic_analysis,
        )

    # Individual studies
    if result.studies:
        for i, study in enumerate(result.studies, 1):
            if not study.synthesis:
                continue
            slug = _slugify(study.title)
            _upload_source(
                f"study-{i:02d}-{slug}.html", f"Study {i}: {study.title[:60]}",
                f"Study {i}: {study.title}", study.synthesis,
            )

    # Q&A clusters
    for i, cluster in enumerate(result.qa_clusters, 1):
        if not cluster.findings:
            continue
        slug = _slugify(cluster.theme)
        _upload_source(
            f"qa-{i:02d}-{slug}.html", f"Q&A: {cluster.theme[:60]}",
            f"Q&A: {cluster.theme}", cluster.findings,
        )

    # Q&A summary
    if result.qa_summary:
        _upload_source(
            "qa-summary.html", "Q&A Summary",
            f"Anticipated Q&A Summary: {query}", result.qa_summary,
        )

    # QUICK/STANDARD — single synthesis
    if result.final_synthesis and not result.master_synthesis:
        _upload_source(
            "synthesis.html", "Research Synthesis",
            f"Research Synthesis: {query}", result.final_synthesis,
        )

    logger.info("Published %d NotebookLM sources for job %s", len(sources), job_id)
    return sources


def delete_result(job_id: str, bucket_name: str) -> bool:
    """Delete HTML + metadata JSON from GCS. Returns True if anything was deleted."""
    if not bucket_name:
        return False
    try:
        from google.cloud import storage

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        deleted = False
        for suffix in [".html", "_meta.json", "_checkpoint.json"]:
            blob = bucket.blob(f"results/{job_id}{suffix}")
            if blob.exists():
                blob.delete()
                deleted = True
                logger.info("Deleted GCS blob: results/%s%s", job_id, suffix)
        # Also delete NotebookLM source files
        nb_blobs = list(bucket.list_blobs(prefix=f"notebooklm/{job_id}/"))
        for blob in nb_blobs:
            blob.delete()
            deleted = True
            logger.info("Deleted GCS blob: %s", blob.name)
        return deleted
    except Exception:
        logger.exception("Failed to delete result blobs for job %s", job_id)
        return False


# ---------------------------------------------------------------------------
# Checkpoint helpers for resumable DEEP pipeline
# ---------------------------------------------------------------------------


def save_checkpoint(result_dict: dict, job_id: str, bucket_name: str) -> None:
    """Save a pipeline checkpoint to GCS at results/{job_id}_checkpoint.json."""
    if not bucket_name:
        return
    try:
        from google.cloud import storage

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(f"results/{job_id}_checkpoint.json")
        blob.upload_from_string(json.dumps(result_dict), content_type="application/json")
        logger.info("Saved checkpoint for job %s", job_id)
    except Exception:
        logger.exception("Failed to save checkpoint for job %s", job_id)


def load_checkpoint(job_id: str, bucket_name: str) -> dict | None:
    """Load a pipeline checkpoint from GCS. Returns None if not found."""
    if not bucket_name:
        return None
    try:
        from google.cloud import storage

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(f"results/{job_id}_checkpoint.json")
        if not blob.exists():
            return None
        data = json.loads(blob.download_as_text())
        logger.info("Loaded checkpoint for job %s (phase: %s)", job_id, data.get("_checkpoint_phase", "?"))
        return data
    except Exception:
        logger.exception("Failed to load checkpoint for job %s", job_id)
        return None


def list_checkpoint_job_ids(bucket_name: str) -> set[str]:
    """Return set of job_ids that have checkpoint blobs in GCS."""
    if not bucket_name:
        return set()
    try:
        from google.cloud import storage

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix="results/", delimiter="/")
        ids = set()
        for blob in blobs:
            name = blob.name
            if name.endswith("_checkpoint.json"):
                # results/{job_id}_checkpoint.json → extract job_id
                jid = name.removeprefix("results/").removesuffix("_checkpoint.json")
                if jid:
                    ids.add(jid)
        return ids
    except Exception:
        logger.exception("Failed to list checkpoint job IDs")
        return set()


def delete_checkpoint(job_id: str, bucket_name: str) -> None:
    """Delete the checkpoint blob on successful completion."""
    if not bucket_name:
        return
    try:
        from google.cloud import storage

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(f"results/{job_id}_checkpoint.json")
        if blob.exists():
            blob.delete()
            logger.info("Deleted checkpoint for job %s", job_id)
    except Exception:
        logger.exception("Failed to delete checkpoint for job %s", job_id)


def update_metadata(job_id: str, bucket_name: str, updates: dict) -> None:
    """Merge updates into an existing metadata JSON in GCS."""
    if not bucket_name:
        return
    try:
        from google.cloud import storage

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(f"results/{job_id}_meta.json")
        if not blob.exists():
            logger.warning("Metadata blob not found for job %s, writing fresh", job_id)
            blob.upload_from_string(json.dumps(updates), content_type="application/json")
            return

        existing = json.loads(blob.download_as_text())
        existing.update(updates)
        blob.upload_from_string(json.dumps(existing), content_type="application/json")
    except Exception:
        logger.exception("Failed to update metadata for job %s", job_id)
