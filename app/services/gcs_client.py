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
        # Master synthesis
        if result.master_synthesis:
            sections.append(
                f'<section id="master">\n<h2>Executive Summary</h2>\n'
                f"{_md_to_html(result.master_synthesis)}\n</section>"
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
    }
    upload_metadata(metadata, job_id, bucket_name)
    return result_url


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
