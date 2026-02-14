"""Flask blueprint for the web UI and its API endpoints."""

import logging
import time

from flask import Blueprint, current_app, jsonify, render_template, request

from app.agents.agent_profiles import AGENTS, get_agent_id
from app.models.depth import ResearchDepth
from app.services import elevenlabs_client, gcs_client
from app.services.job_tracker import JobStatus, count_active_jobs, create_job, get_job
from app.services.research_orchestrator import run_research_for_ui

logger = logging.getLogger(__name__)

ui_api_bp = Blueprint("ui_api", __name__)

_VALID_DEPTHS = {"QUICK", "STANDARD", "DEEP"}

# Simple in-memory cache: {key: (timestamp, data)}
_cache: dict[str, tuple[float, object]] = {}
_CACHE_TTL = 60  # seconds


def _cached(key: str):
    entry = _cache.get(key)
    if entry and (time.time() - entry[0]) < _CACHE_TTL:
        return entry[1]
    return None


def _set_cache(key: str, data):
    _cache[key] = (time.time(), data)


@ui_api_bp.route("/")
def index():
    """Serve the single-page app."""
    return render_template("index.html")


# Estimated durations in seconds per depth (for countdown timer)
_ESTIMATED_DURATION = {"QUICK": 90, "STANDARD": 300, "DEEP": 2400}


@ui_api_bp.route("/api/research/validate", methods=["POST"])
def validate_research():
    """Validate research query clarity using Gemini. Body: {query, depth}."""
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    depth_str = (data.get("depth") or "STANDARD").upper()

    if not query:
        return jsonify({"error": "query is required"}), 400

    # Only run deep validation for DEEP depth
    if depth_str != "DEEP":
        return jsonify({"clear": True, "feedback": "", "suggested_query": ""})

    settings = current_app.config["SETTINGS"]
    try:
        from google import genai
        from google.genai.types import GenerateContentConfig

        client = genai.Client(api_key=settings.google_api_key)
        resp = client.models.generate_content(
            model="gemini-2.0-flash",
            config=GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=500,
                response_mime_type="application/json",
            ),
            contents=f"""You are a research clarity evaluator. A user wants to run a DEEP research pipeline (~40 minutes) on this query:

"{query}"

Evaluate whether this query is specific enough to produce good research results. Consider:
- Is there a clear topic or question?
- Is the scope reasonable (not too broad, not too narrow)?
- Would a researcher know what to look for?

Respond in JSON: {{"clear": true/false, "feedback": "brief explanation if not clear", "suggested_query": "improved version if not clear, empty string if clear"}}

Be reasonably strict — single-word or two-word topics like "AI" or "crypto" are too vague for a 40-minute DEEP research pipeline. But "AI in healthcare diagnostics" or "crypto regulation in the EU" are fine.""",
        )
        import json
        result = json.loads(resp.text)
        return jsonify({
            "clear": bool(result.get("clear", True)),
            "feedback": str(result.get("feedback", "")),
            "suggested_query": str(result.get("suggested_query", "")),
        })
    except Exception as e:
        logger.exception("Validation failed, allowing query through")
        return jsonify({"clear": True, "feedback": "", "suggested_query": ""})


@ui_api_bp.route("/api/research", methods=["POST"])
def start_research():
    """Start a research job. Body: {query, depth} -> Returns {job_id} (202)."""
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    depth_str = (data.get("depth") or "STANDARD").upper()

    if not query:
        return jsonify({"error": "query is required"}), 400
    if depth_str not in _VALID_DEPTHS:
        return jsonify({"error": f"depth must be one of {_VALID_DEPTHS}"}), 400

    depth = ResearchDepth(depth_str.lower())
    settings = current_app.config["SETTINGS"]

    job_id = create_job(query, depth_str)
    run_research_for_ui(job_id, query, depth, settings)

    estimated = _ESTIMATED_DURATION.get(depth_str, 300)
    logger.info("Research job created: job=%s depth=%s query=%s", job_id, depth_str, query[:100])
    return jsonify({"job_id": job_id, "estimated_seconds": estimated}), 202


@ui_api_bp.route("/api/status/<job_id>")
def job_status(job_id: str):
    """Poll status of a research job."""
    job = get_job(job_id)
    if job is None:
        return jsonify({"error": "Job not found"}), 404

    return jsonify({
        "job_id": job.job_id,
        "query": job.query,
        "depth": job.depth,
        "status": job.status.value,
        "phase": job.phase,
        "result_url": job.result_url,
        "error": job.error,
        "elevenlabs_doc_id": job.elevenlabs_doc_id,
        "created_at": job.created_at,
        "completed_at": job.completed_at,
        "study_plan": job.study_plan,
        "study_progress": job.study_progress,
        "current_step": job.current_step,
    })


@ui_api_bp.route("/api/archive")
def archive():
    """List past research results from GCS metadata."""
    settings = current_app.config["SETTINGS"]
    results = gcs_client.list_results_metadata(settings.gcs_results_bucket)
    return jsonify({"results": results})


@ui_api_bp.route("/api/stats")
def stats():
    """Live counters: researching (in-memory) + completed (GCS, cached)."""
    researching = count_active_jobs()

    cached_completed = _cached("completed_count")
    if cached_completed is not None:
        completed = cached_completed
    else:
        settings = current_app.config["SETTINGS"]
        results = gcs_client.list_results_metadata(settings.gcs_results_bucket, limit=500)
        completed = len(results)
        _set_cache("completed_count", completed)

    return jsonify({"researching": researching, "completed": completed})


@ui_api_bp.route("/api/agents")
def list_agents():
    """List the 3 agents with their KB doc names (cached)."""
    settings = current_app.config["SETTINGS"]

    agents_out = []
    for slug, profile in AGENTS.items():
        agent_id = get_agent_id(slug, settings)
        kb_docs = []

        if agent_id:
            cache_key = f"kb_docs_{slug}"
            cached = _cached(cache_key)
            if cached is not None:
                kb_docs = cached
            else:
                try:
                    kb = elevenlabs_client.list_agent_knowledge_base(
                        agent_id, settings.elevenlabs_api_key
                    )
                    kb_docs = [
                        {"id": d.get("id", d.get("document_id", "")), "name": d.get("name", "")}
                        for d in kb
                    ]
                except Exception:
                    logger.exception("Failed to fetch KB for agent %s", slug)
                _set_cache(cache_key, kb_docs)

        agents_out.append({
            "slug": profile.slug,
            "name": profile.name,
            "subtitle": profile.subtitle,
            "personality": profile.personality,
            "icon": profile.icon,
            "color": profile.color,
            "agent_id": agent_id,
            "kb_count": len(kb_docs),
            "kb_docs": kb_docs,
        })

    return jsonify({"agents": agents_out})


@ui_api_bp.route("/api/agents/<slug>/attach", methods=["POST"])
def attach_kb(slug: str):
    """Attach a KB doc to an agent. Body: {doc_id, doc_name}."""
    if slug not in AGENTS:
        return jsonify({"error": "Unknown agent"}), 404

    settings = current_app.config["SETTINGS"]
    agent_id = get_agent_id(slug, settings)
    if not agent_id:
        return jsonify({"error": "Agent ID not configured"}), 400

    data = request.get_json(silent=True) or {}
    doc_id = (data.get("doc_id") or "").strip()
    doc_name = (data.get("doc_name") or "").strip()
    if not doc_id:
        return jsonify({"error": "doc_id is required"}), 400

    try:
        elevenlabs_client.attach_document_to_agent(
            agent_id=agent_id,
            doc_id=doc_id,
            doc_name=doc_name or doc_id,
            api_key=settings.elevenlabs_api_key,
        )
        # Invalidate cache
        _cache.pop(f"kb_count_{slug}", None)
        return jsonify({"ok": True})
    except Exception as e:
        logger.exception("Failed to attach doc %s to agent %s", doc_id, slug)
        return jsonify({"error": str(e)}), 500


@ui_api_bp.route("/api/agents/<slug>/kb")
def list_agent_kb(slug: str):
    """List KB docs attached to an agent."""
    if slug not in AGENTS:
        return jsonify({"error": "Unknown agent"}), 404

    settings = current_app.config["SETTINGS"]
    agent_id = get_agent_id(slug, settings)
    if not agent_id:
        return jsonify({"error": "Agent ID not configured"}), 400

    try:
        kb = elevenlabs_client.list_agent_knowledge_base(
            agent_id, settings.elevenlabs_api_key
        )
        return jsonify({"documents": kb})
    except Exception as e:
        logger.exception("Failed to list KB for agent %s", slug)
        return jsonify({"error": str(e)}), 500


@ui_api_bp.route("/api/agents/<slug>/kb/<doc_id>", methods=["DELETE"])
def detach_kb(slug: str, doc_id: str):
    """Detach a KB doc from an agent."""
    if slug not in AGENTS:
        return jsonify({"error": "Unknown agent"}), 404

    settings = current_app.config["SETTINGS"]
    agent_id = get_agent_id(slug, settings)
    if not agent_id:
        return jsonify({"error": "Agent ID not configured"}), 400

    try:
        elevenlabs_client.detach_document_from_agent(
            agent_id=agent_id,
            doc_id=doc_id,
            api_key=settings.elevenlabs_api_key,
        )
        # Invalidate cache
        _cache.pop(f"kb_count_{slug}", None)
        return jsonify({"ok": True})
    except Exception as e:
        logger.exception("Failed to detach doc %s from agent %s", doc_id, slug)
        return jsonify({"error": str(e)}), 500
