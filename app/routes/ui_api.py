"""Flask blueprint for the web UI and its API endpoints."""

import logging

from flask import Blueprint, current_app, jsonify, render_template, request

from app.models.depth import ResearchDepth
from app.services import gcs_client
from app.services.job_tracker import JobStatus, create_job, get_job
from app.services.research_orchestrator import run_research_for_ui

logger = logging.getLogger(__name__)

ui_api_bp = Blueprint("ui_api", __name__)

_VALID_DEPTHS = {"QUICK", "STANDARD", "DEEP"}


@ui_api_bp.route("/")
def index():
    """Serve the single-page app."""
    return render_template("index.html")


@ui_api_bp.route("/api/research", methods=["POST"])
def start_research():
    """Start a research job. Body: {query, depth} → Returns {job_id} (202)."""
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

    logger.info("Research job created: job=%s depth=%s query=%s", job_id, depth_str, query[:100])
    return jsonify({"job_id": job_id}), 202


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
        "created_at": job.created_at,
        "completed_at": job.completed_at,
    })


@ui_api_bp.route("/api/archive")
def archive():
    """List past research results from GCS metadata."""
    settings = current_app.config["SETTINGS"]
    results = gcs_client.list_results_metadata(settings.gcs_results_bucket)
    return jsonify({"results": results})
