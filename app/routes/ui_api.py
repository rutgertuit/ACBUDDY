"""Flask blueprint for the web UI and its API endpoints."""

import logging
import os
import secrets
import threading
import time

import requests
from flask import Blueprint, current_app, jsonify, render_template, request

from app.agents.agent_profiles import AGENTS, get_agent_id, get_voice_id
from app.models.depth import ResearchDepth
from app.models.research_result import ResearchResult
from app.services import elevenlabs_client, gcs_client
from app.services import knowledge_graph as kg
from app.services import memory_store
from app.services import podcast_service
from app.services import watch_store
from app.agents import podcast_generator
from app.services.job_tracker import JobStatus, count_active_jobs, create_job, get_job, update_job
from app.services.research_orchestrator import run_research_for_ui, run_amendment_for_ui, resume_research_for_ui

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
            model="gemini-2.5-flash",
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

    # Parse optional business context
    business_context = None
    bc = data.get("business_context")
    if isinstance(bc, dict) and any(bc.values()):
        business_context = {
            "user_role": (bc.get("user_role") or "").strip(),
            "industry": (bc.get("industry") or "").strip(),
            "decision_type": (bc.get("decision_type") or "").strip(),
            "stakeholders": (bc.get("stakeholders") or "").strip(),
        }

    job_id = create_job(query, depth_str)
    run_research_for_ui(job_id, query, depth, settings, business_context=business_context)

    estimated = _ESTIMATED_DURATION.get(depth_str, 300)
    logger.info("Research job created: job=%s depth=%s query=%s", job_id, depth_str, query[:100])
    return jsonify({"job_id": job_id, "estimated_seconds": estimated}), 202


@ui_api_bp.route("/api/research/amend", methods=["POST"])
def amend_research():
    """Start an amendment pipeline. Body: {job_id, additional_questions, perspective}."""
    data = request.get_json(silent=True) or {}
    parent_job_id = (data.get("job_id") or "").strip()
    questions_raw = data.get("additional_questions", "")
    perspective = (data.get("perspective") or "").strip()

    if not parent_job_id:
        return jsonify({"error": "job_id is required"}), 400

    # Parse questions: accept string (newline-separated) or list
    if isinstance(questions_raw, str):
        additional_questions = [q.strip() for q in questions_raw.split("\n") if q.strip()]
    elif isinstance(questions_raw, list):
        additional_questions = [str(q).strip() for q in questions_raw if str(q).strip()]
    else:
        additional_questions = []

    if not additional_questions:
        return jsonify({"error": "additional_questions is required"}), 400

    settings = current_app.config["SETTINGS"]

    # Fetch original query from metadata
    meta = gcs_client.get_result_metadata(parent_job_id, settings.gcs_results_bucket)
    original_query = meta.get("query", "") if meta else ""
    if not original_query:
        return jsonify({"error": "Original research not found"}), 404

    # Create new job
    new_job_id = create_job(f"Amendment: {original_query[:80]}", "STANDARD")
    update_job(new_job_id, parent_job_id=parent_job_id)

    run_amendment_for_ui(
        job_id=new_job_id,
        parent_job_id=parent_job_id,
        original_query=original_query,
        additional_questions=additional_questions,
        perspective=perspective,
        settings=settings,
    )

    logger.info("Amendment job created: job=%s parent=%s", new_job_id, parent_job_id)
    return jsonify({
        "job_id": new_job_id,
        "parent_job_id": parent_job_id,
        "estimated_seconds": 300,
    }), 202


@ui_api_bp.route("/api/research/resume", methods=["POST"])
def resume_research():
    """Resume a failed DEEP research job from its last checkpoint.

    Body: {job_id}
    Returns: {job_id, resumed: true, checkpoint_phase}
    """
    data = request.get_json(silent=True) or {}
    job_id = (data.get("job_id") or "").strip()
    if not job_id:
        return jsonify({"error": "job_id is required"}), 400

    settings = current_app.config["SETTINGS"]
    bucket = settings.gcs_results_bucket

    # Load checkpoint to verify it exists
    checkpoint = gcs_client.load_checkpoint(job_id, bucket)
    if not checkpoint:
        return jsonify({"error": "No checkpoint found for this job"}), 404

    # Verify job is in failed state (or lost from memory after restart)
    job = get_job(job_id)
    if job and job.status not in (JobStatus.FAILED,):
        return jsonify({"error": f"Job is {job.status.value}, not failed"}), 400

    checkpoint_phase = checkpoint.get("_checkpoint_phase", "unknown")

    # Get query from metadata for the response
    meta = gcs_client.get_result_metadata(job_id, bucket)
    query = meta.get("query", "") if meta else ""

    resume_research_for_ui(job_id, settings)

    estimated = _ESTIMATED_DURATION.get("DEEP", 2400)
    logger.info("Research resume started: job=%s from_phase=%s", job_id, checkpoint_phase)
    return jsonify({
        "job_id": job_id,
        "resumed": True,
        "checkpoint_phase": checkpoint_phase,
        "estimated_seconds": estimated,
        "query": query,
    }), 202


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
        "phase_timings": job.phase_timings,
        "research_stats": job.research_stats,
        "parent_job_id": job.parent_job_id,
        "notebooklm_urls": job.notebooklm_urls,
    })


@ui_api_bp.route("/api/archive")
def archive():
    """List past research results from GCS metadata."""
    settings = current_app.config["SETTINGS"]
    results = gcs_client.list_results_metadata(settings.gcs_results_bucket)
    return jsonify({"results": results})


@ui_api_bp.route("/api/archive/<job_id>", methods=["DELETE"])
def delete_archive(job_id: str):
    """Delete a research result: detach from agents, delete from GCS."""
    settings = current_app.config["SETTINGS"]
    bucket = settings.gcs_results_bucket

    # Fetch metadata to find elevenlabs_doc_id
    meta = gcs_client.get_result_metadata(job_id, bucket)
    if meta is None:
        return jsonify({"error": "Research not found"}), 404

    doc_id = meta.get("elevenlabs_doc_id", "")

    # Detach from all 3 agents
    if doc_id and settings.elevenlabs_api_key:
        for slug in AGENTS:
            agent_id = get_agent_id(slug, settings)
            if not agent_id:
                continue
            try:
                elevenlabs_client.detach_document_from_agent(
                    agent_id=agent_id,
                    doc_id=doc_id,
                    api_key=settings.elevenlabs_api_key,
                )
            except Exception:
                logger.warning("Failed to detach doc %s from agent %s", doc_id, slug)

    # Delete GCS blobs
    gcs_client.delete_result(job_id, bucket)

    # Invalidate caches
    for s in AGENTS:
        _cache.pop(f"kb_docs_{s}", None)
    _cache.pop("completed_count", None)

    return jsonify({"ok": True})


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


@ui_api_bp.route("/api/timing-estimates")
def timing_estimates():
    """Return average phase durations from past DEEP runs (cached 5 min)."""
    cached = _cached("timing_estimates")
    if cached is not None:
        return jsonify(cached)

    settings = current_app.config["SETTINGS"]
    results = gcs_client.list_results_metadata(settings.gcs_results_bucket, limit=20)

    # Collect timings from DEEP runs that have phase_timings
    phase_totals: dict[str, list[float]] = {}
    total_durations: list[float] = []
    for meta in results:
        if meta.get("depth") != "DEEP":
            continue
        timings = meta.get("phase_timings", {})
        if not timings:
            continue
        run_total = 0.0
        for phase, data in timings.items():
            dur = data.get("duration", 0)
            if dur > 0:
                phase_totals.setdefault(phase, []).append(dur)
                run_total += dur
        if run_total > 0:
            total_durations.append(run_total)

    # Compute averages
    averages = {}
    for phase, durations in phase_totals.items():
        averages[phase] = round(sum(durations) / len(durations))
    total_avg = round(sum(total_durations) / len(total_durations)) if total_durations else 0

    data = {
        "phase_averages": averages,
        "total_average": total_avg,
        "sample_count": len(total_durations),
    }
    _set_cache("timing_estimates", data)
    return jsonify(data)


@ui_api_bp.route("/api/agents")
def list_agents():
    """List the 3 agents with their KB doc names (cached).

    Pass ?fresh=1 to bypass the cache (e.g. after research completes).
    """
    settings = current_app.config["SETTINGS"]
    skip_cache = request.args.get("fresh") == "1"

    agents_out = []
    for slug, profile in AGENTS.items():
        agent_id = get_agent_id(slug, settings)
        kb_docs = []

        if agent_id:
            cache_key = f"kb_docs_{slug}"
            if skip_cache:
                _cache.pop(cache_key, None)
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
        # Trigger both RAG index models so the agent can retrieve the doc
        try:
            elevenlabs_client.trigger_all_rag_indexes(
                doc_id=doc_id,
                api_key=settings.elevenlabs_api_key,
            )
        except Exception:
            logger.warning("RAG index trigger failed for doc %s (non-fatal)", doc_id)
        # Invalidate ALL agent caches so UI reflects the change
        _cache.pop(f"kb_docs_{slug}", None)
        for s in AGENTS:
            _cache.pop(f"kb_docs_{s}", None)
        return jsonify({"ok": True})
    except elevenlabs_client.RagIndexNotReadyError as e:
        logger.warning("RAG not ready for doc %s on agent %s", doc_id, slug)
        return jsonify({"error": str(e), "rag_not_ready": True}), 409
    except requests.exceptions.HTTPError as e:
        body = ""
        if e.response is not None:
            try:
                body = e.response.text[:500]
            except Exception:
                pass
        logger.exception("Failed to attach doc %s to agent %s: %s", doc_id, slug, body)
        return jsonify({"error": f"{e} — {body}"}), 500
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
        # Invalidate ALL agent caches so UI reflects the change
        _cache.pop(f"kb_docs_{slug}", None)
        for s in AGENTS:
            _cache.pop(f"kb_docs_{s}", None)
        return jsonify({"ok": True})
    except Exception as e:
        logger.exception("Failed to detach doc %s from agent %s", doc_id, slug)
        return jsonify({"error": str(e)}), 500


# ── Podcast endpoints ──

# In-memory podcast job tracking: {podcast_job_id: {job_id, status, phase, audio_url, script_preview, style, error}}
_podcast_jobs: dict[str, dict] = {}
_podcast_lock = threading.Lock()


def _load_research_result(job_id: str, settings) -> tuple[ResearchResult | None, dict | None]:
    """Load a ResearchResult from GCS checkpoint or reconstruct from metadata."""
    bucket = settings.gcs_results_bucket

    # Try checkpoint first (DEEP pipeline)
    checkpoint = gcs_client.load_checkpoint(job_id, bucket)
    if checkpoint:
        checkpoint.pop("_checkpoint_phase", None)
        return ResearchResult.from_dict(checkpoint), None

    # Fetch metadata
    meta = gcs_client.get_result_metadata(job_id, bucket)
    if not meta:
        return None, None

    # Try to download the HTML and extract text content for a basic result
    result = ResearchResult(original_query=meta.get("query", ""))

    # For completed research, we need the research content.
    # The metadata itself doesn't contain full text, but the HTML blob does.
    # Try to fetch the HTML blob and extract text.
    try:
        from google.cloud import storage
        import re
        client = storage.Client()
        bucket_obj = client.bucket(bucket)
        blob = bucket_obj.blob(f"results/{job_id}.html")
        if blob.exists():
            html_content = blob.download_as_text()
            # Strip HTML tags to get plain text
            text = re.sub(r"<[^>]+>", " ", html_content)
            text = re.sub(r"\s+", " ", text).strip()
            # Use as final_synthesis (good enough for podcast script generation)
            result.final_synthesis = text[:50000]
    except Exception:
        logger.warning("Could not load HTML for job %s", job_id)

    return result, meta


@ui_api_bp.route("/api/podcast/analyze", methods=["POST"])
def analyze_podcast():
    """Analyze research for podcast generation. Body: {job_id} -> {storylines, styles}."""
    data = request.get_json(silent=True) or {}
    job_id = (data.get("job_id") or "").strip()

    if not job_id:
        return jsonify({"error": "job_id is required"}), 400

    settings = current_app.config["SETTINGS"]
    result, meta = _load_research_result(job_id, settings)

    if result is None:
        return jsonify({"error": "Research not found"}), 404

    if not result.final_synthesis and not result.master_synthesis:
        return jsonify({"error": "Research has no content for podcast generation"}), 400

    query = result.original_query or (meta or {}).get("query", "Research")

    try:
        analysis = podcast_generator.analyze_for_podcast(result, query)

        # Add available hosts from agent profiles
        hosts = [
            {
                "slug": p.slug,
                "name": p.name,
                "subtitle": p.subtitle,
                "personality": p.personality,
                "icon": p.icon,
                "color": p.color,
            }
            for p in AGENTS.values()
        ]
        analysis["hosts"] = hosts

        return jsonify(analysis)
    except Exception as e:
        logger.exception("Podcast analysis failed for job %s", job_id)
        return jsonify({"error": str(e)}), 500


@ui_api_bp.route("/api/podcast/generate", methods=["POST"])
def generate_podcast():
    """Start podcast generation. Body: {job_id, style, host_slug?, guest_slug?, angles?, scenario?} -> 202 {podcast_job_id}."""
    data = request.get_json(silent=True) or {}
    job_id = (data.get("job_id") or "").strip()
    style = (data.get("style") or "").strip()
    host_slug = (data.get("host_slug") or "").strip()
    guest_slug = (data.get("guest_slug") or "").strip()
    selected_angles = data.get("angles") or []
    selected_scenario = data.get("scenario") or None

    if not job_id:
        return jsonify({"error": "job_id is required"}), 400
    if style not in ("executive", "curious", "debate"):
        return jsonify({"error": "style must be executive, curious, or debate"}), 400

    settings = current_app.config["SETTINGS"]
    result, meta = _load_research_result(job_id, settings)
    if result is None:
        return jsonify({"error": "Research not found"}), 404

    query = result.original_query or (meta or {}).get("query", "Research")

    # Resolve host/guest profiles and voice IDs
    host_profile = None
    guest_profile = None
    host_voice = ""
    guest_voice = ""

    if host_slug and host_slug in AGENTS:
        p = AGENTS[host_slug]
        host_profile = {"name": p.name, "personality": p.personality}
        host_voice = get_voice_id(host_slug, settings)
    if guest_slug and guest_slug in AGENTS:
        p = AGENTS[guest_slug]
        guest_profile = {"name": p.name, "personality": p.personality}
        guest_voice = get_voice_id(guest_slug, settings)

    podcast_job_id = secrets.token_hex(6)
    with _podcast_lock:
        _podcast_jobs[podcast_job_id] = {
            "job_id": job_id,
            "status": "scripting",
            "phase": "Generating script...",
            "audio_url": "",
            "script_preview": "",
            "style": style,
            "error": "",
        }

    # Update the research job tracker if it exists
    research_job = get_job(job_id)
    if research_job:
        update_job(job_id, podcast_job_id=podcast_job_id, podcast_status="scripting", podcast_style=style)

    # Build speaker → voice_id mapping for the TTS service
    speaker_voices: dict[str, str] = {}
    if host_profile and host_profile.get("name") and host_voice:
        speaker_voices[host_profile["name"]] = host_voice
    if guest_profile and guest_profile.get("name") and guest_voice:
        speaker_voices[guest_profile["name"]] = guest_voice

    # Launch background thread for the full pipeline
    def _run_podcast_pipeline():
        api_key = settings.elevenlabs_api_key
        bucket = settings.gcs_results_bucket

        try:
            # Phase 1: Generate script (with v3 audio tags)
            script = podcast_generator.generate_podcast_script(
                result, query, style,
                host_profile=host_profile,
                guest_profile=guest_profile,
                angles=selected_angles if selected_angles else None,
                scenario=selected_scenario,
            )
            preview = script[:200].replace("\n", " ")

            with _podcast_lock:
                pj = _podcast_jobs.get(podcast_job_id)
                if pj:
                    pj["status"] = "generating"
                    pj["phase"] = "Creating podcast audio..."
                    pj["script_preview"] = preview
            if research_job:
                update_job(job_id, podcast_status="generating", podcast_script_preview=preview)

            # Save script to GCS for review
            script_url = podcast_service.upload_podcast_script(script, job_id, bucket)
            if script_url:
                logger.info("Podcast script saved: %s", script_url)

            # Phase 2: Generate audio per speaker turn using ElevenLabs v3 TTS
            def _on_progress(current, total):
                with _podcast_lock:
                    pj = _podcast_jobs.get(podcast_job_id)
                    if pj:
                        pj["phase"] = f"Generating audio... ({current}/{total} turns)"

            audio_bytes = podcast_service.create_podcast(
                script=script,
                speaker_voices=speaker_voices,
                api_key=api_key,
                on_progress=_on_progress,
            )

            # Phase 3: Upload to GCS
            with _podcast_lock:
                pj = _podcast_jobs.get(podcast_job_id)
                if pj:
                    pj["phase"] = "Uploading audio..."

            audio_url = podcast_service.upload_podcast_audio(audio_bytes, job_id, bucket)
            if not audio_url:
                raise RuntimeError("Failed to upload podcast audio to storage")

            with _podcast_lock:
                pj = _podcast_jobs.get(podcast_job_id)
                if pj:
                    pj["status"] = "completed"
                    pj["phase"] = "Done"
                    pj["audio_url"] = audio_url

            if research_job:
                update_job(job_id, podcast_status="completed", podcast_audio_url=audio_url)

            # Update GCS metadata
            gcs_client.update_metadata(job_id, bucket, {
                "podcast_url": audio_url,
                "podcast_style": style,
                "podcast_script_url": script_url,
            })

            logger.info("Podcast complete: podcast_job=%s audio_url=%s", podcast_job_id, audio_url)

        except Exception as e:
            logger.exception("Podcast pipeline failed: podcast_job=%s", podcast_job_id)
            with _podcast_lock:
                pj = _podcast_jobs.get(podcast_job_id)
                if pj:
                    pj["status"] = "failed"
                    pj["phase"] = "Failed"
                    pj["error"] = str(e)
            if research_job:
                update_job(job_id, podcast_status="failed")

    thread = threading.Thread(target=_run_podcast_pipeline, daemon=True)
    thread.start()

    logger.info("Podcast generation started: podcast_job=%s job=%s style=%s", podcast_job_id, job_id, style)
    return jsonify({"podcast_job_id": podcast_job_id}), 202


@ui_api_bp.route("/api/podcast/status/<podcast_job_id>")
def podcast_status(podcast_job_id: str):
    """Poll podcast generation status."""
    with _podcast_lock:
        pj = _podcast_jobs.get(podcast_job_id)
    if pj is None:
        return jsonify({"error": "Podcast job not found"}), 404
    return jsonify(pj)


# ── Knowledge Graph endpoints ──


@ui_api_bp.route("/api/graph")
def get_graph():
    """Return the full knowledge graph with stats."""
    settings = current_app.config["SETTINGS"]
    graph = kg.load_graph(settings.gcs_results_bucket)
    stats = kg.get_graph_stats(graph)
    return jsonify({
        "stats": stats,
        "entities": [
            {"name": e.name, "type": e.type, "aliases": e.aliases, "mentions": len(e.source_jobs)}
            for e in graph.entities.values()
        ],
        "relationships": [
            {"from": r.from_entity, "to": r.to_entity, "type": r.type,
             "description": r.description, "mentions": len(r.source_jobs)}
            for r in graph.relationships
        ],
    })


@ui_api_bp.route("/api/graph/entity/<name>")
def get_graph_entity(name: str):
    """Find connections for a specific entity."""
    settings = current_app.config["SETTINGS"]
    graph = kg.load_graph(settings.gcs_results_bucket)
    result = kg.find_connections(graph, name)
    return jsonify(result)


# ── Memory endpoints ──


@ui_api_bp.route("/api/memory")
def get_memory():
    """List all memories with stats."""
    settings = current_app.config["SETTINGS"]
    store = memory_store.load_memory(settings.gcs_results_bucket)
    stats = memory_store.get_memory_stats(store)
    entries = [
        {"id": e.id, "type": e.type, "content": e.content,
         "source_query": e.source_query, "tags": e.tags, "created_at": e.created_at}
        for e in store.entries
    ]
    return jsonify({"stats": stats, "entries": entries})


@ui_api_bp.route("/api/memory/recall")
def recall_memory():
    """Recall relevant memories for a query. Query param: ?q=..."""
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"error": "q parameter required"}), 400
    settings = current_app.config["SETTINGS"]
    store = memory_store.load_memory(settings.gcs_results_bucket)
    results = memory_store.recall(store, query)
    return jsonify({"results": results, "count": len(results)})


@ui_api_bp.route("/api/memory/<memory_id>", methods=["DELETE"])
def delete_memory_entry(memory_id: str):
    """Delete a memory entry."""
    settings = current_app.config["SETTINGS"]
    store = memory_store.load_memory(settings.gcs_results_bucket)
    if memory_store.delete_memory(store, memory_id):
        memory_store.save_memory(store, settings.gcs_results_bucket)
        return jsonify({"ok": True})
    return jsonify({"error": "Memory not found"}), 404


# ── Watch endpoints ──


@ui_api_bp.route("/api/watches", methods=["POST"])
def create_watch_endpoint():
    """Create a research watch. Body: {query, interval_hours}."""
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    interval_hours = int(data.get("interval_hours", 24))

    if not query:
        return jsonify({"error": "query is required"}), 400

    settings = current_app.config["SETTINGS"]
    watch = watch_store.create_watch(query, interval_hours, settings.gcs_results_bucket)

    # Optional notification settings
    notification_email = (data.get("notification_email") or "").strip()
    notification_webhook = (data.get("notification_webhook") or "").strip()
    if notification_email:
        watch.notification_email = notification_email
    if notification_webhook:
        watch.notification_webhook = notification_webhook
    if notification_email or notification_webhook:
        watch_store._save_watch(watch, settings.gcs_results_bucket)

    return jsonify({
        "id": watch.id,
        "query": watch.query,
        "interval_hours": watch.interval_hours,
        "created_at": watch.created_at,
    }), 201


@ui_api_bp.route("/api/watches")
def list_watches_endpoint():
    """List all research watches."""
    settings = current_app.config["SETTINGS"]
    watches = watch_store.list_watches(settings.gcs_results_bucket)
    return jsonify({
        "watches": [
            {
                "id": w.id, "query": w.query, "interval_hours": w.interval_hours,
                "created_at": w.created_at, "last_checked": w.last_checked,
                "active": w.active, "history_count": len(w.history),
                "last_changed": next(
                    (h["checked_at"] for h in reversed(w.history) if h.get("changed")), ""
                ),
            }
            for w in watches
        ]
    })


@ui_api_bp.route("/api/watches/<watch_id>/check", methods=["POST"])
def check_watch_endpoint(watch_id: str):
    """Manually trigger a watch check."""
    import asyncio

    settings = current_app.config["SETTINGS"]
    watch = watch_store.get_watch(watch_id, settings.gcs_results_bucket)
    if not watch:
        return jsonify({"error": "Watch not found"}), 404

    try:
        from app.agents.watch_checker import check_watch
        findings = asyncio.run(check_watch(watch.query))
        update = watch_store.record_check(watch, findings, settings.gcs_results_bucket)

        # Send notification if changes detected
        if update.changed and (watch.notification_email or watch.notification_webhook):
            try:
                from app.services.notification_client import send_watch_notification
                asyncio.run(send_watch_notification(watch, update))
            except Exception:
                logger.warning("Notification failed for watch %s (non-fatal)", watch_id)

        return jsonify({
            "checked_at": update.checked_at,
            "changed": update.changed,
            "summary": update.summary,
        })
    except Exception as e:
        logger.exception("Watch check failed for %s", watch_id)
        return jsonify({"error": str(e)}), 500


@ui_api_bp.route("/api/watches/<watch_id>", methods=["DELETE"])
def delete_watch_endpoint(watch_id: str):
    """Delete a research watch."""
    settings = current_app.config["SETTINGS"]
    if watch_store.delete_watch(watch_id, settings.gcs_results_bucket):
        return jsonify({"ok": True})
    return jsonify({"error": "Watch not found"}), 404


@ui_api_bp.route("/api/watches/check-all", methods=["POST"])
def check_all_watches_endpoint():
    """Check all due watches. For Cloud Scheduler automation."""
    import asyncio

    settings = current_app.config["SETTINGS"]
    due = watch_store.get_due_watches(settings.gcs_results_bucket)
    if not due:
        return jsonify({"checked": 0, "message": "No watches due"})

    results = []
    for watch in due:
        try:
            from app.agents.watch_checker import check_watch
            findings = asyncio.run(check_watch(watch.query))
            update = watch_store.record_check(watch, findings, settings.gcs_results_bucket)

            # Send notification if changes detected
            if update.changed and (watch.notification_email or watch.notification_webhook):
                try:
                    from app.services.notification_client import send_watch_notification
                    asyncio.run(send_watch_notification(watch, update))
                except Exception:
                    logger.warning("Notification failed for watch %s (non-fatal)", watch.id)

            results.append({
                "watch_id": watch.id,
                "query": watch.query,
                "changed": update.changed,
            })
        except Exception as e:
            logger.exception("Watch check failed for %s", watch.id)
            results.append({"watch_id": watch.id, "error": str(e)})

    return jsonify({"checked": len(results), "results": results})
