import asyncio
import logging
import threading
import time
from datetime import datetime, timezone

from app.agents.agent_profiles import AGENTS, get_agent_id
from app.config import Settings
from app.models.depth import ResearchDepth
from app.services import elevenlabs_client, gcs_client
from app.services.job_tracker import JobStatus, update_job
from app.agents.root_agent import execute_research

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
INITIAL_BACKOFF = 2


def run_research_pipeline(
    conversation_id: str,
    agent_id: str,
    user_query: str,
    settings: Settings,
    depth: ResearchDepth = ResearchDepth.STANDARD,
) -> None:
    """Run the full research pipeline in a background thread.

    All exceptions are caught and logged (no re-raise in background thread).
    """
    try:
        logger.info(
            "Starting %s research pipeline: conversation=%s query=%s",
            depth.value,
            conversation_id,
            user_query[:100],
        )

        # 1. Fetch conversation context
        context = ""
        try:
            conv_data = elevenlabs_client.get_conversation(
                conversation_id, settings.elevenlabs_api_key
            )
            context = elevenlabs_client.format_conversation_context(conv_data)
            logger.info("Fetched conversation context (%d chars)", len(context))
        except Exception:
            logger.warning(
                "Failed to fetch conversation context, proceeding without it",
                exc_info=True,
            )

        # 2. Execute ADK research pipeline
        result = asyncio.run(execute_research(query=user_query, context=context, depth=depth))

        # 3. Upload and attach based on depth
        if depth == ResearchDepth.DEEP:
            _handle_deep_upload(result, user_query, conversation_id, agent_id, settings)
        else:
            _handle_standard_upload(result, user_query, conversation_id, agent_id, settings)

    except Exception:
        logger.exception(
            "Research pipeline failed for conversation %s", conversation_id
        )


def _handle_standard_upload(result, user_query, conversation_id, agent_id, settings):
    """Upload single document for QUICK/STANDARD pipelines."""
    if not result.final_synthesis:
        logger.error("Research pipeline produced empty synthesis")
        return

    logger.info(
        "Research complete: %d questions, %d findings, synthesis=%d chars",
        len(result.unpacked_questions),
        len(result.research_findings),
        len(result.final_synthesis),
    )

    doc_name = f"Research: {user_query[:80]} ({conversation_id[:8]})"
    doc_id = _upload_with_retry(
        text=result.final_synthesis,
        name=doc_name,
        api_key=settings.elevenlabs_api_key,
    )

    if not doc_id:
        logger.error("Failed to upload research to KB after retries")
        return

    elevenlabs_client.attach_document_to_agent(
        agent_id=agent_id,
        doc_id=doc_id,
        doc_name=doc_name,
        api_key=settings.elevenlabs_api_key,
    )
    logger.info("Pipeline complete: doc=%s attached to agent=%s", doc_id, agent_id)

    if settings.gcs_results_bucket:
        url = gcs_client.publish_results(
            result, user_query, "standard", conversation_id, settings.gcs_results_bucket
        )
        if url:
            logger.info("Results page: %s", url)


def _handle_deep_upload(result, user_query, conversation_id, agent_id, settings):
    """Upload multiple documents for DEEP pipeline."""
    all_docs = {}  # {doc_id: doc_name}
    api_key = settings.elevenlabs_api_key
    query_short = user_query[:60]
    conv_short = conversation_id[:8]

    # Per-study documents
    for study in result.studies:
        if not study.synthesis:
            continue
        doc_name = f"Study: {study.title[:60]} - {query_short} ({conv_short})"
        try:
            doc_id = _upload_with_retry(text=study.synthesis, name=doc_name, api_key=api_key)
            if doc_id:
                study.doc_id = doc_id
                all_docs[doc_id] = doc_name
        except Exception:
            logger.exception("Failed to upload study: %s", study.title)

    # Master synthesis
    if result.master_synthesis:
        doc_name = f"Master Briefing: {query_short} ({conv_short})"
        try:
            doc_id = _upload_with_retry(text=result.master_synthesis, name=doc_name, api_key=api_key)
            if doc_id:
                result.master_doc_id = doc_id
                all_docs[doc_id] = doc_name
        except Exception:
            logger.exception("Failed to upload master synthesis")

    # Q&A cluster documents
    for cluster in result.qa_clusters:
        if not cluster.findings:
            continue
        doc_name = f"Q&A: {cluster.theme[:60]} - {query_short} ({conv_short})"
        try:
            doc_id = _upload_with_retry(text=cluster.findings, name=doc_name, api_key=api_key)
            if doc_id:
                cluster.doc_id = doc_id
                all_docs[doc_id] = doc_name
        except Exception:
            logger.exception("Failed to upload Q&A cluster: %s", cluster.theme)

    # Q&A summary
    if result.qa_summary:
        doc_name = f"Anticipated Q&A: {query_short} ({conv_short})"
        try:
            doc_id = _upload_with_retry(text=result.qa_summary, name=doc_name, api_key=api_key)
            if doc_id:
                result.qa_summary_doc_id = doc_id
                all_docs[doc_id] = doc_name
        except Exception:
            logger.exception("Failed to upload Q&A summary")

    # Batch attach all documents
    if all_docs:
        try:
            elevenlabs_client.attach_documents_to_agent(
                agent_id=agent_id,
                doc_map=all_docs,
                api_key=api_key,
            )
        except Exception:
            logger.exception("Failed to batch attach documents to agent")

    result.all_doc_ids = list(all_docs.keys())
    logger.info(
        "DEEP pipeline complete: %d documents uploaded and attached to agent %s",
        len(all_docs),
        agent_id,
    )

    if settings.gcs_results_bucket:
        url = gcs_client.publish_results(
            result, user_query, "deep", conversation_id, settings.gcs_results_bucket
        )
        if url:
            logger.info("Results page: %s", url)


def _build_consolidated_text(result, query: str, depth: str) -> str:
    """Combine all research outputs into a single text document for KB upload."""
    parts = [f"Research Briefing: {query}", f"Depth: {depth.upper()}", ""]

    if depth.upper() == "DEEP":
        if result.master_synthesis:
            parts.append("=== EXECUTIVE SUMMARY ===")
            parts.append(result.master_synthesis)
            parts.append("")

        if result.studies:
            for i, study in enumerate(result.studies, 1):
                if study.synthesis:
                    parts.append(f"=== STUDY {i}: {study.title} ===")
                    parts.append(study.synthesis)
                    parts.append("")

        for cluster in getattr(result, "qa_clusters", []):
            if cluster.findings:
                parts.append(f"=== Q&A: {cluster.theme} ===")
                parts.append(cluster.findings)
                parts.append("")

        if result.qa_summary:
            parts.append("=== ANTICIPATED Q&A SUMMARY ===")
            parts.append(result.qa_summary)
            parts.append("")
    else:
        if result.final_synthesis:
            parts.append(result.final_synthesis)

    return "\n".join(parts)


def run_research_for_ui(
    job_id: str,
    user_query: str,
    depth: ResearchDepth,
    settings: Settings,
) -> None:
    """Launch research in a daemon thread for the web UI.

    Updates job_tracker at each phase. Uploads a consolidated KB doc to ElevenLabs.
    """

    def _run():
        try:
            update_job(
                job_id,
                status=JobStatus.RUNNING,
                phase="Starting research pipeline",
            )
            logger.info(
                "UI research started: job=%s depth=%s query=%s",
                job_id,
                depth.value,
                user_query[:100],
            )

            # Execute ADK research pipeline
            update_job(job_id, phase=f"Running {depth.value.upper()} pipeline")
            result = asyncio.run(
                execute_research(query=user_query, context="", depth=depth)
            )

            # Upload consolidated KB doc to ElevenLabs
            elevenlabs_doc_id = ""
            if settings.elevenlabs_api_key:
                update_job(job_id, phase="Uploading to knowledge base")
                try:
                    consolidated = _build_consolidated_text(result, user_query, depth.value)
                    if consolidated.strip():
                        doc_name = f"Research: {user_query[:80]} ({job_id})"
                        elevenlabs_doc_id = _upload_with_retry(
                            text=consolidated,
                            name=doc_name,
                            api_key=settings.elevenlabs_api_key,
                        )
                        logger.info("Uploaded consolidated KB doc: %s", elevenlabs_doc_id)
                except Exception:
                    logger.exception("Failed to upload consolidated KB doc for job %s", job_id)

            # Auto-attach to all agents + trigger RAG indexing
            if elevenlabs_doc_id and settings.elevenlabs_api_key:
                update_job(job_id, phase="Assigning research to agents")
                doc_name = f"Research: {user_query[:80]} ({job_id})"
                for slug in AGENTS:
                    agent_id = get_agent_id(slug, settings)
                    if not agent_id:
                        continue
                    try:
                        elevenlabs_client.attach_document_to_agent(
                            agent_id=agent_id,
                            doc_id=elevenlabs_doc_id,
                            doc_name=doc_name,
                            api_key=settings.elevenlabs_api_key,
                        )
                        logger.info("Attached doc %s to agent %s (%s)", elevenlabs_doc_id, slug, agent_id)
                    except Exception:
                        logger.exception("Failed to attach doc to agent %s", slug)

                # Trigger RAG indexing
                try:
                    elevenlabs_client.trigger_rag_index(
                        doc_id=elevenlabs_doc_id,
                        api_key=settings.elevenlabs_api_key,
                    )
                except Exception:
                    logger.exception("Failed to trigger RAG index for doc %s", elevenlabs_doc_id)

            # Upload results to GCS
            update_job(job_id, phase="Uploading results")
            result_url = ""
            if settings.gcs_results_bucket:
                result_url = gcs_client.publish_results_with_metadata(
                    result,
                    user_query,
                    depth.value,
                    job_id,
                    settings.gcs_results_bucket,
                    elevenlabs_doc_id=elevenlabs_doc_id,
                )

            update_job(
                job_id,
                status=JobStatus.COMPLETED,
                phase="Complete",
                result_url=result_url,
                elevenlabs_doc_id=elevenlabs_doc_id,
                completed_at=datetime.now(timezone.utc).isoformat(),
            )
            logger.info("UI research complete: job=%s url=%s doc_id=%s", job_id, result_url, elevenlabs_doc_id)

        except Exception as e:
            logger.exception("UI research failed: job=%s", job_id)
            update_job(
                job_id,
                status=JobStatus.FAILED,
                phase="Failed",
                error=str(e),
                completed_at=datetime.now(timezone.utc).isoformat(),
            )

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()


def _upload_with_retry(text: str, name: str, api_key: str) -> str:
    """Upload to KB with exponential backoff on 5xx errors."""
    backoff = INITIAL_BACKOFF
    for attempt in range(MAX_RETRIES):
        try:
            return elevenlabs_client.upload_to_knowledge_base(
                text=text, name=name, api_key=api_key
            )
        except Exception as e:
            error_str = str(e)
            is_server_error = "500" in error_str or "502" in error_str or "503" in error_str
            if is_server_error and attempt < MAX_RETRIES - 1:
                logger.warning(
                    "KB upload attempt %d failed (5xx), retrying in %ds: %s",
                    attempt + 1,
                    backoff,
                    e,
                )
                time.sleep(backoff)
                backoff *= 2
            else:
                logger.error("KB upload failed on attempt %d: %s", attempt + 1, e)
                raise
    return ""
