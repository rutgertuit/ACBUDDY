import asyncio
import logging
import threading
import time
from datetime import datetime, timezone

from app.agents.agent_profiles import AGENTS, get_agent_id
from app.config import Settings
from app.models.depth import ResearchDepth
from app.services import elevenlabs_client, gcs_client
from app.services.job_tracker import (
    JobStatus, get_job, update_job, record_phase_timing, finalize_timings,
)
from app.services.research_stats import init_stats, get_stats, compute_human_hours
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

    # Attach to ALL agents, not just the triggering one
    for slug in AGENTS:
        aid = get_agent_id(slug, settings)
        if not aid:
            continue
        try:
            elevenlabs_client.attach_document_to_agent(
                agent_id=aid,
                doc_id=doc_id,
                doc_name=doc_name,
                api_key=settings.elevenlabs_api_key,
            )
            logger.info("Attached doc %s to agent %s (%s)", doc_id, slug, aid)
        except Exception:
            logger.exception("Failed to attach doc to agent %s", slug)

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

    # Batch attach all documents to ALL agents
    if all_docs:
        for slug in AGENTS:
            aid = get_agent_id(slug, settings)
            if not aid:
                continue
            try:
                elevenlabs_client.attach_documents_to_agent(
                    agent_id=aid,
                    doc_map=all_docs,
                    api_key=api_key,
                )
                logger.info("Attached %d docs to agent %s (%s)", len(all_docs), slug, aid)
            except Exception:
                logger.exception("Failed to batch attach documents to agent %s", slug)

    result.all_doc_ids = list(all_docs.keys())
    logger.info(
        "DEEP pipeline complete: %d documents uploaded and attached to all agents",
        len(all_docs),
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
        # Include quality scores if available
        if result.synthesis_score > 0:
            parts.append(f"Synthesis Quality Score: {result.synthesis_score:.1f}/10")
            if result.synthesis_scores:
                score_parts = ", ".join(f"{k}: {v}/10" for k, v in result.synthesis_scores.items())
                parts.append(f"Dimension Scores: {score_parts}")
            if result.refinement_rounds > 0:
                parts.append(f"Refinement Rounds: {result.refinement_rounds}")
            parts.append("")

        if result.master_synthesis:
            parts.append("=== EXECUTIVE SUMMARY ===")
            parts.append(result.master_synthesis)
            parts.append("")

        if result.strategic_analysis:
            parts.append("=== STRATEGIC ANALYSIS ===")
            parts.append(result.strategic_analysis)
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
    business_context: dict | None = None,
) -> None:
    """Launch research in a daemon thread for the web UI.

    Updates job_tracker at each phase. Uploads a consolidated KB doc to ElevenLabs.
    """

    def _run():
        try:
            init_stats(job_id=job_id)
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

            # Progress callback for DEEP pipeline
            def _on_progress(phase, **kwargs):
                updates = {"phase": phase}
                if "step" in kwargs:
                    updates["current_step"] = kwargs["step"]
                    # Record phase timing (normalize study_N -> studies)
                    step = kwargs["step"]
                    timing_key = "studies" if step.startswith("study_") else ("refinement" if step.startswith("gap_study_") else step)
                    record_phase_timing(job_id, timing_key)
                if "study_plan" in kwargs:
                    updates["study_plan"] = kwargs["study_plan"]
                if "study_progress" in kwargs:
                    updates["study_progress"] = kwargs["study_progress"]
                # Update individual study status
                if "study_idx" in kwargs and "study_status" in kwargs:
                    job = get_job(job_id)
                    if job and job.study_progress:
                        idx = kwargs["study_idx"]
                        if idx < len(job.study_progress):
                            job.study_progress[idx]["status"] = kwargs["study_status"]
                update_job(job_id, **updates)

            # Execute ADK research pipeline
            update_job(job_id, phase=f"Running {depth.value.upper()} pipeline")
            result = asyncio.run(
                execute_research(query=user_query, context="", depth=depth,
                                 on_progress=_on_progress,
                                 gcs_bucket=settings.gcs_results_bucket,
                                 business_context=business_context)
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

            # Capture final research stats
            final_stats = get_stats()
            num_studies = len(result.studies) if result.studies else 0
            num_qa = len([c for c in result.qa_clusters if c.findings]) if result.qa_clusters else 0
            human_hours = compute_human_hours(
                final_stats, num_studies=num_studies,
                num_qa_clusters=num_qa, depth=depth.value,
            )
            update_job(job_id, research_stats={**final_stats, "human_hours": human_hours})

            # Finalize phase timings
            record_phase_timing(job_id, "upload")
            timings = finalize_timings(job_id)

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
                    phase_timings=timings,
                    research_stats={**final_stats, "human_hours": human_hours},
                )

            # Extract memories + entities in parallel (Feature 9: performance)
            # Skip for QUICK depth to save API calls
            if depth.value.upper() != "QUICK":
                consolidated = _build_consolidated_text(result, user_query, depth.value)
                if consolidated.strip():
                    update_job(job_id, phase="Extracting memories & knowledge graph")

                    async def _extract_both(text, _settings):
                        """Run memory and entity extraction in parallel."""
                        from app.agents.memory_extractor import extract_memories
                        from app.agents.entity_extractor import extract_entities
                        mem_task = extract_memories(text[:15000])
                        ent_task = extract_entities(text[:20000])
                        return await asyncio.gather(mem_task, ent_task)

                    try:
                        memories, extraction = asyncio.run(
                            _extract_both(consolidated, settings)
                        )

                        # Save memories
                        if memories:
                            try:
                                from app.services import memory_store
                                store = memory_store.load_memory(settings.gcs_results_bucket)
                                added = memory_store.add_memories(store, memories, job_id, user_query)
                                memory_store.save_memory(store, settings.gcs_results_bucket)
                                logger.info("Added %d memories from job %s", added, job_id)
                            except Exception:
                                logger.exception("Memory save failed (non-fatal)")

                        # Save entities
                        if extraction and extraction.get("entities"):
                            try:
                                from app.services import knowledge_graph as kg
                                graph = kg.load_graph(settings.gcs_results_bucket)
                                kg.merge_extraction(graph, extraction, job_id)
                                kg.save_graph(graph, settings.gcs_results_bucket)
                                logger.info(
                                    "Knowledge graph updated: +%d entities, +%d relationships",
                                    len(extraction.get("entities", [])),
                                    len(extraction.get("relationships", [])),
                                )
                            except Exception:
                                logger.exception("Knowledge graph save failed (non-fatal)")
                    except Exception:
                        logger.exception("Parallel extraction failed (non-fatal)")

            update_job(
                job_id,
                status=JobStatus.COMPLETED,
                phase="Complete",
                result_url=result_url,
                elevenlabs_doc_id=elevenlabs_doc_id,
                completed_at=datetime.now(timezone.utc).isoformat(),
            )
            logger.info("UI research complete: job=%s url=%s doc_id=%s timings=%s", job_id, result_url, elevenlabs_doc_id, timings)

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


def run_amendment_for_ui(
    job_id: str,
    parent_job_id: str,
    original_query: str,
    additional_questions: list[str],
    perspective: str,
    settings: Settings,
) -> None:
    """Launch amendment pipeline in a daemon thread for the web UI."""

    def _run():
        try:
            update_job(
                job_id,
                status=JobStatus.RUNNING,
                phase="Preparing amendment",
            )
            logger.info("Amendment started: job=%s parent=%s", job_id, parent_job_id)

            # Fetch original result from GCS
            original_synthesis = ""
            if settings.gcs_results_bucket:
                meta = gcs_client.get_result_metadata(parent_job_id, settings.gcs_results_bucket)
                if meta and meta.get("result_url"):
                    # Download HTML and extract text
                    try:
                        import requests, re
                        resp = requests.get(meta["result_url"], timeout=30)
                        resp.raise_for_status()
                        text = re.sub(r"<[^>]+>", " ", resp.text)
                        text = re.sub(r"\s+", " ", text).strip()
                        original_synthesis = text[:40000]
                    except Exception:
                        logger.warning("Failed to fetch original result HTML, using metadata")

            if not original_synthesis:
                original_synthesis = f"Original research query: {original_query}"

            # Progress callback
            def _on_progress(phase, **kwargs):
                updates = {"phase": phase}
                if "step" in kwargs:
                    updates["current_step"] = kwargs["step"]
                    step = kwargs["step"]
                    timing_key = "studies" if step.startswith("study_") else ("refinement" if step.startswith("gap_study_") else step)
                    record_phase_timing(job_id, timing_key)
                update_job(job_id, **updates)

            # Execute amendment pipeline
            from app.agents.amendment_researcher import execute_amendment
            result = asyncio.run(
                execute_amendment(
                    original_query=original_query,
                    original_synthesis=original_synthesis,
                    additional_questions=additional_questions,
                    perspective=perspective,
                    on_progress=_on_progress,
                )
            )

            # Upload to ElevenLabs KB
            elevenlabs_doc_id = ""
            if result.final_synthesis and settings.elevenlabs_api_key:
                update_job(job_id, phase="Uploading to knowledge base")
                try:
                    doc_name = f"Amendment: {original_query[:60]} ({job_id})"
                    elevenlabs_doc_id = _upload_with_retry(
                        text=result.final_synthesis,
                        name=doc_name,
                        api_key=settings.elevenlabs_api_key,
                    )
                except Exception:
                    logger.exception("Failed to upload amendment KB doc")

            # Auto-attach to all agents
            if elevenlabs_doc_id and settings.elevenlabs_api_key:
                update_job(job_id, phase="Assigning to agents")
                doc_name = f"Amendment: {original_query[:60]} ({job_id})"
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
                    except Exception:
                        logger.exception("Failed to attach amendment to agent %s", slug)
                try:
                    elevenlabs_client.trigger_rag_index(
                        doc_id=elevenlabs_doc_id,
                        api_key=settings.elevenlabs_api_key,
                    )
                except Exception:
                    logger.exception("Failed to trigger RAG index for amendment")

            # Finalize timings
            record_phase_timing(job_id, "upload")
            timings = finalize_timings(job_id)

            # Upload to GCS
            result_url = ""
            if settings.gcs_results_bucket and result.final_synthesis:
                update_job(job_id, phase="Uploading results")
                result_url = gcs_client.publish_results_with_metadata(
                    result,
                    f"Amendment of: {original_query}",
                    "standard",
                    job_id,
                    settings.gcs_results_bucket,
                    elevenlabs_doc_id=elevenlabs_doc_id,
                    phase_timings=timings,
                )
                # Update metadata with parent link
                gcs_client.update_metadata(job_id, settings.gcs_results_bucket, {
                    "parent_job_id": parent_job_id,
                    "amendment": True,
                })

            update_job(
                job_id,
                status=JobStatus.COMPLETED,
                phase="Complete",
                result_url=result_url,
                elevenlabs_doc_id=elevenlabs_doc_id,
                completed_at=datetime.now(timezone.utc).isoformat(),
            )
            logger.info("Amendment complete: job=%s url=%s", job_id, result_url)

        except Exception as e:
            logger.exception("Amendment failed: job=%s", job_id)
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
