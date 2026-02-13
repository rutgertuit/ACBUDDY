import asyncio
import logging
import time

from app.config import Settings
from app.models.depth import ResearchDepth
from app.services import elevenlabs_client
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
