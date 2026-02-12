import asyncio
import logging
import time

from app.config import Settings
from app.services import elevenlabs_client
from app.agents.root_agent import execute_research

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
INITIAL_BACKOFF = 2  # seconds


def run_research_pipeline(
    conversation_id: str,
    agent_id: str,
    user_query: str,
    settings: Settings,
) -> None:
    """Run the full research pipeline in a background thread.

    1. Fetch conversation context from ElevenLabs
    2. Execute ADK research pipeline
    3. Upload result to ElevenLabs Knowledge Base
    4. Attach document to agent

    All exceptions are caught and logged (no re-raise in background thread).
    """
    try:
        logger.info(
            "Starting research pipeline: conversation=%s query=%s",
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
        result = asyncio.run(execute_research(query=user_query, context=context))

        if not result.final_synthesis:
            logger.error("Research pipeline produced empty synthesis")
            return

        logger.info(
            "Research complete: %d questions, %d findings, %d follow-ups, synthesis=%d chars",
            len(result.unpacked_questions),
            len(result.research_findings),
            len(result.follow_up_findings),
            len(result.final_synthesis),
        )

        # 3. Upload to Knowledge Base with retry
        doc_name = f"Research: {user_query[:80]} ({conversation_id[:8]})"
        doc_id = _upload_with_retry(
            text=result.final_synthesis,
            name=doc_name,
            api_key=settings.elevenlabs_api_key,
        )

        if not doc_id:
            logger.error("Failed to upload research to KB after retries")
            return

        # 4. Attach document to agent
        elevenlabs_client.attach_document_to_agent(
            agent_id=agent_id,
            doc_id=doc_id,
            api_key=settings.elevenlabs_api_key,
        )
        logger.info(
            "Research pipeline complete: doc=%s attached to agent=%s",
            doc_id,
            agent_id,
        )

    except Exception:
        logger.exception(
            "Research pipeline failed for conversation %s", conversation_id
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
