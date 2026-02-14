"""Gemini Deep Research — autonomous multi-step research via Interactions API.

Uses the deep-research-pro-preview-12-2025 agent for thorough, cited research.
Takes 2-10 minutes per study, costs ~$2-5 per call.
"""

import asyncio
import logging
import os
import time

logger = logging.getLogger(__name__)

AGENT_ID = "deep-research-pro-preview-12-2025"
POLL_INTERVAL = 10  # seconds between status checks
MAX_WAIT = 600  # 10 minute timeout


async def run_deep_research(query: str, context: str = "") -> str:
    """Execute a Gemini Deep Research query and return the synthesized report.

    This is an async wrapper around the blocking Interactions API poll loop.

    Args:
        query: Research question or study topic.
        context: Optional context to include (e.g., study angle, prior findings).

    Returns:
        Full research report text with citations, or empty string on failure.
    """
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        logger.warning("Gemini Deep Research: No GOOGLE_API_KEY configured")
        return ""

    prompt = query
    if context:
        prompt = f"Context:\n{context[:5000]}\n\nResearch question: {query}"

    # Run the blocking API call in a thread to keep async compatibility
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None, _run_deep_research_sync, api_key, prompt
        )
        return result
    except Exception:
        logger.exception("Gemini Deep Research failed for: %s", query[:100])
        return ""


def _run_deep_research_sync(api_key: str, prompt: str) -> str:
    """Synchronous Gemini Deep Research execution with polling."""
    from google import genai

    client = genai.Client(api_key=api_key)

    logger.info("Starting Gemini Deep Research: %s", prompt[:100])
    start = time.time()

    try:
        # Create the interaction (background=True for async execution)
        interaction = client.interactions.create(
            input=prompt,
            agent=AGENT_ID,
            background=True,
        )

        interaction_id = interaction.id
        logger.info("Deep Research interaction created: %s", interaction_id)

        # Poll for completion
        while (time.time() - start) < MAX_WAIT:
            time.sleep(POLL_INTERVAL)

            interaction = client.interactions.get(interaction_id)
            status = getattr(interaction, "status", "unknown")

            elapsed = int(time.time() - start)
            logger.debug("Deep Research %s: status=%s (%ds elapsed)", interaction_id, status, elapsed)

            if status == "completed":
                # Extract the final output
                outputs = getattr(interaction, "outputs", [])
                if outputs:
                    text = outputs[-1].text if hasattr(outputs[-1], "text") else str(outputs[-1])
                    logger.info(
                        "Deep Research complete: %d chars in %ds",
                        len(text), elapsed,
                    )
                    return text
                logger.warning("Deep Research completed but no outputs found")
                return ""

            if status in ("failed", "cancelled", "expired"):
                logger.error("Deep Research %s: %s after %ds", interaction_id, status, elapsed)
                return ""

        logger.error("Deep Research timed out after %ds", MAX_WAIT)
        return ""

    except AttributeError:
        # Interactions API not available in this version of the SDK
        logger.warning(
            "Gemini Interactions API not available (SDK may need upgrade). "
            "Install: pip install google-genai>=1.0.0"
        )
        return ""
    except Exception:
        logger.exception("Deep Research execution failed")
        return ""
