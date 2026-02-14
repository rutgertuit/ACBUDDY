"""Memory extractor — extracts key findings worth remembering from research synthesis."""

import logging

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from app.agents.json_utils import parse_json_response

logger = logging.getLogger(__name__)

MODEL = "gemini-2.0-flash"
APP_NAME = "acbuddy_research"

EXTRACTOR_INSTRUCTION = """You are a memory curator for a research system. Extract the most important
and reusable findings from the research synthesis that would be valuable context for future research.

Focus on:
- Hard facts and statistics that are broadly useful
- Market trends and industry patterns
- Regulatory or policy insights
- Key relationships between entities
- Counter-intuitive or surprising findings

DO NOT extract:
- Opinions or speculation
- Highly specific details only relevant to the exact query
- Common knowledge

Output ONLY valid JSON:
{
  "memories": [
    {
      "type": "finding|pattern|fact|recommendation",
      "content": "Concise statement of the memory (1-2 sentences)",
      "tags": ["relevant", "keywords"]
    }
  ]
}

Extract at most 8 memories per research run. Quality over quantity.
No explanation, no markdown fences, just the JSON."""


async def extract_memories(text: str) -> list[dict]:
    """Extract memorable findings from research synthesis.

    Args:
        text: Research synthesis text (truncated to 15K chars).

    Returns:
        List of memory dicts with type, content, tags.
    """
    if not text:
        return []

    session_service = InMemorySessionService()

    agent = LlmAgent(
        name="memory_extractor",
        model=MODEL,
        instruction=EXTRACTOR_INSTRUCTION,
        output_key="memories_raw",
    )

    runner = Runner(agent=agent, app_name=APP_NAME, session_service=session_service)
    session = session_service.create_session(app_name=APP_NAME, user_id="system")
    content = types.Content(
        role="user",
        parts=[types.Part(text=f"Extract memorable findings:\n\n{text[:15000]}")],
    )

    result_text = ""
    async for event in runner.run_async(
        user_id="system", session_id=session.id, new_message=content
    ):
        if event.is_final_response() and event.content and event.content.parts:
            result_text = event.content.parts[0].text

    if not result_text:
        session = session_service.get_session(
            app_name=APP_NAME, user_id="system", session_id=session.id
        )
        if session:
            result_text = session.state.get("memories_raw", "")

    parsed = parse_json_response(result_text)
    if isinstance(parsed, dict) and "memories" in parsed:
        return parsed["memories"]
    if isinstance(parsed, list):
        return parsed

    logger.warning("Failed to parse memory extraction: %s", str(result_text)[:200])
    return []
