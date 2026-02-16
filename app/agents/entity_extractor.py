"""Entity extractor — extracts entities and relationships from research synthesis
for knowledge graph construction."""

import logging

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from app.agents.json_utils import parse_json_response

logger = logging.getLogger(__name__)

MODEL = "gemini-2.5-flash"
APP_NAME = "luminary_research"

EXTRACTOR_INSTRUCTION = """You are an entity and relationship extractor for a knowledge graph.

Given a research synthesis, extract:
1. Named entities (companies, people, products, concepts, technologies, regulations, markets)
2. Relationships between entities

Output ONLY valid JSON:
{
  "entities": [
    {"name": "Entity Name", "type": "company|person|product|concept|technology|regulation|market", "aliases": ["alt names"]}
  ],
  "relationships": [
    {"from": "Entity A", "to": "Entity B", "type": "competes_with|produces|regulates|partners_with|acquired|invests_in|uses|part_of|affects", "description": "brief description"}
  ]
}

Rules:
- Extract at most 30 entities and 40 relationships.
- Normalize entity names (e.g., "Google LLC" -> "Google").
- Include aliases for commonly known alternate names.
- Only extract relationships that are clearly stated or strongly implied in the text.
- Relationship types: competes_with, produces, regulates, partners_with, acquired, invests_in, uses, part_of, affects.

No explanation, no markdown fences, just the JSON."""


async def extract_entities(text: str) -> dict:
    """Extract entities and relationships from research synthesis text.

    Args:
        text: The research synthesis text (will be truncated to 20K chars).

    Returns:
        Dict with "entities" and "relationships" lists, or empty dict on failure.
    """
    if not text:
        return {"entities": [], "relationships": []}

    session_service = InMemorySessionService()

    agent = LlmAgent(
        name="entity_extractor",
        model=MODEL,
        instruction=EXTRACTOR_INSTRUCTION,
        output_key="extraction",
    )

    runner = Runner(agent=agent, app_name=APP_NAME, session_service=session_service)
    session = session_service.create_session(app_name=APP_NAME, user_id="system")
    content = types.Content(
        role="user",
        parts=[types.Part(text=f"Extract entities and relationships:\n\n{text[:20000]}")],
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
            result_text = session.state.get("extraction", "")

    parsed = parse_json_response(result_text)
    if isinstance(parsed, dict) and ("entities" in parsed or "relationships" in parsed):
        return {
            "entities": parsed.get("entities", []),
            "relationships": parsed.get("relationships", []),
        }

    logger.warning("Failed to parse entity extraction output: %s", str(result_text)[:200])
    return {"entities": [], "relationships": []}
