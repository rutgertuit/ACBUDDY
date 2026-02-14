"""Watch checker agent — lightweight QUICK search to detect changes for a watch."""

import logging

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from app.agents.deep_research import _web_search

logger = logging.getLogger(__name__)

MODEL = "gemini-2.0-flash"
APP_NAME = "acbuddy_research"

CHECKER_INSTRUCTION = """You are a research watch checker. Your job is to find the latest
information on a topic and summarize what's new or changed.

Do a quick but thorough web search and provide:
1. A brief summary of the current state of the topic (2-3 paragraphs)
2. Any recent developments or changes
3. Key data points or statistics

Focus on RECENT information. If you find significant new developments, highlight them clearly.

Output a clear, concise summary with sources."""


async def check_watch(query: str) -> str:
    """Run a lightweight check for a watch query.

    Args:
        query: The watch query to check.

    Returns:
        Summary text of current findings.
    """
    session_service = InMemorySessionService()

    agent = LlmAgent(
        name="watch_checker",
        model=MODEL,
        instruction=CHECKER_INSTRUCTION,
        tools=[_web_search],
        output_key="watch_findings",
    )

    runner = Runner(agent=agent, app_name=APP_NAME, session_service=session_service)
    session = session_service.create_session(app_name=APP_NAME, user_id="system")
    content = types.Content(
        role="user",
        parts=[types.Part(text=f"Check for latest updates on: {query}")],
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
            result_text = session.state.get("watch_findings", "")

    return result_text or "No findings from watch check."
