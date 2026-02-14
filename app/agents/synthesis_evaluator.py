"""Synthesis evaluator agent — grades a master synthesis and identifies gaps."""

import logging

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from app.agents.json_utils import parse_json_response

logger = logging.getLogger(__name__)

APP_NAME = "acbuddy_research"
MODEL = "gemini-2.5-flash"

# Minimum overall score to skip refinement
REFINEMENT_THRESHOLD = 8.0


def build_evaluator(model: str = MODEL) -> LlmAgent:
    """Build an LlmAgent that evaluates a master synthesis for gaps and quality."""
    instruction = """You are a rigorous research quality evaluator. You assess executive research
briefings and identify specific gaps, weak claims, and missing perspectives.

Review the master synthesis in {master_synthesis} for the research query described by the user.

Evaluate on these dimensions (each 1-10):
- **completeness**: Are all aspects of the query thoroughly covered?
- **evidence_quality**: Are claims backed by specific data, numbers, named sources, URLs?
- **actionability**: Could a business leader make decisions based on this?
- **balance**: Are multiple perspectives, counterarguments, and risks represented?

Then identify:
- **gaps**: Specific missing information that would improve the briefing (max 4)
- **weak_claims**: Statements that lack quantitative backing or source citation
- **missing_perspectives**: Viewpoints not represented (e.g., regulatory, customer, competitor)

Respond ONLY in JSON (no markdown fences, no preamble):
{
  "overall_score": 7.2,
  "scores": {
    "completeness": 7,
    "evidence_quality": 8,
    "actionability": 6,
    "balance": 7
  },
  "gaps": [
    {
      "description": "No data on competitor pricing",
      "priority": "high",
      "research_question": "What are competitor pricing strategies in this market?"
    }
  ],
  "weak_claims": [
    "The statement 'most consumers prefer...' lacks quantitative backing"
  ],
  "missing_perspectives": [
    "Regulatory/compliance angle not covered"
  ],
  "refinement_needed": true
}

Be strict. Business strategy research needs strong evidence. Score honestly —
most first-pass syntheses will score 5-7 and need refinement.
Only set refinement_needed=false if overall_score >= 8 AND no high-priority gaps exist."""

    return LlmAgent(
        name="synthesis_evaluator",
        model=model,
        instruction=instruction,
        output_key="synthesis_evaluation",
    )


async def evaluate_synthesis(
    query: str,
    master_synthesis: str,
    model: str = MODEL,
) -> dict:
    """Evaluate a master synthesis and return structured evaluation.

    Returns dict with: overall_score, scores, gaps, weak_claims,
    missing_perspectives, refinement_needed.
    """
    session_service = InMemorySessionService()
    evaluator = build_evaluator(model=model)
    runner = Runner(
        agent=evaluator,
        app_name=APP_NAME,
        session_service=session_service,
    )

    state = {"master_synthesis": master_synthesis}
    session = session_service.create_session(
        app_name=APP_NAME, user_id="system", state=state
    )
    content = types.Content(
        role="user",
        parts=[types.Part(text=f"Evaluate the master synthesis for this research query: {query}")],
    )

    eval_text = ""
    async for event in runner.run_async(
        user_id="system", session_id=session.id, new_message=content
    ):
        if event.is_final_response() and event.content and event.content.parts:
            eval_text = event.content.parts[0].text

    if not eval_text:
        session = session_service.get_session(
            app_name=APP_NAME, user_id="system", session_id=session.id
        )
        if session and "synthesis_evaluation" in session.state:
            eval_text = session.state["synthesis_evaluation"]

    evaluation = parse_json_response(eval_text)
    if not isinstance(evaluation, dict):
        logger.warning("Failed to parse evaluation, returning default (needs refinement)")
        return {
            "overall_score": 5.0,
            "scores": {},
            "gaps": [],
            "weak_claims": [],
            "missing_perspectives": [],
            "refinement_needed": True,
        }

    # Ensure refinement_needed is set correctly
    overall = evaluation.get("overall_score", 5.0)
    has_high_gaps = any(
        g.get("priority") == "high" for g in evaluation.get("gaps", [])
    )
    evaluation["refinement_needed"] = overall < REFINEMENT_THRESHOLD or has_high_gaps

    logger.info(
        "Synthesis evaluation: score=%.1f, gaps=%d, weak_claims=%d, refine=%s",
        overall,
        len(evaluation.get("gaps", [])),
        len(evaluation.get("weak_claims", [])),
        evaluation["refinement_needed"],
    )
    return evaluation
