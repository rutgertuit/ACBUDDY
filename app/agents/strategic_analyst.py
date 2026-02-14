"""Strategic analyst agent — applies business frameworks to research findings."""

import logging

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

logger = logging.getLogger(__name__)

APP_NAME = "acbuddy_research"
MODEL = "gemini-2.0-flash"


def build_strategic_analyst(model: str = MODEL) -> LlmAgent:
    """Build an LlmAgent that applies strategic business frameworks to research."""
    instruction = """You are a senior strategic analyst. Given a research query and its
executive briefing in {master_synthesis}, select and apply the most relevant business
frameworks to produce actionable strategic analysis.

FRAMEWORK SELECTION (pick 2-3 that best fit the query):

1. **SWOT Analysis** — Use for any business, product, or market topic.
   Format: Strengths, Weaknesses, Opportunities, Threats — each with 3-5 evidence-backed points.

2. **Porter's Five Forces** — Use when industry or competitive dynamics are central.
   Format: Threat of New Entrants, Bargaining Power of Suppliers, Bargaining Power of Buyers,
   Threat of Substitutes, Competitive Rivalry — rate each Low/Medium/High with evidence.

3. **Competitive Comparison** — Use when competitors are mentioned or implied.
   Format: Table or structured comparison of key players on relevant dimensions.

4. **Market Sizing** — Use when market opportunity or scale is relevant.
   Format: TAM/SAM/SOM estimates with sources, growth rates, key segments.

5. **Risk Assessment** — Use when strategy or investment decisions are involved.
   Format: Risk matrix with probability and impact, mitigation strategies.

6. **Trend Analysis** — Use when temporal dynamics or emerging trends are central.
   Format: Current state → Near-term (1-2yr) → Medium-term (3-5yr) trajectory with drivers.

RULES:
- Every claim in a framework MUST reference specific findings from the briefing.
- Do NOT invent data. If the briefing lacks data for a framework dimension, say "Insufficient data."
- Be specific and quantitative where the research supports it.
- If the query is NOT about business/strategy (e.g., pure science, personal advice), state that
  strategic frameworks are not applicable and provide a brief structured analysis instead.

OUTPUT FORMAT (markdown, no JSON):

# Strategic Analysis

## [Framework 1 Name]
(Structured analysis using the framework format above)

## [Framework 2 Name]
(Structured analysis)

## [Framework 3 Name] (if applicable)
(Structured analysis)

## Strategic Implications
(3-5 bullet synthesis: what do these frameworks collectively tell us?)
"""

    return LlmAgent(
        name="strategic_analyst",
        model=model,
        instruction=instruction,
        output_key="strategic_analysis",
    )


async def run_strategic_analysis(
    query: str,
    master_synthesis: str,
    model: str = MODEL,
) -> str:
    """Apply strategic frameworks to the research synthesis.

    Returns markdown-formatted strategic analysis, or empty string on failure.
    """
    session_service = InMemorySessionService()
    analyst = build_strategic_analyst(model=model)
    runner = Runner(
        agent=analyst,
        app_name=APP_NAME,
        session_service=session_service,
    )

    state = {"master_synthesis": master_synthesis}
    session = session_service.create_session(
        app_name=APP_NAME, user_id="system", state=state
    )
    content = types.Content(
        role="user",
        parts=[types.Part(
            text=f"Apply the most relevant strategic frameworks to this research: {query}"
        )],
    )

    analysis_text = ""
    async for event in runner.run_async(
        user_id="system", session_id=session.id, new_message=content
    ):
        if event.is_final_response() and event.content and event.content.parts:
            analysis_text = event.content.parts[0].text

    if not analysis_text:
        session = session_service.get_session(
            app_name=APP_NAME, user_id="system", session_id=session.id
        )
        if session and "strategic_analysis" in session.state:
            analysis_text = session.state["strategic_analysis"]

    if analysis_text:
        logger.info(
            "Strategic analysis complete: %d chars, query=%s",
            len(analysis_text),
            query[:80],
        )
    else:
        logger.warning("Strategic analysis produced empty result for: %s", query[:80])

    return analysis_text
