"""Amendment researcher agent — runs targeted research on new questions
given the context of existing research, without repeating known findings."""

import asyncio
import logging

from google.adk.agents import LlmAgent, ParallelAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from app.agents.deep_research import build_researcher
from app.models.research_result import ResearchResult

logger = logging.getLogger(__name__)

MODEL = "gemini-2.0-flash"
APP_NAME = "acbuddy_research"
MAX_CONCURRENT_AMENDMENTS = 3


async def execute_amendment(
    original_query: str,
    original_synthesis: str,
    additional_questions: list[str],
    perspective: str = "",
    on_progress=None,
) -> ResearchResult:
    """Run an amendment pipeline on top of existing research.

    Args:
        original_query: The original research query.
        original_synthesis: The full text of the original research synthesis.
        additional_questions: New questions or areas to explore.
        perspective: Optional new perspective or focus.
        on_progress: Optional callback(phase, **kwargs).

    Returns:
        ResearchResult with amendment synthesis in final_synthesis.
    """
    def _progress(phase, **kwargs):
        if on_progress:
            try:
                on_progress(phase, **kwargs)
            except Exception:
                pass

    result = ResearchResult(original_query=original_query)
    session_service = InMemorySessionService()

    # Phase 1: Research the new questions in parallel
    _progress("Researching new questions", step="studies")
    logger.info("Amendment: researching %d new questions", len(additional_questions))

    sem = asyncio.Semaphore(MAX_CONCURRENT_AMENDMENTS)

    async def _research_question(idx, question):
        async with sem:
            try:
                svc = InMemorySessionService()
                researcher = build_researcher(idx, model=MODEL, prefix="amendment")
                runner = Runner(agent=researcher, app_name=APP_NAME, session_service=svc)
                sess = svc.create_session(app_name=APP_NAME, user_id="system")
                msg = types.Content(
                    role="user",
                    parts=[types.Part(text=question)],
                )
                result_text = ""
                async for event in runner.run_async(
                    user_id="system", session_id=sess.id, new_message=msg
                ):
                    if event.is_final_response() and event.content and event.content.parts:
                        result_text = event.content.parts[0].text

                if not result_text:
                    sess = svc.get_session(app_name=APP_NAME, user_id="system", session_id=sess.id)
                    if sess:
                        result_text = sess.state.get(f"amendment_{idx}", "")

                return result_text
            except Exception:
                logger.exception("Amendment research %d failed: %s", idx, question[:60])
                return ""

    tasks = [_research_question(i, q) for i, q in enumerate(additional_questions)]
    findings = await asyncio.gather(*tasks)
    findings = [f for f in findings if f]

    if not findings:
        logger.warning("All amendment research failed")
        result.final_synthesis = "Amendment research produced no new findings."
        return result

    # Phase 2: Synthesize amendment
    _progress("Synthesizing amendment", step="synthesis")
    logger.info("Amendment: synthesizing %d findings", len(findings))

    findings_refs = "\n".join(
        f"- Finding {i+1}: {{amendment_finding_{i}}}"
        for i in range(len(findings))
    )

    perspective_note = ""
    if perspective:
        perspective_note = f"\n\nNew perspective/focus requested: {perspective}"

    synth_instruction = f"""You are an amendment researcher. You have access to existing research
and new targeted findings. Produce an AMENDMENT that adds to the original research.

ORIGINAL RESEARCH CONTEXT (do NOT repeat this — only add NEW insights):
{{original_synthesis}}

NEW RESEARCH FINDINGS:
{findings_refs}
{perspective_note}

RULES:
- Only include genuinely new information not already covered in the original research.
- Explicitly reference how new findings relate to, extend, or modify the original research.
- If new findings contradict the original, highlight the contradiction with evidence.
- Cite all sources with URLs.

Format as:

# Research Amendment

## New Questions Addressed
(List the questions this amendment researched)

## New Findings
(Detailed new findings organized by question, with sources)

## Impact on Original Research
(How these findings modify, extend, or reinforce the original conclusions)

## Updated Recommendations
(Any new or revised recommendations based on the combined research)

## Sources
(All new sources cited)"""

    synth_agent = LlmAgent(
        name="amendment_synthesizer",
        model=MODEL,
        instruction=synth_instruction,
        output_key="amendment_synthesis",
    )

    synth_runner = Runner(
        agent=synth_agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

    state = {"original_synthesis": original_synthesis[:30000]}  # cap context
    for i, f in enumerate(findings):
        state[f"amendment_finding_{i}"] = f

    sess = session_service.create_session(
        app_name=APP_NAME, user_id="system", state=state
    )
    msg = types.Content(
        role="user",
        parts=[types.Part(text=f"Create an amendment for: {original_query}")],
    )

    async for event in synth_runner.run_async(
        user_id="system", session_id=sess.id, new_message=msg
    ):
        if event.is_final_response() and event.content and event.content.parts:
            result.final_synthesis = event.content.parts[0].text

    if not result.final_synthesis:
        sess = session_service.get_session(
            app_name=APP_NAME, user_id="system", session_id=sess.id
        )
        if sess:
            result.final_synthesis = sess.state.get("amendment_synthesis", "")

    logger.info("Amendment complete: %d chars", len(result.final_synthesis))
    return result
