import asyncio
import json
import logging

from google.adk.agents import ParallelAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from app.agents.deep_research import build_researcher
from app.agents.gap_analyzer import build_gap_analyzer
from app.agents.json_utils import parse_json_response
from app.agents.synthesizer import build_synthesizer
from app.models.research_result import StudyResult

logger = logging.getLogger(__name__)

APP_NAME = "acbuddy_research"
MODEL = "gemini-2.0-flash"
ROUND_MAX_RETRIES = 2
ROUND_RETRY_BACKOFF = 5


async def run_iterative_study(
    study_index: int,
    study: dict,
    session_service: InMemorySessionService,
    model: str = MODEL,
    max_rounds: int = 3,
) -> StudyResult:
    """Run iterative deep research for a single study.

    Each round: parallel researchers → gap analyzer → decide whether to continue.
    After all rounds: per-study synthesis.
    """
    title = study.get("title", f"Study {study_index}")
    angle = study.get("angle", "")
    questions = study.get("questions", [])
    if not questions:
        questions = [title]

    result = StudyResult(title=title, angle=angle, questions=questions)
    state = {}

    for round_idx in range(max_rounds):
        logger.info("Study %d '%s' — round %d with %d questions", study_index, title, round_idx, len(questions))

        # Build parallel researchers for this round's questions
        prefix = f"study_{study_index}_round_{round_idx}_researcher"
        researchers = [
            build_researcher(j, model=model, prefix=f"study_{study_index}_round_{round_idx}_researcher")
            for j in range(len(questions))
        ]

        if len(researchers) == 1:
            research_agent = researchers[0]
        else:
            research_agent = ParallelAgent(
                name=f"parallel_s{study_index}_r{round_idx}",
                sub_agents=researchers,
            )

        prompt = "Research the following questions:\n" + "\n".join(
            f"{j+1}. {q}" for j, q in enumerate(questions)
        )

        for retry in range(ROUND_MAX_RETRIES + 1):
            try:
                runner = Runner(
                    agent=research_agent,
                    app_name=APP_NAME,
                    session_service=session_service,
                )
                session = session_service.create_session(
                    app_name=APP_NAME, user_id="system", state=dict(state)
                )
                content = types.Content(role="user", parts=[types.Part(text=prompt)])

                async for event in runner.run_async(
                    user_id="system", session_id=session.id, new_message=content
                ):
                    pass
                break  # success
            except Exception as e:
                error_str = str(e).lower()
                is_retryable = any(kw in error_str for kw in [
                    "connect", "timeout", "read", "reset", "429", "503", "unavailable",
                ])
                if is_retryable and retry < ROUND_MAX_RETRIES:
                    wait = ROUND_RETRY_BACKOFF * (retry + 1)
                    logger.warning(
                        "Study %d round %d research failed (attempt %d), retrying in %ds: %s",
                        study_index, round_idx, retry + 1, wait, e,
                    )
                    await asyncio.sleep(wait)
                else:
                    raise

        # Collect findings from session state
        session = session_service.get_session(
            app_name=APP_NAME, user_id="system", session_id=session.id
        )
        if session:
            state.update(session.state)

        round_findings = {}
        for j in range(len(questions)):
            key = f"study_{study_index}_round_{round_idx}_researcher_{j}"
            if key in state:
                round_findings[key] = state[key]
            else:
                # Ensure key exists so gap analyzer template doesn't crash
                state[key] = "No research findings available for this question."
                logger.warning("Researcher %s did not produce output", key)
        result.rounds.append(round_findings)

        # Gap analysis (skip on last round)
        if round_idx >= max_rounds - 1:
            break

        gap_text = ""
        for retry in range(ROUND_MAX_RETRIES + 1):
            try:
                gap_agent = build_gap_analyzer(study_index, round_idx, len(questions), model=model)
                gap_runner = Runner(
                    agent=gap_agent,
                    app_name=APP_NAME,
                    session_service=session_service,
                )
                gap_session = session_service.create_session(
                    app_name=APP_NAME, user_id="system", state=dict(state)
                )
                gap_prompt = f"Analyze research gaps for study: {title}"
                gap_content = types.Content(role="user", parts=[types.Part(text=gap_prompt)])

                async for event in gap_runner.run_async(
                    user_id="system", session_id=gap_session.id, new_message=gap_content
                ):
                    if event.is_final_response() and event.content and event.content.parts:
                        gap_text = event.content.parts[0].text
                break
            except Exception as e:
                error_str = str(e).lower()
                is_retryable = any(kw in error_str for kw in [
                    "connect", "timeout", "read", "reset", "429", "503", "unavailable",
                ])
                if is_retryable and retry < ROUND_MAX_RETRIES:
                    wait = ROUND_RETRY_BACKOFF * (retry + 1)
                    logger.warning(
                        "Study %d gap analysis round %d failed (attempt %d), retrying in %ds: %s",
                        study_index, round_idx, retry + 1, wait, e,
                    )
                    await asyncio.sleep(wait)
                else:
                    raise

        gap_session = session_service.get_session(
            app_name=APP_NAME, user_id="system", session_id=gap_session.id
        )
        if gap_session:
            state.update(gap_session.state)

        # Parse gap analysis
        gap_key = f"study_{study_index}_gaps_{round_idx}"
        raw = state.get(gap_key, gap_text)
        gap_data = parse_json_response(raw) if isinstance(raw, str) else raw

        if not isinstance(gap_data, dict) or gap_data.get("escalate", True):
            logger.info("Study %d — no more gaps after round %d", study_index, round_idx)
            break

        new_questions = gap_data.get("gaps", [])
        if not new_questions:
            break

        questions = new_questions[:3]
        logger.info("Study %d — %d gap questions for next round: %s", study_index, len(questions), questions)

    # Per-study synthesis
    logger.info("Study %d — synthesizing findings from %d rounds", study_index, len(result.rounds))

    # Count all researcher outputs for synthesis
    all_research_keys = []
    for round_findings in result.rounds:
        all_research_keys.extend(round_findings.keys())

    # Build a custom synthesizer for this study
    synth_refs = "\n".join(f"- {{{key}}}" for key in all_research_keys)
    from google.adk.agents import LlmAgent

    synth_instruction = f"""You are a research synthesizer for the study: "{title}"
Study angle: {angle}

Synthesize ALL the following research findings into a comprehensive study document:
{synth_refs}

IMPORTANT RULES:
- Only include findings that are backed by a specific, verifiable source URL. If a finding
  says "source could not be verified" or lacks a concrete URL, EXCLUDE it from the synthesis.
- Stay strictly within the geographic and topical scope of the original research query. Remove
  any data, examples, or references from outside the relevant geography (e.g., do not include
  German or UK broadcaster data in a study about the Netherlands).
- Prefer fewer, well-sourced insights over a long list of unverified claims.

SOURCE QUALITY RULES:
- Prioritize claims backed by multiple independent sources. If 3+ sources agree, note this.
- Weight authoritative domains higher: government (.gov), academic (.edu), major publications
  (Reuters, Bloomberg, FT, WSJ) > general web sources > blogs/forums.
- When a claim comes from a single source only, note: "(single source: [domain])".
- Flag potential bias from vendor reports, sponsored content, or advocacy sources.
- Tag each major finding with a confidence level:
  [HIGH CONFIDENCE] — 3+ independent credible sources
  [MEDIUM CONFIDENCE] — 1-2 credible sources
  [LOW CONFIDENCE] — single source, potentially biased, or conflicting data

Format your output as a professional study document with:
# {title}

## Overview
(2-3 paragraph summary of this study's findings)

## Detailed Findings
(Organized by subtopic with bullet points and data. Each major finding tagged with confidence level.)

## Source Reliability Notes
- High confidence: [findings backed by 3+ sources]
- Medium confidence: [findings from 1-2 credible sources]
- Low confidence / needs verification: [single or biased sources]

## Sources
(All URLs referenced — only include URLs that back claims used above)

## Key Takeaways
(3-5 actionable insights from this study, noting confidence level for each)

Write clearly, cite sources inline, be thorough."""

    synth_agent = LlmAgent(
        name=f"synthesizer_study_{study_index}",
        model=model,
        instruction=synth_instruction,
        output_key=f"study_{study_index}_synthesis",
    )

    for retry in range(ROUND_MAX_RETRIES + 1):
        try:
            synth_runner = Runner(
                agent=synth_agent,
                app_name=APP_NAME,
                session_service=session_service,
            )
            synth_session = session_service.create_session(
                app_name=APP_NAME, user_id="system", state=dict(state)
            )
            synth_content = types.Content(
                role="user",
                parts=[types.Part(text=f"Synthesize all findings for study: {title}")],
            )

            async for event in synth_runner.run_async(
                user_id="system", session_id=synth_session.id, new_message=synth_content
            ):
                if event.is_final_response() and event.content and event.content.parts:
                    result.synthesis = event.content.parts[0].text
            break
        except Exception as e:
            error_str = str(e).lower()
            is_retryable = any(kw in error_str for kw in [
                "connect", "timeout", "read", "reset", "429", "503", "unavailable",
            ])
            if is_retryable and retry < ROUND_MAX_RETRIES:
                wait = ROUND_RETRY_BACKOFF * (retry + 1)
                logger.warning(
                    "Study %d synthesis failed (attempt %d), retrying in %ds: %s",
                    study_index, retry + 1, wait, e,
                )
                await asyncio.sleep(wait)
            else:
                raise

    if not result.synthesis:
        synth_session = session_service.get_session(
            app_name=APP_NAME, user_id="system", session_id=synth_session.id
        )
        if synth_session:
            result.synthesis = synth_session.state.get(f"study_{study_index}_synthesis", "")
            state.update(synth_session.state)

    logger.info("Study %d '%s' complete — synthesis: %d chars", study_index, title, len(result.synthesis))
    return result
