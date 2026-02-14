import json
import logging
from typing import Optional

from google.adk.agents import ParallelAgent, SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from app.agents.json_utils import parse_json_response
from app.agents.question_unpacker import build_question_unpacker
from app.agents.deep_research import build_researcher
from app.agents.follow_up_handler import build_follow_up_identifier
from app.agents.synthesizer import build_synthesizer
from app.models.depth import ResearchDepth
from app.models.research_result import ResearchResult

logger = logging.getLogger(__name__)

MODEL = "gemini-2.0-flash"
APP_NAME = "acbuddy_research"


async def execute_research(
    query: str, context: str = "", depth: ResearchDepth = ResearchDepth.STANDARD,
    on_progress=None,
) -> ResearchResult:
    """Execute research pipeline at the specified depth.

    QUICK: Single researcher, no follow-ups.
    STANDARD: Sub-questions → parallel research → follow-ups → synthesis.
    DEEP: Multi-study iterative pipeline (delegated to deep_pipeline).
    """
    if depth == ResearchDepth.DEEP:
        from app.agents.deep_pipeline import execute_deep_research
        return await execute_deep_research(query=query, context=context, on_progress=on_progress)

    if depth == ResearchDepth.QUICK:
        return await _execute_quick_research(query=query, context=context)

    # ---- STANDARD pipeline ----
    result = ResearchResult(original_query=query)
    session_service = InMemorySessionService()

    # ---- Phase 1: Unpack questions ----
    unpacker = build_question_unpacker(model=MODEL)
    phase1_runner = Runner(
        agent=unpacker,
        app_name=APP_NAME,
        session_service=session_service,
    )

    prompt = f"Research query: {query}"
    if context:
        prompt = f"Conversation context:\n{context}\n\nResearch query: {query}"

    session = session_service.create_session(
        app_name=APP_NAME, user_id="system"
    )

    content = types.Content(
        role="user", parts=[types.Part(text=prompt)]
    )

    unpacked_text = ""
    async for event in phase1_runner.run_async(
        user_id="system", session_id=session.id, new_message=content
    ):
        if event.is_final_response() and event.content and event.content.parts:
            unpacked_text = event.content.parts[0].text
            break

    # Parse sub-questions from JSON (robust: handles markdown fences)
    sub_questions = parse_json_response(unpacked_text)
    if not isinstance(sub_questions, list) or not sub_questions:
        logger.warning("Failed to parse unpacker output, using original query")
        sub_questions = [query]

    # Limit to 5 sub-questions
    sub_questions = sub_questions[:5]
    result.unpacked_questions = sub_questions
    logger.info("Unpacked %d sub-questions: %s", len(sub_questions), sub_questions)

    # ---- Phase 2: Parallel research → follow-up → synthesis ----
    num_questions = len(sub_questions)

    # Build parallel researchers
    researchers = [
        build_researcher(i, model=MODEL, prefix="research")
        for i in range(num_questions)
    ]
    parallel_research = ParallelAgent(
        name="parallel_research",
        sub_agents=researchers,
    )

    # Build follow-up identifier
    follow_up_agent = build_follow_up_identifier(num_questions, model=MODEL)

    # Build phase 2 sequential pipeline (without follow-up research for now)
    # We'll add follow-up research dynamically after identifying gaps
    phase2_agents = [parallel_research, follow_up_agent]

    phase2_pipeline = SequentialAgent(
        name="research_pipeline",
        sub_agents=phase2_agents,
    )

    phase2_runner = Runner(
        agent=phase2_pipeline,
        app_name=APP_NAME,
        session_service=session_service,
    )

    # Create session with sub-questions pre-loaded in state
    initial_state = {}
    for i, q in enumerate(sub_questions):
        initial_state[f"research_question_{i}"] = q

    session2 = session_service.create_session(
        app_name=APP_NAME, user_id="system", state=initial_state
    )

    # Format the research prompt with all sub-questions
    research_prompt = "Research the following questions:\n" + "\n".join(
        f"{i+1}. {q}" for i, q in enumerate(sub_questions)
    )
    content2 = types.Content(
        role="user", parts=[types.Part(text=research_prompt)]
    )

    follow_up_text = ""
    async for event in phase2_runner.run_async(
        user_id="system", session_id=session2.id, new_message=content2
    ):
        if event.is_final_response() and event.content and event.content.parts:
            follow_up_text = event.content.parts[0].text

    # Collect research findings from session state
    session2 = session_service.get_session(
        app_name=APP_NAME, user_id="system", session_id=session2.id
    )
    state = session2.state if session2 else {}

    for i in range(num_questions):
        key = f"research_{i}"
        if key in state:
            result.research_findings[key] = state[key]

    # Parse follow-up questions
    follow_up_questions = []
    try:
        follow_up_raw = state.get("follow_up_questions", follow_up_text)
        if isinstance(follow_up_raw, str):
            follow_up_questions = json.loads(follow_up_raw)
        elif isinstance(follow_up_raw, list):
            follow_up_questions = follow_up_raw
    except (json.JSONDecodeError, TypeError):
        logger.info("No follow-up questions parsed")

    follow_up_questions = follow_up_questions[:3]
    result.follow_up_questions = follow_up_questions

    # ---- Phase 2b: Follow-up research (if any) ----
    num_follow_ups = len(follow_up_questions)
    if num_follow_ups > 0:
        logger.info("Running %d follow-up researchers", num_follow_ups)
        follow_up_researchers = [
            build_researcher(i, model=MODEL, prefix="follow_up")
            for i in range(num_follow_ups)
        ]
        parallel_follow_up = ParallelAgent(
            name="parallel_follow_up",
            sub_agents=follow_up_researchers,
        )

        follow_up_runner = Runner(
            agent=parallel_follow_up,
            app_name=APP_NAME,
            session_service=session_service,
        )

        # Carry forward state from phase 2
        session3 = session_service.create_session(
            app_name=APP_NAME, user_id="system", state=dict(state)
        )

        follow_up_prompt = "Research the following follow-up questions:\n" + "\n".join(
            f"{i+1}. {q}" for i, q in enumerate(follow_up_questions)
        )
        content3 = types.Content(
            role="user", parts=[types.Part(text=follow_up_prompt)]
        )

        async for event in follow_up_runner.run_async(
            user_id="system", session_id=session3.id, new_message=content3
        ):
            pass  # Just run to completion

        session3 = session_service.get_session(
            app_name=APP_NAME, user_id="system", session_id=session3.id
        )
        if session3:
            state.update(session3.state)

        for i in range(num_follow_ups):
            key = f"follow_up_{i}"
            if key in state:
                result.follow_up_findings[key] = state[key]

    # ---- Phase 3: Synthesis ----
    synth_agent = build_synthesizer(num_questions, num_follow_ups, model=MODEL)
    synth_runner = Runner(
        agent=synth_agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

    session4 = session_service.create_session(
        app_name=APP_NAME, user_id="system", state=dict(state)
    )

    synth_prompt = f"Synthesize all research findings for the query: {query}"
    content4 = types.Content(
        role="user", parts=[types.Part(text=synth_prompt)]
    )

    async for event in synth_runner.run_async(
        user_id="system", session_id=session4.id, new_message=content4
    ):
        if event.is_final_response() and event.content and event.content.parts:
            result.final_synthesis = event.content.parts[0].text

    # Fallback: check session state for synthesis
    if not result.final_synthesis:
        session4 = session_service.get_session(
            app_name=APP_NAME, user_id="system", session_id=session4.id
        )
        if session4 and "final_synthesis" in session4.state:
            result.final_synthesis = session4.state["final_synthesis"]

    logger.info("STANDARD pipeline complete. Synthesis length: %d chars", len(result.final_synthesis))
    return result


async def _execute_quick_research(query: str, context: str = "") -> ResearchResult:
    """Single researcher, no unpacking, no follow-ups, simple synthesis."""
    result = ResearchResult(original_query=query)
    session_service = InMemorySessionService()

    researcher = build_researcher(0, model=MODEL, prefix="research")
    runner = Runner(agent=researcher, app_name=APP_NAME, session_service=session_service)

    prompt = f"Research query: {query}"
    if context:
        prompt = f"Context:\n{context}\n\n{prompt}"

    session = session_service.create_session(app_name=APP_NAME, user_id="system")
    content = types.Content(role="user", parts=[types.Part(text=prompt)])

    async for event in runner.run_async(
        user_id="system", session_id=session.id, new_message=content
    ):
        if event.is_final_response() and event.content and event.content.parts:
            result.research_findings["research_0"] = event.content.parts[0].text

    # Quick synthesis
    session = session_service.get_session(
        app_name=APP_NAME, user_id="system", session_id=session.id
    )
    state = session.state if session else {}

    synth_agent = build_synthesizer(1, 0, model=MODEL)
    synth_runner = Runner(agent=synth_agent, app_name=APP_NAME, session_service=session_service)
    synth_session = session_service.create_session(
        app_name=APP_NAME, user_id="system", state=dict(state)
    )
    synth_content = types.Content(
        role="user", parts=[types.Part(text=f"Synthesize research for: {query}")]
    )

    async for event in synth_runner.run_async(
        user_id="system", session_id=synth_session.id, new_message=synth_content
    ):
        if event.is_final_response() and event.content and event.content.parts:
            result.final_synthesis = event.content.parts[0].text

    if not result.final_synthesis:
        synth_session = session_service.get_session(
            app_name=APP_NAME, user_id="system", session_id=synth_session.id
        )
        if synth_session and "final_synthesis" in synth_session.state:
            result.final_synthesis = synth_session.state["final_synthesis"]

    logger.info("QUICK pipeline complete. Synthesis length: %d chars", len(result.final_synthesis))
    return result
