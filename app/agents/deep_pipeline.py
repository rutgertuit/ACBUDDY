import asyncio
import json
import logging

from google.adk.agents import LlmAgent, ParallelAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from app.agents.json_utils import parse_json_response
from app.agents.study_planner import build_study_planner
from app.agents.iterative_researcher import run_iterative_study
from app.agents.deep_research import build_researcher
from app.agents.qa_anticipator import build_qa_anticipator
from app.models.research_result import ResearchResult, StudyResult, QAClusterResult

logger = logging.getLogger(__name__)

MODEL = "gemini-2.0-flash"
APP_NAME = "acbuddy_research"
MAX_CONCURRENT_STUDIES = 3
MAX_CONCURRENT_QA = 3


async def execute_deep_research(
    query: str,
    context: str = "",
    max_studies: int = 6,
    max_rounds_per_study: int = 3,
    max_qa_rounds: int = 2,
) -> ResearchResult:
    """Execute the full DEEP multi-study research pipeline.

    Phase 1: Study planning
    Phase 2: Parallel iterative studies
    Phase 3: Per-study synthesis (done within iterative_researcher)
    Phase 4: Master synthesis
    Phase 5: Anticipatory Q&A research
    """
    result = ResearchResult(original_query=query)
    session_service = InMemorySessionService()

    # ---- Phase 1: Study Planning ----
    logger.info("DEEP Phase 1: Planning studies for query: %s", query[:100])

    planner = build_study_planner(model=MODEL)
    planner_runner = Runner(
        agent=planner,
        app_name=APP_NAME,
        session_service=session_service,
    )

    prompt = f"Research query: {query}"
    if context:
        prompt = f"Conversation context:\n{context}\n\nResearch query: {query}"

    session = session_service.create_session(app_name=APP_NAME, user_id="system")
    content = types.Content(role="user", parts=[types.Part(text=prompt)])

    plan_text = ""
    async for event in planner_runner.run_async(
        user_id="system", session_id=session.id, new_message=content
    ):
        if event.is_final_response() and event.content and event.content.parts:
            plan_text = event.content.parts[0].text
            break

    # Parse study plan (robust: handles markdown fences, preamble)
    studies = parse_json_response(plan_text)
    if not isinstance(studies, list) or not studies:
        logger.warning("Failed to parse study plan, using single study fallback")
        studies = [{"title": query, "angle": "General research", "questions": [query]}]

    studies = studies[:max_studies]
    result.study_plan = studies
    logger.info("DEEP Phase 1 complete: %d studies planned", len(studies))

    # ---- Phase 2 & 3: Parallel Iterative Studies ----
    logger.info("DEEP Phase 2: Running %d iterative studies (max concurrent: %d)", len(studies), MAX_CONCURRENT_STUDIES)

    sem = asyncio.Semaphore(MAX_CONCURRENT_STUDIES)

    async def _study_with_sem(idx, study_dict):
        async with sem:
            try:
                return await run_iterative_study(
                    study_index=idx,
                    study=study_dict,
                    session_service=InMemorySessionService(),  # each study gets own session service
                    model=MODEL,
                    max_rounds=max_rounds_per_study,
                )
            except Exception:
                logger.exception("Study %d '%s' failed", idx, study_dict.get("title", ""))
                return StudyResult(title=study_dict.get("title", f"Study {idx}"), angle=study_dict.get("angle", ""))

    study_tasks = [_study_with_sem(i, s) for i, s in enumerate(studies)]
    study_results = await asyncio.gather(*study_tasks)
    result.studies = list(study_results)

    successful_studies = [s for s in result.studies if s.synthesis]
    logger.info("DEEP Phase 2-3 complete: %d/%d studies produced synthesis", len(successful_studies), len(result.studies))

    if not successful_studies:
        logger.error("No studies produced synthesis, aborting DEEP pipeline")
        return result

    # ---- Phase 4: Master Synthesis ----
    logger.info("DEEP Phase 4: Master synthesis from %d studies", len(successful_studies))

    study_refs = "\n".join(
        f"- Study {i+1} '{s.title}': {{study_{i}_synthesis}}"
        for i, s in enumerate(result.studies) if s.synthesis
    )

    master_instruction = f"""You are an executive research synthesizer. Combine the following
independent study findings into a single executive briefing.

Available study syntheses:
{study_refs}

IMPORTANT RULES:
- Only include insights that are backed by specific, verifiable source URLs from the studies.
  If a study flags findings as unverified or lacking sources, do NOT carry those into this briefing.
- Maintain strict geographic and topical scope. If the research query targets a specific country
  or region, exclude data and examples from other geographies unless explicitly comparative.

Format as:

# Executive Research Briefing: {query}

## Executive Summary
(3-5 paragraph high-level overview synthesizing ALL studies)

## Study Summaries
(Brief summary of each study's key findings)

## Cross-Study Analysis
(Patterns, contradictions, and connections across studies)

## Key Findings & Recommendations
(Top 10 actionable findings with supporting evidence)

## Sources
(Consolidated list of all sources across studies — only URLs that back claims used above)

## Confidence Assessment
(Overall confidence: High/Medium/Low with justification per study area)

Be comprehensive, cite sources, highlight cross-study patterns."""

    master_agent = LlmAgent(
        name="master_synthesizer",
        model=MODEL,
        instruction=master_instruction,
        output_key="master_synthesis",
    )

    master_runner = Runner(
        agent=master_agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

    # Load all study syntheses into state
    master_state = {}
    for i, s in enumerate(result.studies):
        if s.synthesis:
            master_state[f"study_{i}_synthesis"] = s.synthesis

    master_session = session_service.create_session(
        app_name=APP_NAME, user_id="system", state=master_state
    )
    master_content = types.Content(
        role="user",
        parts=[types.Part(text=f"Create an executive briefing for: {query}")],
    )

    async for event in master_runner.run_async(
        user_id="system", session_id=master_session.id, new_message=master_content
    ):
        if event.is_final_response() and event.content and event.content.parts:
            result.master_synthesis = event.content.parts[0].text

    if not result.master_synthesis:
        master_session = session_service.get_session(
            app_name=APP_NAME, user_id="system", session_id=master_session.id
        )
        if master_session and "master_synthesis" in master_session.state:
            result.master_synthesis = master_session.state["master_synthesis"]

    logger.info("DEEP Phase 4 complete: master synthesis %d chars", len(result.master_synthesis))

    # ---- Phase 5: Anticipatory Q&A Research ----
    if not result.master_synthesis:
        logger.warning("No master synthesis, skipping Q&A phase")
        return result

    logger.info("DEEP Phase 5: Anticipatory Q&A research")

    qa_anticipator = build_qa_anticipator(model=MODEL)
    qa_runner = Runner(
        agent=qa_anticipator,
        app_name=APP_NAME,
        session_service=session_service,
    )

    qa_state = {"master_synthesis": result.master_synthesis}
    qa_session = session_service.create_session(
        app_name=APP_NAME, user_id="system", state=qa_state
    )
    qa_content = types.Content(
        role="user",
        parts=[types.Part(text="Generate anticipated follow-up questions and group into clusters.")],
    )

    qa_text = ""
    async for event in qa_runner.run_async(
        user_id="system", session_id=qa_session.id, new_message=qa_content
    ):
        if event.is_final_response() and event.content and event.content.parts:
            qa_text = event.content.parts[0].text

    # Parse Q&A clusters (robust: handles markdown fences)
    clusters = []
    qa_session = session_service.get_session(
        app_name=APP_NAME, user_id="system", session_id=qa_session.id
    )
    raw = qa_session.state.get("qa_clusters_raw", qa_text) if qa_session else qa_text
    qa_data = parse_json_response(raw) if isinstance(raw, str) else raw
    if isinstance(qa_data, dict):
        clusters = qa_data.get("clusters", [])
    elif isinstance(qa_data, list):
        clusters = qa_data
    if not clusters:
        logger.warning("Failed to parse Q&A clusters from: %s", str(raw)[:200])

    if not clusters:
        logger.info("No Q&A clusters generated, skipping Q&A research")
        return result

    clusters = clusters[:5]
    logger.info("DEEP Phase 5: Researching %d Q&A clusters", len(clusters))

    # Research each Q&A cluster in parallel
    qa_sem = asyncio.Semaphore(MAX_CONCURRENT_QA)

    async def _research_qa_cluster(cluster_idx, cluster_data):
        async with qa_sem:
            theme = cluster_data.get("theme", f"Cluster {cluster_idx}")
            questions = cluster_data.get("questions", [])
            if not questions:
                return QAClusterResult(theme=theme)

            try:
                cluster_result = QAClusterResult(theme=theme, questions=questions)
                qa_session_svc = InMemorySessionService()

                # Build researchers for cluster questions
                researchers = [
                    build_researcher(j, model=MODEL, prefix=f"qa_cluster_{cluster_idx}_researcher")
                    for j in range(len(questions))
                ]
                if len(researchers) == 1:
                    agent = researchers[0]
                else:
                    agent = ParallelAgent(
                        name=f"qa_cluster_{cluster_idx}",
                        sub_agents=researchers,
                    )

                runner = Runner(agent=agent, app_name=APP_NAME, session_service=qa_session_svc)
                sess = qa_session_svc.create_session(app_name=APP_NAME, user_id="system")

                research_prompt = f"Research these questions about '{theme}':\n" + "\n".join(
                    f"{j+1}. {q}" for j, q in enumerate(questions)
                )
                msg = types.Content(role="user", parts=[types.Part(text=research_prompt)])

                async for event in runner.run_async(
                    user_id="system", session_id=sess.id, new_message=msg
                ):
                    pass

                # Synthesize cluster findings
                sess = qa_session_svc.get_session(
                    app_name=APP_NAME, user_id="system", session_id=sess.id
                )
                cluster_state = sess.state if sess else {}

                findings_refs = "\n".join(
                    f"- {{qa_cluster_{cluster_idx}_researcher_{j}}}"
                    for j in range(len(questions))
                )

                synth_instruction = f"""Synthesize research findings for the Q&A cluster: "{theme}"

Findings:
{findings_refs}

Format as:
# {theme}

## Answers
(Answer each question with evidence and sources)

## Summary
(2-3 sentence summary of this cluster's key insights)"""

                synth_agent = LlmAgent(
                    name=f"qa_synth_{cluster_idx}",
                    model=MODEL,
                    instruction=synth_instruction,
                    output_key=f"qa_cluster_{cluster_idx}_synthesis",
                )
                synth_runner = Runner(agent=synth_agent, app_name=APP_NAME, session_service=qa_session_svc)
                synth_sess = qa_session_svc.create_session(
                    app_name=APP_NAME, user_id="system", state=dict(cluster_state)
                )
                synth_msg = types.Content(
                    role="user",
                    parts=[types.Part(text=f"Synthesize Q&A findings for: {theme}")],
                )

                async for event in synth_runner.run_async(
                    user_id="system", session_id=synth_sess.id, new_message=synth_msg
                ):
                    if event.is_final_response() and event.content and event.content.parts:
                        cluster_result.findings = event.content.parts[0].text

                if not cluster_result.findings:
                    synth_sess = qa_session_svc.get_session(
                        app_name=APP_NAME, user_id="system", session_id=synth_sess.id
                    )
                    if synth_sess:
                        cluster_result.findings = synth_sess.state.get(
                            f"qa_cluster_{cluster_idx}_synthesis", ""
                        )

                logger.info("Q&A cluster %d '%s' complete: %d chars", cluster_idx, theme, len(cluster_result.findings))
                return cluster_result
            except Exception:
                logger.exception("Q&A cluster %d '%s' failed", cluster_idx, theme)
                return QAClusterResult(theme=theme, questions=questions)

    qa_tasks = [_research_qa_cluster(k, c) for k, c in enumerate(clusters)]
    qa_results = await asyncio.gather(*qa_tasks)
    result.qa_clusters = list(qa_results)

    # Build Q&A summary
    successful_qa = [c for c in result.qa_clusters if c.findings]
    if successful_qa:
        qa_summary_parts = [f"# Anticipated Questions & Answers\n\nBased on research: {query}\n"]
        for c in successful_qa:
            qa_summary_parts.append(f"\n---\n\n{c.findings}")
        result.qa_summary = "\n".join(qa_summary_parts)

    logger.info(
        "DEEP pipeline complete: %d studies, master=%d chars, %d Q&A clusters, summary=%d chars",
        len(successful_studies),
        len(result.master_synthesis),
        len(successful_qa),
        len(result.qa_summary),
    )
    return result
