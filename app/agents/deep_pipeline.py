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
from app.agents.synthesis_evaluator import evaluate_synthesis
from app.agents.strategic_analyst import run_strategic_analysis
from app.agents.query_analyzer import analyze_query
from app.agents.claim_validator import validate_claims
from app.models.research_result import ResearchResult, StudyResult, QAClusterResult
from app.services.model_router import (
    get_model_for_phase, get_gemini_model, should_use_deep_research, has_openai,
)
from app.services import openai_client as openai_svc

logger = logging.getLogger(__name__)

MODEL = get_gemini_model()
APP_NAME = "acbuddy_research"
MAX_CONCURRENT_STUDIES = 3
MAX_CONCURRENT_QA = 3


async def _run_deep_research_study(idx: int, study: dict) -> StudyResult:
    """Run a study using Gemini Deep Research (autonomous agent).

    Returns a StudyResult with synthesis populated from the deep research report.
    Falls back with empty synthesis on failure (caller will retry with iterative).
    """
    title = study.get("title", f"Study {idx}")
    angle = study.get("angle", "")
    questions = study.get("questions", [])

    # Build a comprehensive research prompt
    prompt_parts = [f"Research study: {title}"]
    if angle:
        prompt_parts.append(f"Research angle: {angle}")
    if questions:
        prompt_parts.append("Key questions to answer:")
        for q in questions:
            prompt_parts.append(f"- {q}")
    prompt_parts.append(
        "\nProvide a comprehensive, well-cited research report with specific data, "
        "statistics, and source URLs. Include confidence levels for key findings."
    )

    try:
        from app.services.gemini_deep_research import run_deep_research
        report = await run_deep_research(
            query="\n".join(prompt_parts),
            context=f"This is study {idx + 1} of a multi-study research pipeline.",
        )
        if report:
            logger.info("Deep Research study %d '%s': %d chars", idx, title, len(report))
            return StudyResult(
                title=title,
                angle=angle,
                questions=questions,
                rounds=[{"deep_research": report}],
                synthesis=report,
            )
    except Exception:
        logger.exception("Deep Research study %d failed", idx)

    return StudyResult(title=title, angle=angle, questions=questions)


async def execute_deep_research(
    query: str,
    context: str = "",
    max_studies: int = 0,
    max_rounds_per_study: int = 3,
    max_qa_rounds: int = 2,
    on_progress=None,
    business_context: dict | None = None,
) -> ResearchResult:
    """Execute the full DEEP multi-study research pipeline.

    Phase 1: Study planning
    Phase 2: Parallel iterative studies
    Phase 3: Per-study synthesis (done within iterative_researcher)
    Phase 4: Master synthesis
    Phase 4b: Synthesis evaluation & refinement
    Phase 4c: Strategic analysis
    Phase 5: Anticipatory Q&A research

    Args:
        on_progress: Optional callback(phase, **kwargs) for reporting progress.
    """
    def _progress(phase, **kwargs):
        if on_progress:
            try:
                on_progress(phase, **kwargs)
            except Exception:
                pass

    result = ResearchResult(original_query=query)
    session_service = InMemorySessionService()

    # ---- Phase 0: Query Analysis ----
    _progress("Analyzing query", step="analysis")
    logger.info("DEEP Phase 0: Analyzing query: %s", query[:100])
    query_analysis = await analyze_query(query, context)
    result.query_analysis = query_analysis
    logger.info("DEEP Phase 0 complete: domains=%s, complexity=%s",
                query_analysis.get("domains"), query_analysis.get("complexity"))

    # ---- Phase 1: Study Planning ----
    _progress("planning", step="planning")
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

    # Inject query analysis into planning prompt
    qa_domains = query_analysis.get("domains", [])
    qa_complexity = query_analysis.get("complexity", "medium")
    if qa_domains:
        prompt += f"\n\nQuery analysis suggests domains: {', '.join(qa_domains)}, complexity: {qa_complexity}. Plan studies accordingly."
    if context and "Relevant findings from past research" in context:
        prompt += "\nPrior research findings are available (see context). Focus studies on areas NOT already covered, or on updating stale findings."

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

    if max_studies > 0:
        studies = studies[:max_studies]
    result.study_plan = studies
    logger.info("DEEP Phase 1 complete: %d studies planned", len(studies))
    _progress(
        f"Planned {len(studies)} studies",
        step="studies",
        study_plan=[{"title": s.get("title", ""), "angle": s.get("angle", "")} for s in studies],
        study_progress=[{"title": s.get("title", ""), "status": "pending", "rounds": 0} for s in studies],
    )

    # ---- Phase 2 & 3: Parallel Iterative Studies ----
    logger.info("DEEP Phase 2: Running %d iterative studies (max concurrent: %d)", len(studies), MAX_CONCURRENT_STUDIES)

    sem = asyncio.Semaphore(MAX_CONCURRENT_STUDIES)

    async def _study_with_sem(idx, study_dict):
        async with sem:
            title = study_dict.get("title", f"Study {idx}")
            role = study_dict.get("recommended_role", "general")
            domain = study_dict.get("domain", "")
            # Use query analysis to default to domain expert if no role specified
            if role == "general" and query_analysis.get("domain_for_expert"):
                role = "domain_expert"
                domain = domain or query_analysis["domain_for_expert"]
            _progress(f"Researching: {title}", step=f"study_{idx}",
                      study_idx=idx, study_status="running")
            try:
                # Route complex studies to Gemini Deep Research
                use_deep = should_use_deep_research(study_dict, query_analysis)
                if use_deep:
                    sr = await _run_deep_research_study(idx, study_dict)
                    if sr.synthesis:
                        _progress(f"Completed: {title}", step=f"study_{idx}",
                                  study_idx=idx, study_status="done")
                        return sr
                    # Fall through to iterative if deep research fails
                    logger.warning("Deep Research failed for study %d, falling back to iterative", idx)

                # Select researcher builder based on role
                researcher_builder = None
                if role == "domain_expert" and domain:
                    try:
                        from app.agents.specialized_roles import build_domain_expert
                        researcher_builder = lambda i, m, p: build_domain_expert(i, domain, m, p)
                    except Exception:
                        pass  # fall back to default

                sr = await run_iterative_study(
                    study_index=idx,
                    study=study_dict,
                    session_service=InMemorySessionService(),
                    model=MODEL,
                    max_rounds=max_rounds_per_study,
                    researcher_builder=researcher_builder,
                )
                _progress(f"Completed: {title}", step=f"study_{idx}",
                          study_idx=idx, study_status="done")
                return sr
            except Exception:
                logger.exception("Study %d '%s' failed", idx, study_dict.get("title", ""))
                _progress(f"Failed: {title}", step=f"study_{idx}",
                          study_idx=idx, study_status="failed")
                return StudyResult(title=title, angle=study_dict.get("angle", ""))

    study_tasks = [_study_with_sem(i, s) for i, s in enumerate(studies)]
    study_results = await asyncio.gather(*study_tasks)
    result.studies = list(study_results)

    successful_studies = [s for s in result.studies if s.synthesis]
    logger.info("DEEP Phase 2-3 complete: %d/%d studies produced synthesis", len(successful_studies), len(result.studies))

    if not successful_studies:
        logger.error("No studies produced synthesis, aborting DEEP pipeline")
        return result

    # ---- Phase 4: Master Synthesis ----
    _progress(f"Synthesizing {len(successful_studies)} studies", step="synthesis")
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

SOURCE QUALITY RULES:
- Prioritize claims corroborated across multiple independent studies. Cross-study agreement
  significantly increases confidence.
- Weight authoritative domains higher: government (.gov), academic (.edu, peer-reviewed),
  major publications (Reuters, Bloomberg, FT, WSJ, NYT) > industry reports > general web.
- When a claim appears in only one study with a single source, note: "(single source)".
- Flag potential bias: vendor/consulting reports, advocacy organizations, sponsored research.
  Note: "Source may have commercial/advocacy interest."
- For each key finding, assign a confidence tag:
  [HIGH CONFIDENCE] — corroborated across studies OR backed by 3+ authoritative sources
  [MEDIUM CONFIDENCE] — single study with 1-2 credible sources
  [LOW CONFIDENCE] — single source, potentially biased, or studies conflict

Format as:

# Executive Research Briefing: {query}

## Executive Summary
(3-5 paragraph high-level overview synthesizing ALL studies)

## Study Summaries
(Brief summary of each study's key findings)

## Cross-Study Analysis
(Patterns, contradictions, and connections across studies. Note where studies
agree [HIGH CONFIDENCE] vs. where only one study covers a topic [MEDIUM/LOW].)

## Key Findings & Recommendations
(Top 10 actionable findings with supporting evidence. Each tagged with confidence level.)

## Source Reliability Notes
- High confidence findings: [list findings corroborated across 2+ studies or 3+ sources]
- Medium confidence findings: [single study, 1-2 credible sources]
- Low confidence / needs verification: [single source, biased, or conflicting]
- Potential bias flags: [any findings from vendor/advocacy sources]

## Sources
(Consolidated list of all sources — grouped by authority tier)

## Confidence Assessment
(Overall confidence: High/Medium/Low with justification per study area)

Be comprehensive, cite sources, highlight cross-study patterns."""

    # Load all study syntheses into state
    master_state = {}
    for i, s in enumerate(result.studies):
        if s.synthesis:
            master_state[f"study_{i}_synthesis"] = s.synthesis

    # Route master synthesis: OpenAI for deep reasoning, Gemini for fallback
    provider, synth_model = get_model_for_phase("master_synthesis")
    logger.info("Master synthesis routing: provider=%s, model=%s", provider, synth_model)

    if provider == "openai":
        # Resolve template variables for direct OpenAI API call
        resolved_instruction = master_instruction
        for key, value in master_state.items():
            resolved_instruction = resolved_instruction.replace(f"{{{key}}}", value)

        loop = asyncio.get_running_loop()
        result.master_synthesis = await loop.run_in_executor(None, lambda: openai_svc.complete(
            system_prompt=resolved_instruction,
            user_prompt=f"Create an executive briefing for: {query}",
            model=synth_model,
            max_tokens=12000,
            timeout=180,
        ))
    else:
        master_agent = LlmAgent(
            name="master_synthesizer",
            model=synth_model,
            instruction=master_instruction,
            output_key="master_synthesis",
        )
        master_runner = Runner(
            agent=master_agent,
            app_name=APP_NAME,
            session_service=session_service,
        )
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

    # ---- Phase 4a: Claim Validation (contradiction detection) ----
    claim_validation = {}
    if result.master_synthesis:
        _progress("Validating claims", step="claim_validation")
        logger.info("DEEP Phase 4a: Cross-source claim validation")
        try:
            claim_validation = await validate_claims(result.master_synthesis)
            result.claim_validation = claim_validation
            logger.info(
                "Claim validation: %d contradictions, consistency=%s",
                len(claim_validation.get("contradictions", [])),
                claim_validation.get("consistency_rating"),
            )
        except Exception:
            logger.exception("Claim validation failed (non-fatal)")

    # ---- Phase 4b: Synthesis Evaluation & Refinement ----
    if result.master_synthesis:
        max_refinement_rounds = 2
        for refine_round in range(max_refinement_rounds):
            _progress(f"Evaluating quality (round {refine_round + 1})", step="evaluation")
            logger.info("DEEP Phase 4b: Evaluating synthesis (round %d)", refine_round + 1)

            evaluation = await evaluate_synthesis(
                query=query,
                master_synthesis=result.master_synthesis,
                model=MODEL,
            )
            result.synthesis_score = evaluation.get("overall_score", 0.0)
            result.synthesis_scores = evaluation.get("scores", {})
            result.refinement_rounds = refine_round + 1

            if not evaluation.get("refinement_needed", False):
                logger.info(
                    "Synthesis scored %.1f — no refinement needed",
                    result.synthesis_score,
                )
                break

            # Extract high/medium priority gap questions
            gaps = evaluation.get("gaps", [])
            gap_questions = [
                g["research_question"]
                for g in gaps
                if g.get("research_question") and g.get("priority") in ("high", "medium")
            ]
            if not gap_questions:
                logger.info("No actionable gap questions, skipping refinement")
                break

            gap_questions = gap_questions[:6]
            logger.info(
                "Synthesis scored %.1f — running %d additional studies for gaps: %s",
                result.synthesis_score,
                len(gap_questions),
                [q[:60] for q in gap_questions],
            )

            # Run full iterative studies for gaps (not just lightweight researchers)
            gap_study_offset = len(result.studies)

            async def _gap_study_with_sem(idx, question):
                async with sem:
                    gap_study = {
                        "title": f"Gap Study: {question[:80]}",
                        "angle": f"Addressing gap identified in synthesis evaluation",
                        "questions": [question],
                    }
                    _progress(f"Gap study: {question[:50]}", step=f"gap_study_{idx}",
                              study_idx=gap_study_offset + idx, study_status="running")
                    try:
                        sr = await run_iterative_study(
                            study_index=gap_study_offset + idx,
                            study=gap_study,
                            session_service=InMemorySessionService(),
                            model=MODEL,
                            max_rounds=2,
                        )
                        _progress(f"Completed gap: {question[:50]}", step=f"gap_study_{idx}",
                                  study_idx=gap_study_offset + idx, study_status="done")
                        return sr
                    except Exception:
                        logger.exception("Gap study %d failed: %s", idx, question[:60])
                        _progress(f"Failed gap: {question[:50]}", step=f"gap_study_{idx}",
                                  study_idx=gap_study_offset + idx, study_status="failed")
                        return StudyResult(title=gap_study["title"], angle=gap_study["angle"])

            gap_tasks = [_gap_study_with_sem(i, q) for i, q in enumerate(gap_questions)]
            gap_study_results = await asyncio.gather(*gap_tasks)
            gap_study_results = [s for s in gap_study_results if s.synthesis]

            if not gap_study_results:
                logger.warning("All gap studies failed, keeping original synthesis")
                break

            # Add gap studies to result and update master state
            result.studies.extend(gap_study_results)
            for i, gs in enumerate(gap_study_results):
                gap_state_idx = gap_study_offset + i
                master_state[f"study_{gap_state_idx}_synthesis"] = gs.synthesis

            # Update study_refs for refinement
            study_refs = "\n".join(
                f"- Study {i+1} '{s.title}': {{study_{i}_synthesis}}"
                for i, s in enumerate(result.studies) if s.synthesis
            )

            gap_findings = [gs.synthesis for gs in gap_study_results]
            successful_studies = [s for s in result.studies if s.synthesis]
            logger.info(
                "Gap round %d: %d new studies completed (%d total studies now)",
                refine_round + 1, len(gap_study_results), len(successful_studies),
            )

            # Regenerate master synthesis with all studies (original + gap)
            _progress(f"Refining synthesis (round {refine_round + 1})", step="refinement")
            logger.info(
                "Refining synthesis with %d total studies (%d new gap studies)",
                len([s for s in result.studies if s.synthesis]),
                len(gap_findings),
            )

            weak_claims_note = ""
            weak = evaluation.get("weak_claims", [])
            if weak:
                weak_claims_note = (
                    "\n\nThe following claims in the previous synthesis were flagged as weak "
                    "and need stronger evidence:\n"
                    + "\n".join(f"- {w}" for w in weak[:5])
                )

            missing_note = ""
            missing = evaluation.get("missing_perspectives", [])
            if missing:
                missing_note = (
                    "\n\nThe following perspectives were missing and should be addressed "
                    "if the new gap studies provide relevant information:\n"
                    + "\n".join(f"- {m}" for m in missing[:5])
                )

            # Inject claim contradictions if found
            contradiction_note = ""
            contradictions = claim_validation.get("contradictions", [])
            if contradictions:
                high_sev = [c for c in contradictions if c.get("severity") == "high"]
                if high_sev:
                    contradiction_note = (
                        "\n\nThe following contradictions were found and must be resolved in this refined draft:\n"
                        + "\n".join(
                            f"- CONFLICT: '{c.get('claim_a', '')}' vs '{c.get('claim_b', '')}' "
                            f"(resolution hint: {c.get('likely_resolution', 'investigate')})"
                            for c in high_sev[:5]
                        )
                    )

            refine_instruction = f"""You are an executive research synthesizer producing a REFINED
draft of a research briefing. You now have {len([s for s in result.studies if s.synthesis])} studies total
(including new gap studies that address previously identified weaknesses).

All study syntheses:
{study_refs}
{weak_claims_note}
{missing_note}
{contradiction_note}

Produce an improved executive briefing that:
- Incorporates findings from ALL studies including the new gap studies
- Strengthens or removes claims that lacked evidence
- Includes any newly discovered perspectives
- Resolves any identified contradictions by investigating source authority and recency
- Maintains all well-supported content from the original synthesis

SOURCE QUALITY RULES (apply rigorously in this refined draft):
- Prioritize claims corroborated across multiple studies.
- Weight authoritative sources higher: government, academic, major publications > general web.
- Remove or downgrade claims that remain single-sourced.
- Flag any remaining potential bias from vendor/advocacy sources.
- Tag each key finding with confidence level:
  [HIGH CONFIDENCE] — corroborated across studies or 3+ authoritative sources
  [MEDIUM CONFIDENCE] — 1-2 credible sources
  [LOW CONFIDENCE] — single source, biased, or conflicting

Format as:

# Executive Research Briefing: {query}

## Executive Summary
(3-5 paragraph high-level overview synthesizing ALL studies)

## Study Summaries
(Brief summary of each study's key findings)

## Cross-Study Analysis
(Patterns, contradictions, and connections across all studies.
Note confidence levels for cross-study vs. single-study findings.)

## Key Findings & Recommendations
(Top 10 actionable findings with supporting evidence. Each tagged with confidence level.)

## Source Reliability Notes
- High confidence findings: [corroborated across 2+ studies or 3+ sources]
- Medium confidence findings: [single study, 1-2 credible sources]
- Low confidence / needs verification: [single source, biased, or conflicting]

## Sources
(Consolidated list of ALL sources — grouped by authority tier)

## Confidence Assessment
(Overall confidence: High/Medium/Low with justification per area)

Be comprehensive. Mark any remaining areas of uncertainty explicitly."""

            refine_state = dict(master_state)

            # Route refinement same as master synthesis
            refined_text = ""
            if provider == "openai":
                resolved_refine = refine_instruction
                for key, value in refine_state.items():
                    resolved_refine = resolved_refine.replace(f"{{{key}}}", value)

                loop = asyncio.get_running_loop()
                refined_text = await loop.run_in_executor(
                    None,
                    lambda ri=resolved_refine: openai_svc.complete(
                        system_prompt=ri,
                        user_prompt=f"Create a refined executive briefing for: {query}",
                        model=synth_model,
                        max_tokens=12000,
                        timeout=180,
                    ),
                )
            else:
                refine_agent = LlmAgent(
                    name="master_synthesizer_refine",
                    model=synth_model,
                    instruction=refine_instruction,
                    output_key="master_synthesis_refined",
                )
                refine_runner = Runner(
                    agent=refine_agent,
                    app_name=APP_NAME,
                    session_service=session_service,
                )
                refine_session = session_service.create_session(
                    app_name=APP_NAME, user_id="system", state=refine_state
                )
                refine_content = types.Content(
                    role="user",
                    parts=[types.Part(
                        text=f"Create a refined executive briefing for: {query}"
                    )],
                )
                async for event in refine_runner.run_async(
                    user_id="system",
                    session_id=refine_session.id,
                    new_message=refine_content,
                ):
                    if event.is_final_response() and event.content and event.content.parts:
                        refined_text = event.content.parts[0].text

                if not refined_text:
                    refine_session = session_service.get_session(
                        app_name=APP_NAME,
                        user_id="system",
                        session_id=refine_session.id,
                    )
                    if refine_session:
                        refined_text = refine_session.state.get(
                            "master_synthesis_refined", ""
                        )

            if refined_text:
                result.master_synthesis = refined_text
                logger.info(
                    "Refinement round %d complete: %d chars (was %d)",
                    refine_round + 1,
                    len(refined_text),
                    len(result.master_synthesis),
                )
            else:
                logger.warning("Refinement produced empty result, keeping previous synthesis")
                break

    # ---- Phase 4d: Enhanced Verification ----
    # Trigger when: score < 7.5 OR query analysis says fact-checking needed
    needs_fact_check = query_analysis.get("needs_fact_checking", False)
    is_controversial = query_analysis.get("controversial", False)
    low_score = result.synthesis_score > 0 and result.synthesis_score < 7.5

    if result.master_synthesis and (low_score or needs_fact_check):
        _progress("Running enhanced verification", step="verification")
        logger.info(
            "DEEP Phase 4d: Enhanced verification (score=%.1f, fact_check=%s, controversial=%s)",
            result.synthesis_score, needs_fact_check, is_controversial,
        )

        try:
            from app.agents.specialized_roles import build_fact_checker, build_devils_advocate

            verify_svc = InMemorySessionService()

            # Run fact-checker
            fact_checker = build_fact_checker(0, model=MODEL)
            fc_runner = Runner(agent=fact_checker, app_name=APP_NAME, session_service=verify_svc)
            fc_sess = verify_svc.create_session(app_name=APP_NAME, user_id="system")
            fc_msg = types.Content(
                role="user",
                parts=[types.Part(text=f"Fact-check this research synthesis:\n\n{result.master_synthesis[:15000]}")],
            )
            fc_text = ""
            async for event in fc_runner.run_async(
                user_id="system", session_id=fc_sess.id, new_message=fc_msg
            ):
                if event.is_final_response() and event.content and event.content.parts:
                    fc_text = event.content.parts[0].text

            # Run devil's advocate — always when controversial, or when score is low
            da_text = ""
            if is_controversial or low_score:
                da_agent = build_devils_advocate(0, model=MODEL)
                da_runner = Runner(agent=da_agent, app_name=APP_NAME, session_service=verify_svc)
                da_sess = verify_svc.create_session(app_name=APP_NAME, user_id="system")
                da_msg = types.Content(
                    role="user",
                    parts=[types.Part(text=f"Challenge this research synthesis:\n\n{result.master_synthesis[:15000]}")],
                )
                async for event in da_runner.run_async(
                    user_id="system", session_id=da_sess.id, new_message=da_msg
                ):
                    if event.is_final_response() and event.content and event.content.parts:
                        da_text = event.content.parts[0].text

            # Incorporate verification findings into synthesis
            if fc_text or da_text:
                _progress("Incorporating verification findings", step="refinement")
                verify_refs = ""
                verify_state = dict(master_state)
                if fc_text:
                    verify_refs += "\n- Fact-check findings: {fact_check_findings}"
                    verify_state["fact_check_findings"] = fc_text
                if da_text:
                    verify_refs += "\n- Devil's advocate findings: {devils_advocate_findings}"
                    verify_state["devils_advocate_findings"] = da_text

                verify_instruction = f"""You are refining a research briefing using verification feedback.

Original study syntheses:
{study_refs}

Verification findings:
{verify_refs}

Produce an improved briefing that:
- Removes or weakens claims flagged as unverified by the fact-checker
- Acknowledges strong counter-evidence from the devil's advocate
- Strengthens well-verified claims
- Maintains the same format as the original briefing

Format as: # Executive Research Briefing: {query}
(same sections as before)"""

                verify_agent = LlmAgent(
                    name="verified_synthesizer",
                    model=MODEL,
                    instruction=verify_instruction,
                    output_key="verified_synthesis",
                )
                v_runner = Runner(agent=verify_agent, app_name=APP_NAME, session_service=session_service)
                v_sess = session_service.create_session(
                    app_name=APP_NAME, user_id="system", state=verify_state
                )
                v_msg = types.Content(
                    role="user",
                    parts=[types.Part(text=f"Create a verified briefing for: {query}")],
                )
                v_text = ""
                async for event in v_runner.run_async(
                    user_id="system", session_id=v_sess.id, new_message=v_msg
                ):
                    if event.is_final_response() and event.content and event.content.parts:
                        v_text = event.content.parts[0].text

                if v_text:
                    result.master_synthesis = v_text
                    logger.info("Enhanced verification complete: %d chars", len(v_text))
        except Exception:
            logger.exception("Enhanced verification failed, continuing with existing synthesis")

    # ---- Phase 4c: Strategic Analysis ----
    if result.master_synthesis:
        _progress("Applying strategic frameworks", step="strategic_analysis")
        logger.info("DEEP Phase 4c: Strategic analysis")
        try:
            result.strategic_analysis = await run_strategic_analysis(
                query=query,
                master_synthesis=result.master_synthesis,
                model=MODEL,
            )
            logger.info(
                "DEEP Phase 4c complete: strategic analysis %d chars",
                len(result.strategic_analysis),
            )
        except Exception:
            logger.exception("Strategic analysis failed, continuing without it")

    # ---- Phase 4e: Knowledge Graph Context Enrichment ----
    if result.master_synthesis and context and "Known entity relationships:" in context:
        try:
            # Extract graph context that was injected by root_agent
            kg_start = context.find("Known entity relationships:")
            kg_end = context.find("\n\n", kg_start + 1)
            kg_section = context[kg_start:kg_end] if kg_end > kg_start else context[kg_start:]
            if kg_section.strip():
                result.master_synthesis += (
                    f"\n\n## Knowledge Graph Insights\n\n"
                    f"The following entity relationships were known prior to this research "
                    f"and cross-referenced with synthesis findings:\n\n{kg_section}"
                )
                logger.info("Appended KG insights section to master synthesis")
        except Exception:
            logger.warning("Failed to append KG insights (non-fatal)")

    # ---- Phase 5: Anticipatory Q&A Research ----
    if not result.master_synthesis:
        logger.warning("No master synthesis, skipping Q&A phase")
        return result

    _progress("Generating anticipated Q&A", step="qa")
    logger.info("DEEP Phase 5: Anticipatory Q&A research")

    qa_anticipator = build_qa_anticipator(model=MODEL, business_context=business_context)
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
