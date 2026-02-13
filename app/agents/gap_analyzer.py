from google.adk.agents import LlmAgent


def build_gap_analyzer(
    study_index: int,
    round_index: int,
    num_researchers: int,
    model: str = "gemini-2.0-flash",
) -> LlmAgent:
    findings_refs = "\n".join(
        f"- {{study_{study_index}_round_{round_index}_researcher_{j}}}"
        for j in range(num_researchers)
    )

    instruction = f"""You are a research gap analyst for an ongoing iterative study.

Review the research findings from this round:
{findings_refs}

Evaluate whether the findings are comprehensive or if important gaps remain.

Output ONLY valid JSON:
- If research is sufficient: {{"escalate": true, "gaps": []}}
- If gaps remain: {{"escalate": false, "gaps": ["specific gap question 1", "specific gap question 2"]}}

Maximum 3 gap questions. Only include truly important gaps that would significantly
improve the research quality."""

    return LlmAgent(
        name=f"gap_analyzer_s{study_index}_r{round_index}",
        model=model,
        instruction=instruction,
        output_key=f"study_{study_index}_gaps_{round_index}",
    )
