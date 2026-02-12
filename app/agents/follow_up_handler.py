from google.adk.agents import LlmAgent


def build_follow_up_identifier(num_research_outputs: int, model: str = "gemini-2.0-flash") -> LlmAgent:
    """Build an LlmAgent that identifies gaps and generates follow-up questions.

    Args:
        num_research_outputs: Number of research outputs to read from session state.
        model: Model to use.

    Returns:
        Configured LlmAgent for follow-up identification.
    """
    state_refs = "\n".join(
        f"- research_{i}: {{research_{i}}}" for i in range(num_research_outputs)
    )

    instruction = f"""You are a research gap analyst. Review the following research findings
and identify any important gaps, contradictions, or areas that need deeper investigation.

Research findings available in session state:
{state_refs}

Based on your analysis, output a JSON array of 0-3 follow-up questions that would
fill the most critical gaps. If the research is already comprehensive, output an empty array [].

Output ONLY a valid JSON array of strings. No explanation, no markdown.

Example: ["What is the timeline for X regulation?", "How does Y compare to Z in terms of cost?"]
Or if no follow-ups needed: []
"""
    return LlmAgent(
        name="follow_up_identifier",
        model=model,
        instruction=instruction,
        output_key="follow_up_questions",
    )
