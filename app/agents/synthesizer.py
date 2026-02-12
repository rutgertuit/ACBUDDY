from google.adk.agents import LlmAgent


def build_synthesizer(num_research: int, num_follow_ups: int, model: str = "gemini-2.0-flash") -> LlmAgent:
    """Build an LlmAgent that synthesizes all findings into a final document.

    Args:
        num_research: Number of primary research outputs.
        num_follow_ups: Number of follow-up research outputs.
        model: Model to use.

    Returns:
        Configured LlmAgent for synthesis.
    """
    research_refs = "\n".join(
        f"- research_{i}: {{research_{i}}}" for i in range(num_research)
    )
    follow_up_refs = ""
    if num_follow_ups > 0:
        follow_up_refs = "\n\nFollow-up research findings:\n" + "\n".join(
            f"- follow_up_{i}: {{follow_up_{i}}}" for i in range(num_follow_ups)
        )

    instruction = f"""You are a research synthesizer. Combine all research findings into a
single, well-structured document.

Primary research findings:
{research_refs}
{follow_up_refs}

Format your output as a professional research document with these sections:

# Executive Summary
(2-3 paragraph overview of key findings)

# Key Findings
(Detailed findings organized by topic, with bullet points)

# Sources
(List all source URLs referenced in the research)

# Confidence Level
(Rate overall confidence: High/Medium/Low with brief justification)

# Areas for Further Research
(Any remaining gaps or suggested next steps)

Write clearly, cite sources inline, and ensure the document is actionable.
"""
    return LlmAgent(
        name="synthesizer",
        model=model,
        instruction=instruction,
        output_key="final_synthesis",
    )
