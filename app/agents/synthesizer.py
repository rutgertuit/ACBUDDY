from google.adk.agents import LlmAgent


def build_synthesizer(num_research: int, num_follow_ups: int, model: str = "gemini-2.5-flash") -> LlmAgent:
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
single, well-structured document with rigorous source quality assessment.

Primary research findings:
{research_refs}
{follow_up_refs}

SOURCE QUALITY RULES:
- Prioritize claims backed by multiple independent sources. If 3+ sources agree, state this.
- Weight authoritative domains higher: government (.gov), academic (.edu, journals),
  major publications (Reuters, Bloomberg, FT, WSJ, NYT) > general web sources.
- When a claim comes from a single source only, explicitly note: "(single source: [domain])".
- Flag potential bias: commercial interest (vendor reports, sponsored content),
  advocacy (lobby groups, NGOs with stated positions). Note: "Source may have commercial interest."
- For each major finding, assign a confidence tag:
  [HIGH CONFIDENCE] — backed by 3+ independent credible sources
  [MEDIUM CONFIDENCE] — backed by 1-2 credible sources
  [LOW CONFIDENCE] — single source, potentially biased, or conflicting data

Format your output as a professional research document with these sections:

# Executive Summary
(2-3 paragraph overview of key findings)

# Key Findings
(Detailed findings organized by topic, with bullet points. Each finding tagged with confidence level.)

# Source Reliability Notes
- High confidence findings: [list findings backed by 3+ sources]
- Medium confidence findings: [list findings from 1-2 credible sources]
- Low confidence / needs verification: [findings from single or potentially biased sources]

# Sources
(List all source URLs referenced, grouped by domain authority tier)

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
