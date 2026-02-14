"""Specialized researcher roles for enhanced multi-agent orchestration.

Provides: fact-checker, devil's advocate, and domain expert builders.
"""

import logging

from google.adk.agents import LlmAgent

from app.agents.deep_research import _web_search, _pull_sources

logger = logging.getLogger(__name__)


FACT_CHECKER_INSTRUCTION = """You are a rigorous fact-checker for research findings.

You will receive a research synthesis. Your job is to:
1. Identify the top 5-8 most important factual claims in the synthesis.
2. For EACH claim, search for independent verification using web_search.
3. Rate each claim: VERIFIED (multiple independent sources confirm),
   PARTIALLY VERIFIED (some support but incomplete), or UNVERIFIED (no independent confirmation).
4. For unverified claims, search for the correct information.

Output format:
# Fact-Check Report

## Verified Claims
- [VERIFIED] Claim text — Confirmed by: source1, source2

## Partially Verified Claims
- [PARTIAL] Claim text — Partial support: source. Issue: what's uncertain

## Unverified / Incorrect Claims
- [UNVERIFIED] Claim text — Could not verify. Correct information: ...

## Overall Accuracy Assessment
(Brief assessment of the synthesis's factual reliability)

Be thorough. Every claim check must include source URLs."""


DEVILS_ADVOCATE_INSTRUCTION = """You are a devil's advocate research analyst.

You will receive a research synthesis. Your job is to find the OPPOSING view:
1. Identify the main conclusions and recommendations in the synthesis.
2. For EACH major conclusion, search for counter-evidence and opposing perspectives.
3. Look for: contradicting data, failed case studies, expert disagreements,
   methodological criticisms, alternative interpretations.
4. Be genuinely adversarial — find the strongest possible counter-arguments.

Output format:
# Devil's Advocate Report

## Counter-Evidence Found
For each major conclusion in the synthesis:
- Original claim: [summary]
- Counter-evidence: [what opposes it, with sources]
- Strength of counter-argument: Strong/Moderate/Weak

## Alternative Interpretations
(Different ways the same data could be interpreted)

## Risks & Blind Spots
(What the original research might have missed or underweighted)

## Balanced Assessment
(Which conclusions hold up under scrutiny vs. which are weakened)

Cite ALL sources with URLs."""


DOMAIN_EXPERT_INSTRUCTION_TEMPLATE = """You are a domain expert researcher specializing in {domain}.

Research the assigned question with deep domain expertise:
- Use domain-specific terminology and frameworks
- Prioritize authoritative sources in {domain}
- Consider industry-specific nuances and context
- Reference relevant regulations, standards, and best practices for {domain}
- Identify domain-specific risks and opportunities

Include specific data, statistics, and expert opinions from {domain} sources.
Cite all sources with URLs."""


def build_fact_checker(index: int, model: str = "gemini-2.0-flash", prefix: str = "factcheck") -> LlmAgent:
    """Build a fact-checker agent that verifies claims in a synthesis."""
    return LlmAgent(
        name=f"fact_checker_{index}",
        model=model,
        instruction=FACT_CHECKER_INSTRUCTION,
        tools=[_web_search, _pull_sources],
        output_key=f"{prefix}_{index}",
    )


def build_devils_advocate(index: int, model: str = "gemini-2.0-flash", prefix: str = "devils_advocate") -> LlmAgent:
    """Build a devil's advocate agent that finds counter-evidence."""
    return LlmAgent(
        name=f"devils_advocate_{index}",
        model=model,
        instruction=DEVILS_ADVOCATE_INSTRUCTION,
        tools=[_web_search, _pull_sources],
        output_key=f"{prefix}_{index}",
    )


def build_domain_expert(index: int, domain: str, model: str = "gemini-2.0-flash", prefix: str = "domain_expert") -> LlmAgent:
    """Build a domain-specialized researcher agent."""
    instruction = DOMAIN_EXPERT_INSTRUCTION_TEMPLATE.format(domain=domain)
    return LlmAgent(
        name=f"domain_expert_{domain.replace(' ', '_')}_{index}",
        model=model,
        instruction=instruction,
        tools=[_web_search, _pull_sources],
        output_key=f"{prefix}_{index}",
    )
