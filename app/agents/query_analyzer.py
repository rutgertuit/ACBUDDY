"""Query analyzer — proactively determines specialized agent mix before research begins.

Single LLM call (~200 tokens in, ~100 out). Falls back to safe defaults on failure.
"""

import logging
import os

from app.agents.json_utils import parse_json_response

logger = logging.getLogger(__name__)

QUERY_ANALYZER_INSTRUCTION = """Analyze this research query and determine what specialized
expertise and verification is needed. Output ONLY valid JSON:
{
  "domains": ["finance", "technology"],
  "needs_fact_checking": true,
  "controversial": false,
  "expertise_needed": ["financial_analyst", "domain_expert"],
  "domain_for_expert": "fintech",
  "complexity": "high"
}

Rules:
- "domains": list primary knowledge domains (e.g., finance, technology, healthcare, politics, law, energy, media)
- "needs_fact_checking": true if the topic involves critical factual claims, statistics, or data that must be verified
- "controversial": true if the topic is actively debated, politically sensitive, or has strong opposing viewpoints
- "expertise_needed": list from [financial_analyst, domain_expert, fact_checker, devils_advocate]
- "domain_for_expert": primary domain string if domain_expert is needed, empty string otherwise
- "complexity": low (1-2 simple angles), medium (3-4 angles), high (5+ angles or cross-domain)

No explanation, no markdown fences."""

_DEFAULTS = {
    "domains": [],
    "needs_fact_checking": False,
    "controversial": False,
    "expertise_needed": [],
    "domain_for_expert": "",
    "complexity": "medium",
}


async def analyze_query(query: str, context: str = "", model: str = "gemini-2.5-flash") -> dict:
    """Run query analysis. Returns parsed dict. Falls back to safe defaults on failure."""
    try:
        from google import genai
        from google.genai.types import GenerateContentConfig

        api_key = os.getenv("GOOGLE_API_KEY", "")
        client = genai.Client(api_key=api_key)

        prompt = f"Research query: {query}"
        if context:
            prompt = f"Context:\n{context[:2000]}\n\nResearch query: {query}"

        response = client.models.generate_content(
            model=model,
            contents=f"{QUERY_ANALYZER_INSTRUCTION}\n\n{prompt}",
            config=GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=300,
            ),
        )

        text = response.text if response.text else ""
        parsed = parse_json_response(text)
        if isinstance(parsed, dict):
            # Merge with defaults to ensure all keys present
            result = dict(_DEFAULTS)
            result.update(parsed)
            logger.info("Query analysis: domains=%s, complexity=%s, fact_check=%s, controversial=%s",
                        result["domains"], result["complexity"],
                        result["needs_fact_checking"], result["controversial"])
            return result

        logger.warning("Query analysis returned non-dict: %s", type(parsed))
        return dict(_DEFAULTS)

    except Exception:
        logger.exception("Query analysis failed, using defaults")
        return dict(_DEFAULTS)
