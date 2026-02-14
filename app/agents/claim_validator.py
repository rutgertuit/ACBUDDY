"""Claim validator — detects contradictions across studies in a research synthesis.

Single LLM call (~5K tokens in, ~500 out). Returns structured contradiction data.
"""

import logging
import os

from app.agents.json_utils import parse_json_response

logger = logging.getLogger(__name__)

CLAIM_VALIDATOR_INSTRUCTION = """You are a claim contradiction detector. Given a research synthesis:
1. Extract top 10-15 factual claims (numbers, stats, rankings, causal statements)
2. Compare claims — identify contradictions
3. For each contradiction: which studies/sources support each side
4. Rate consistency: HIGH/MEDIUM/LOW

Output ONLY valid JSON:
{
  "claims_extracted": 12,
  "contradictions": [
    {
      "claim_a": "Market valued at $50B (2024)",
      "claim_b": "Market valued at $35B",
      "sources_a": ["Study 1"],
      "sources_b": ["Study 3"],
      "severity": "high",
      "likely_resolution": "Different time periods or scope definitions"
    }
  ],
  "consistency_rating": "medium",
  "notes": "Most claims consistent; two major data discrepancies found"
}
No explanation, no markdown fences."""

_DEFAULTS = {
    "claims_extracted": 0,
    "contradictions": [],
    "consistency_rating": "unknown",
    "notes": "",
}


async def validate_claims(synthesis: str, model: str = "") -> dict:
    """Run claim validation on a synthesis. Routes to OpenAI or Gemini.

    Returns parsed dict with contradictions.
    """
    import asyncio
    from app.services.model_router import get_model_for_phase, get_gemini_model

    provider, routed_model = get_model_for_phase("claim_validation")

    if provider == "openai":
        return await _validate_with_openai(synthesis, routed_model)

    if not model:
        model = get_gemini_model()

    try:
        from google import genai
        from google.genai.types import GenerateContentConfig

        api_key = os.getenv("GOOGLE_API_KEY", "")
        client = genai.Client(api_key=api_key)

        # Truncate synthesis to fit context window
        truncated = synthesis[:20000]

        response = client.models.generate_content(
            model=model,
            contents=f"{CLAIM_VALIDATOR_INSTRUCTION}\n\nResearch synthesis to validate:\n\n{truncated}",
            config=GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=1500,
            ),
        )

        text = response.text if response.text else ""
        return _parse_result(text)

    except Exception:
        logger.exception("Claim validation failed, returning defaults")
        return dict(_DEFAULTS)


async def _validate_with_openai(synthesis: str, model: str) -> dict:
    """Run claim validation via OpenAI reasoning model."""
    import asyncio
    from app.services import openai_client as openai_svc

    try:
        truncated = synthesis[:20000]
        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(None, lambda: openai_svc.complete(
            system_prompt=CLAIM_VALIDATOR_INSTRUCTION,
            user_prompt=f"Research synthesis to validate:\n\n{truncated}",
            model=model,
            max_tokens=1500,
            timeout=60,
        ))
        return _parse_result(text)
    except Exception:
        logger.exception("OpenAI claim validation failed, returning defaults")
        return dict(_DEFAULTS)


def _parse_result(text: str) -> dict:
    """Parse claim validation LLM output into structured dict."""
    parsed = parse_json_response(text)
    if isinstance(parsed, dict):
        result = dict(_DEFAULTS)
        result.update(parsed)
        logger.info(
            "Claim validation: %d claims, %d contradictions, consistency=%s",
            result["claims_extracted"],
            len(result["contradictions"]),
            result["consistency_rating"],
        )
        return result

    logger.warning("Claim validation returned non-dict: %s", type(parsed))
    return dict(_DEFAULTS)
