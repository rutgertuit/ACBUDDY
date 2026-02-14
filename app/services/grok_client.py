"""Thin wrapper around xAI Grok API for real-time web + social search."""

import logging

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://api.x.ai/v1/chat/completions"


def search_with_grok(
    query: str,
    api_key: str,
    model: str = "grok-3-mini-fast",
) -> str:
    """Use Grok for real-time web and social media search.

    Returns synthesized findings as text, or empty string on failure.
    """
    if not api_key:
        return ""

    try:
        resp = requests.post(
            BASE_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "search_mode": "auto",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a research assistant. Search the web and social media "
                            "for the most current information. Include specific data points, "
                            "dates, and source references. Focus on recent developments, "
                            "trending discussions, and social sentiment."
                        ),
                    },
                    {
                        "role": "user",
                        "content": query,
                    },
                ],
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if content:
            logger.info("Grok returned %d chars for: %s", len(content), query[:80])
        else:
            logger.warning("Grok returned empty response for: %s", query[:80])
        return content

    except Exception as e:
        logger.warning("Grok search failed for '%s': %s", query[:80], e)
        return ""
