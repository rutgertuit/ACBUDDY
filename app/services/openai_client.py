"""Wrapper for OpenAI API — deep analytical reasoning."""

import logging

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://api.openai.com/v1/chat/completions"


def deep_reason(
    question: str,
    context: str,
    api_key: str,
    model: str = "gpt-4o",
) -> str:
    """Use OpenAI for complex analytical reasoning.

    Args:
        question: The analytical question to reason about.
        context: Research context/findings to reason over.
        api_key: OpenAI API key.
        model: Model to use (default gpt-4o).

    Returns analysis text, or empty string on failure.
    """
    if not api_key:
        return ""

    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert analyst. Provide deep, structured analytical "
                    "reasoning based on the provided research context. Focus on "
                    "implications, causal relationships, second-order effects, and "
                    "non-obvious insights. Be specific and evidence-based."
                ),
            },
        ]

        if context:
            messages.append({
                "role": "user",
                "content": f"Research context:\n{context[:8000]}",
            })

        messages.append({
            "role": "user",
            "content": question,
        })

        resp = requests.post(
            BASE_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": messages,
                "max_tokens": 4000,
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()

        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if content:
            logger.info("OpenAI reasoning returned %d chars for: %s", len(content), question[:80])
        else:
            logger.warning("OpenAI returned empty response for: %s", question[:80])
        return content

    except Exception as e:
        logger.warning("OpenAI reasoning failed for '%s': %s", question[:80], e)
        return ""
