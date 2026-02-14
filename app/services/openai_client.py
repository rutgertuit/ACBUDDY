"""Wrapper for OpenAI API — deep analytical reasoning and o4-mini completions."""

import logging
import os

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


def complete(
    system_prompt: str,
    user_prompt: str,
    model: str = "",
    max_tokens: int = 8000,
    timeout: int = 120,
) -> str:
    """Generic OpenAI completion for synthesis/analysis tasks.

    Supports o4-mini / o3-mini reasoning models with automatic parameter adjustment.
    Falls back gracefully — returns empty string on failure.

    Args:
        system_prompt: System instruction.
        user_prompt: User message (can be long — study findings, synthesis text).
        model: Model override. Defaults to OPENAI_REASONING_MODEL env var or o4-mini.
        max_tokens: Max output tokens.
        timeout: Request timeout in seconds.

    Returns:
        Model response text, or empty string on failure.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return ""

    if not model:
        model = os.getenv("OPENAI_REASONING_MODEL", "o4-mini")

    try:
        # o3-mini and o-series models use different parameters
        is_reasoning = model.startswith("o3") or model.startswith("o4") or model.startswith("o1")

        body = {
            "model": model,
            "messages": [
                {"role": "system" if not is_reasoning else "developer", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        if is_reasoning:
            # Reasoning models use max_completion_tokens, not max_tokens
            body["max_completion_tokens"] = max_tokens
            # o3-mini supports reasoning_effort
            if "mini" in model:
                body["reasoning_effort"] = "medium"
        else:
            body["max_tokens"] = max_tokens

        resp = requests.post(
            BASE_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = data.get("usage", {})
        logger.info(
            "OpenAI %s complete: %d chars (tokens: %d in, %d out)",
            model,
            len(content),
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
        )
        return content

    except Exception as e:
        logger.warning("OpenAI complete failed (model=%s): %s", model, e)
        return ""
