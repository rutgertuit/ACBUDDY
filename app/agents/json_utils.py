import json
import re


def parse_json_response(text: str) -> any:
    """Parse JSON from LLM output, stripping markdown fences and preamble."""
    if not text:
        return None

    # Strip markdown code fences
    cleaned = re.sub(r"```(?:json)?\s*", "", text)
    cleaned = re.sub(r"```\s*$", "", cleaned)
    cleaned = cleaned.strip()

    # Try direct parse
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        pass

    # Try to find JSON array or object in the text
    for pattern in [r"\[[\s\S]*\]", r"\{[\s\S]*\}"]:
        match = re.search(pattern, cleaned)
        if match:
            try:
                return json.loads(match.group())
            except (json.JSONDecodeError, TypeError):
                continue

    return None
