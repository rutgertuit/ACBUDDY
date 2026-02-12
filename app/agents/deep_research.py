import logging
import re

import requests
from google.adk.agents import LlmAgent
from google.adk.tools import google_search

logger = logging.getLogger(__name__)


def _pull_sources(urls: list[str]) -> str:
    """Fetch URLs, strip HTML tags, and return truncated plain text.

    Args:
        urls: List of URLs to fetch content from.

    Returns:
        Combined text content from all successfully fetched URLs.
    """
    results = []
    for url in urls[:5]:  # Limit to 5 URLs
        try:
            resp = requests.get(url, timeout=10, headers={"User-Agent": "ACBUDDY-Research/1.0"})
            resp.raise_for_status()
            # Strip HTML tags
            text = re.sub(r"<[^>]+>", " ", resp.text)
            # Collapse whitespace
            text = re.sub(r"\s+", " ", text).strip()
            # Truncate to 5K chars per source
            results.append(f"[Source: {url}]\n{text[:5000]}\n")
        except Exception as e:
            logger.warning("Failed to fetch %s: %s", url, e)
            results.append(f"[Source: {url}] Error: {e}\n")
    return "\n---\n".join(results)


RESEARCHER_INSTRUCTION = """You are a thorough research agent. Your task is to research the following question
using Google Search and source fetching.

Steps:
1. Use google_search to find relevant results for the question
2. Use pull_sources to fetch and read the most relevant URLs from search results
3. Synthesize your findings into a clear, detailed summary with citations

Include specific facts, data points, and source URLs in your response.
Be thorough but concise. Focus on accuracy and relevance.
"""


def build_researcher(index: int, model: str = "gemini-2.0-flash", prefix: str = "research") -> LlmAgent:
    """Build an LlmAgent with google_search and pull_sources tools.

    Args:
        index: Researcher index (for naming and output key).
        model: Model to use.
        prefix: Output key prefix.

    Returns:
        Configured LlmAgent for deep research.
    """
    return LlmAgent(
        name=f"researcher_{index}",
        model=model,
        instruction=RESEARCHER_INSTRUCTION,
        tools=[google_search, _pull_sources],
        output_key=f"{prefix}_{index}",
    )
