import logging
import os
import re
import time

import requests
from google.adk.agents import LlmAgent

logger = logging.getLogger(__name__)

MAX_SEARCH_RETRIES = 3
SEARCH_INITIAL_BACKOFF = 2


def _web_search(query: str) -> str:
    """Search the web using Gemini's built-in search grounding and return results.

    Args:
        query: The search query string.

    Returns:
        Search results as formatted text with sources.
    """
    from google import genai
    from google.genai.types import Tool, GenerateContentConfig

    api_key = os.getenv("GOOGLE_API_KEY", "")
    client = genai.Client(api_key=api_key)

    backoff = SEARCH_INITIAL_BACKOFF
    last_error = None

    for attempt in range(MAX_SEARCH_RETRIES):
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=f"Search and summarize information about: {query}",
                config=GenerateContentConfig(
                    tools=[Tool(google_search={})],
                ),
            )

            result_parts = []
            if response.text:
                result_parts.append(response.text)

            # Extract grounding metadata if available
            candidate = response.candidates[0] if response.candidates else None
            if candidate and candidate.grounding_metadata:
                chunks = candidate.grounding_metadata.grounding_chunks or []
                for chunk in chunks:
                    if chunk.web:
                        result_parts.append(f"[Source: {chunk.web.title} - {chunk.web.uri}]")

            return "\n".join(result_parts) if result_parts else f"No results found for: {query}"
        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            is_retryable = any(kw in error_str for kw in [
                "429", "500", "503", "connect", "timeout", "read", "reset",
                "resource_exhausted", "rate", "unavailable",
            ])
            if is_retryable and attempt < MAX_SEARCH_RETRIES - 1:
                logger.warning(
                    "Web search attempt %d failed (retryable), retrying in %ds: %s",
                    attempt + 1, backoff, e,
                )
                time.sleep(backoff)
                backoff *= 2
            else:
                break

    logger.warning("Web search failed for query '%s' after %d attempts: %s", query, MAX_SEARCH_RETRIES, last_error)
    return f"Search failed after retries: {last_error}"


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
            resp = requests.get(url, timeout=15, headers={"User-Agent": "ACBUDDY-Research/1.0"})
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
using web search and source fetching.

Steps:
1. Use web_search to find relevant results for the question
2. Use pull_sources to fetch and read the most relevant URLs from search results
3. Synthesize your findings into a clear, detailed summary with citations

Rules:
- Include specific facts, data points, and source URLs in your response.
- Every claim MUST be backed by a specific source URL. If you cannot verify a claim with a
  concrete source, DO NOT include it. Omit unverified or speculative information entirely.
- Stay strictly within the geographic, temporal, and topical scope of the question. If the
  question is about a specific country or region, only include data and examples from that
  geography. Do not pad findings with data from other regions.
- Be thorough but concise. Focus on accuracy and relevance.
"""


def build_researcher(index: int, model: str = "gemini-2.0-flash", prefix: str = "research") -> LlmAgent:
    """Build an LlmAgent with web_search and pull_sources tools.

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
        tools=[_web_search, _pull_sources],
        output_key=f"{prefix}_{index}",
    )
