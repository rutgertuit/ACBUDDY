import logging
import os
import re
import time

import requests
from google.adk.agents import LlmAgent

from app.services import news_client, grok_client, openai_client

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


def _search_news(query: str) -> str:
    """Search recent news articles for current events, market developments, and media coverage.

    Use this tool when the research question involves:
    - Recent developments or breaking news
    - Market trends and business news
    - Company announcements or product launches
    - Industry events and regulatory changes

    Args:
        query: The news search query string.

    Returns:
        Formatted news results with titles, descriptions, sources, and URLs.
    """
    api_key = os.getenv("NEWSAPI_KEY", "")
    if not api_key:
        return "News search unavailable (no API key configured)"

    articles = news_client.search_news(query=query, api_key=api_key)
    if not articles:
        return f"No recent news found for: {query}"

    parts = []
    for art in articles:
        parts.append(
            f"**{art['title']}** ({art['source']}, {art['published_at'][:10]})\n"
            f"{art['description']}\n"
            f"URL: {art['url']}"
        )
    return "\n\n---\n\n".join(parts)


def _search_grok(query: str) -> str:
    """Search using Grok for real-time web and social media insights.

    Use this tool when the research question involves:
    - Trending topics or viral discussions
    - Social media sentiment and public opinion
    - Real-time market reactions or events
    - X/Twitter discussions and influencer perspectives

    Args:
        query: The search query for real-time web and social data.

    Returns:
        Synthesized findings from Grok including social and web data.
    """
    api_key = os.getenv("GROK_API_KEY", "")
    if not api_key:
        return "Grok search unavailable (no API key configured)"

    result = grok_client.search_with_grok(query=query, api_key=api_key)
    return result or f"No results from Grok for: {query}"


def _deep_reason(question: str, context: str) -> str:
    """Use OpenAI for deep analytical reasoning over complex questions.

    Use this tool when the research question requires:
    - Complex causal analysis or second-order effects
    - Synthesis across multiple conflicting data points
    - Strategic implications or scenario analysis
    - Questions that need careful logical reasoning rather than more data

    Args:
        question: The analytical question to reason about.
        context: Research context or findings gathered so far to reason over.

    Returns:
        Deep analytical reasoning and insights.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return "Deep reasoning unavailable (no API key configured)"

    result = openai_client.deep_reason(
        question=question, context=context, api_key=api_key
    )
    return result or f"No reasoning output for: {question}"


RESEARCHER_INSTRUCTION = """You are a thorough research agent with access to multiple search sources.
Your task is to research the following question using the best combination of tools.

Available tools and when to use them:
1. **web_search** — Default for all queries. Uses Gemini search grounding for broad web results.
2. **search_news** — For current events, recent developments, company/market news, regulatory changes.
3. **search_grok** — For trending topics, social sentiment, X/Twitter discussions, real-time reactions.
4. **deep_reason** — For complex analytical questions. Pass your gathered findings as context and
   ask it to reason about implications, causal chains, or strategic scenarios.
5. **pull_sources** — Fetch and read full content from specific URLs found in search results.

Research strategy:
1. Start with web_search for foundational information.
2. If the question involves recent events or market developments, ALSO use search_news.
3. If the question involves public opinion, trends, or social dynamics, ALSO use search_grok.
4. If you have gathered enough data but need deeper analysis, use deep_reason with your findings as context.
5. Use pull_sources to read full content from the most relevant URLs.
6. Synthesize ALL findings into a clear, detailed summary with citations.

Rules:
- Include specific facts, data points, and source URLs in your response.
- Every claim MUST be backed by a specific source URL. If you cannot verify a claim with a
  concrete source, DO NOT include it. Omit unverified or speculative information entirely.
- Stay strictly within the geographic, temporal, and topical scope of the question. If the
  question is about a specific country or region, only include data and examples from that
  geography. Do not pad findings with data from other regions.
- Use at least 2 different search tools when the topic warrants it.
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
    # Include multi-source tools only when API keys are configured
    tools = [_web_search, _pull_sources]
    if os.getenv("NEWSAPI_KEY", ""):
        tools.append(_search_news)
    if os.getenv("GROK_API_KEY", ""):
        tools.append(_search_grok)
    if os.getenv("OPENAI_API_KEY", ""):
        tools.append(_deep_reason)

    return LlmAgent(
        name=f"researcher_{index}",
        model=model,
        instruction=RESEARCHER_INSTRUCTION,
        tools=tools,
        output_key=f"{prefix}_{index}",
    )
