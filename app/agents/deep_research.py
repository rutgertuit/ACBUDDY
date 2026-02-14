import logging
import os
import re
import time

import requests
from google.adk.agents import LlmAgent

from app.services import news_client, grok_client, openai_client
from app.services.research_stats import increment

logger = logging.getLogger(__name__)

MAX_SEARCH_RETRIES = 3
SEARCH_INITIAL_BACKOFF = 2


def web_search(query: str) -> str:
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

    increment("web_searches")

    for attempt in range(MAX_SEARCH_RETRIES):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
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
            source_count = 0
            if candidate and candidate.grounding_metadata:
                chunks = candidate.grounding_metadata.grounding_chunks or []
                for chunk in chunks:
                    if chunk.web:
                        # Ensure URI and title are str (proto fields can sometimes be bytes)
                        uri = chunk.web.uri if isinstance(chunk.web.uri, str) else str(chunk.web.uri or "", "utf-8", errors="replace")
                        title = chunk.web.title if isinstance(chunk.web.title, str) else str(chunk.web.title or "", "utf-8", errors="replace")
                        # Score source authority
                        from app.services.source_scorer import score_url, format_authority_tag
                        url_score = score_url(uri)
                        tag = format_authority_tag(url_score)
                        result_parts.append(f"[Source: {title} - {uri}] {tag}")
                        source_count += 1
            if source_count:
                increment("pages_read", source_count)

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


def pull_sources(urls: list[str]) -> str:
    """Fetch URLs, strip HTML tags, and return truncated plain text.

    Args:
        urls: List of URLs to fetch content from.

    Returns:
        Combined text content from all successfully fetched URLs.
    """
    # Score and sort URLs by authority (high first)
    from app.services.source_scorer import score_and_sort, format_authority_tag
    scored_urls = score_and_sort(urls[:5])

    # Content types that indicate binary (non-text) responses
    _BINARY_TYPES = {"application/pdf", "application/octet-stream", "image/", "audio/", "video/"}

    results = []
    fetched = 0
    for url, url_score in scored_urls:
        tag = format_authority_tag(url_score)
        try:
            resp = requests.get(url, timeout=15, headers={"User-Agent": "ACBUDDY-Research/1.0"})
            resp.raise_for_status()
            # Skip binary responses (PDFs, images, etc.) — they cause ADK serialization errors
            ctype = (resp.headers.get("content-type", "") or "").lower().split(";")[0].strip()
            if any(ctype.startswith(bt) for bt in _BINARY_TYPES):
                logger.info("Skipping binary content (%s) from %s", ctype, url)
                results.append(f"[Source: {url}] {tag} (binary content, skipped)\n")
                continue
            # Force UTF-8 decoding to avoid encoding issues
            resp.encoding = resp.apparent_encoding or "utf-8"
            # Strip HTML tags
            text = re.sub(r"<[^>]+>", " ", resp.text)
            # Collapse whitespace
            text = re.sub(r"\s+", " ", text).strip()
            # Remove null bytes and other control characters that break proto serialization
            text = text.replace("\x00", "")
            # Truncate to 5K chars per source
            results.append(f"[Source: {url}] {tag}\n{text[:5000]}\n")
            fetched += 1
        except Exception as e:
            logger.warning("Failed to fetch %s: %s", url, e)
            results.append(f"[Source: {url}] {tag} Error: {e}\n")
    increment("urls_fetched", len(scored_urls))
    increment("pages_read", fetched)
    return "\n---\n".join(results)


def search_news(query: str) -> str:
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

    increment("news_searches")
    articles = news_client.search_news(query=query, api_key=api_key)
    if not articles:
        return f"No recent news found for: {query}"

    increment("news_articles", len(articles))
    parts = []
    for art in articles:
        parts.append(
            f"**{art['title']}** ({art['source']}, {art['published_at'][:10]})\n"
            f"{art['description']}\n"
            f"URL: {art['url']}"
        )
    return "\n\n---\n\n".join(parts)


def search_grok(query: str) -> str:
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

    increment("grok_queries")
    result = grok_client.search_with_grok(query=query, api_key=api_key)
    return result or f"No results from Grok for: {query}"


def deep_reason(question: str, context: str) -> str:
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

    increment("reasoning_calls")
    result = openai_client.deep_reason(
        question=question, context=context, api_key=api_key
    )
    return result or f"No reasoning output for: {question}"


def search_financial(query: str) -> str:
    """Search for financial data including stock prices, company fundamentals, and SEC filings.

    Use this tool when the research involves:
    - Stock prices, market caps, P/E ratios
    - Company financial fundamentals
    - SEC filings (10-K, 10-Q, 8-K)
    - Financial comparisons between companies

    Args:
        query: Financial search query (e.g., "AAPL stock overview" or "Tesla SEC 10-K filings").

    Returns:
        Formatted financial data with sources.
    """
    from app.services import financial_client
    from app.services.research_stats import increment

    increment("web_searches")
    parts = []

    # Detect ticker symbols (uppercase 1-5 letter words)
    import re
    tickers = re.findall(r'\b[A-Z]{1,5}\b', query)
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "")

    for ticker in tickers[:3]:
        data = financial_client.get_stock_data(ticker, api_key)
        if "error" not in data:
            parts.append(f"**{data.get('name', ticker)} ({ticker})**")
            for k, v in data.items():
                if k not in ("ticker", "name", "error") and v:
                    parts.append(f"  {k.replace('_', ' ').title()}: {v}")
            parts.append("")

    # Search SEC filings if query mentions filings, SEC, 10-K etc.
    filing_keywords = ["sec", "filing", "10-k", "10-q", "8-k", "annual report"]
    if any(kw in query.lower() for kw in filing_keywords):
        company = query.split("SEC")[0].split("filing")[0].strip() if "SEC" in query or "filing" in query.lower() else query
        filings = financial_client.search_sec_filings(company)
        if filings and "error" not in filings[0]:
            parts.append("**SEC Filings:**")
            for f in filings[:5]:
                parts.append(f"  - {f.get('form_type', '')} filed {f.get('filed_date', '')}: {f.get('url', '')}")
            parts.append("")

    return "\n".join(parts) if parts else f"No financial data found for: {query}"


def search_company(company_name: str) -> str:
    """Search for company profile and competitive intelligence.

    Use this tool when the research involves:
    - Company background and description
    - Funding history and investors
    - Employee count and growth
    - Competitive landscape

    Args:
        company_name: The company name to look up.

    Returns:
        Formatted company profile with sources.
    """
    from app.services import competitive_intel_client
    from app.services.research_stats import increment

    increment("web_searches")
    api_key = os.getenv("CRUNCHBASE_API_KEY", "")
    data = competitive_intel_client.get_company_profile(company_name, api_key)

    if "error" in data and not data.get("description"):
        return f"No company profile found for: {company_name}"

    parts = [f"**{data.get('name', company_name)}**"]
    if data.get("description"):
        parts.append(data["description"])
    for k in ["founded", "employees", "total_funding", "last_funding", "categories", "location"]:
        v = data.get(k)
        if v:
            if isinstance(v, list):
                v = ", ".join(str(x) for x in v)
            parts.append(f"  {k.replace('_', ' ').title()}: {v}")
    if data.get("url"):
        parts.append(f"  Source: {data['url']}")
    parts.append(f"  Data source: {data.get('source', 'unknown')}")

    return "\n".join(parts)


RESEARCHER_INSTRUCTION = """You are a thorough research agent with access to multiple search sources.
Your task is to research the following question using the best combination of tools.

Available tools and when to use them:
1. **web_search** — Default for all queries. Uses Gemini search grounding for broad web results.
2. **search_news** — For current events, recent developments, company/market news, regulatory changes.
3. **search_grok** — For trending topics, social sentiment, X/Twitter discussions, real-time reactions.
4. **deep_reason** — For complex analytical questions. Pass your gathered findings as context and
   ask it to reason about implications, causal chains, or strategic scenarios.
5. **pull_sources** — Fetch and read full content from specific URLs found in search results.
6. **search_financial** — For stock data, company fundamentals, P/E ratios, market caps, SEC filings.
   Use when researching publicly traded companies or financial topics.
7. **search_company** — For company profiles, funding history, employee counts, competitive intelligence.
   Use when researching specific companies or competitive landscapes.

Research strategy:
1. Start with web_search for foundational information.
2. If the question involves recent events or market developments, ALSO use search_news.
3. If the question involves public opinion, trends, or social dynamics, ALSO use search_grok.
4. If researching specific companies, use search_company and search_financial for hard data.
5. If you have gathered enough data but need deeper analysis, use deep_reason with your findings as context.
6. Use pull_sources to read full content from the most relevant URLs.
7. Synthesize ALL findings into a clear, detailed summary with citations.

Rules:
- Include specific facts, data points, and source URLs in your response.
- Every claim MUST be backed by a specific source URL. If you cannot verify a claim with a
  concrete source, DO NOT include it. Omit unverified or speculative information entirely.
- Stay strictly within the geographic, temporal, and topical scope of the question. If the
  question is about a specific country or region, only include data and examples from that
  geography. Do not pad findings with data from other regions.
- Sources include authority tags like [HIGH/MEDIUM/LOW AUTHORITY]. Heavily weight HIGH AUTHORITY
  sources in your findings. Treat LOW AUTHORITY sources as supplementary only — never let a LOW
  AUTHORITY source be the sole evidence for a key claim.
- Use at least 2 different search tools when the topic warrants it.
- Be thorough but concise. Focus on accuracy and relevance.
"""


def build_researcher(index: int, model: str = "gemini-2.5-flash", prefix: str = "research") -> LlmAgent:
    """Build an LlmAgent with web_search and pull_sources tools.

    Args:
        index: Researcher index (for naming and output key).
        model: Model to use.
        prefix: Output key prefix.

    Returns:
        Configured LlmAgent for deep research.
    """
    # Include multi-source tools only when API keys are configured
    tools = [web_search, pull_sources]
    if os.getenv("NEWSAPI_KEY", ""):
        tools.append(search_news)
    if os.getenv("GROK_API_KEY", ""):
        tools.append(search_grok)
    if os.getenv("OPENAI_API_KEY", ""):
        tools.append(deep_reason)
    # Domain tools — always available (they handle missing keys gracefully)
    tools.append(search_financial)
    tools.append(search_company)

    return LlmAgent(
        name=f"researcher_{index}",
        model=model,
        instruction=RESEARCHER_INSTRUCTION,
        tools=tools,
        output_key=f"{prefix}_{index}",
    )
