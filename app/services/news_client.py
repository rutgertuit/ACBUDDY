"""Thin wrapper around NewsAPI.org for recent news search."""

import logging
from datetime import datetime, timedelta, timezone

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://newsapi.org/v2/everything"


def search_news(
    query: str,
    api_key: str,
    days_back: int = 30,
    max_results: int = 10,
) -> list[dict]:
    """Search recent news articles via NewsAPI.

    Returns list of dicts with keys: title, description, url, published_at, source.
    Returns empty list on failure or if no API key.
    """
    if not api_key:
        return []

    from_date = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")

    try:
        resp = requests.get(
            BASE_URL,
            params={
                "q": query,
                "from": from_date,
                "sortBy": "relevancy",
                "pageSize": max_results,
                "language": "en",
                "apiKey": api_key,
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        articles = []
        for art in data.get("articles", []):
            articles.append({
                "title": art.get("title", ""),
                "description": art.get("description", ""),
                "url": art.get("url", ""),
                "published_at": art.get("publishedAt", ""),
                "source": art.get("source", {}).get("name", ""),
            })

        logger.info("NewsAPI returned %d articles for: %s", len(articles), query[:80])
        return articles

    except Exception as e:
        logger.warning("NewsAPI search failed for '%s': %s", query[:80], e)
        return []
