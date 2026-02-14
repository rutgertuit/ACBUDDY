"""Competitive intelligence tools — company profiles, market data."""

import logging

import requests

logger = logging.getLogger(__name__)


def get_company_profile(company_name: str, api_key: str = "") -> dict:
    """Get a company profile with basic business intelligence.

    Uses Crunchbase if API key is provided, otherwise uses free web sources.

    Args:
        company_name: Company name to look up.
        api_key: Optional Crunchbase API key.

    Returns:
        Dict with company information.
    """
    if api_key:
        return _crunchbase_profile(company_name, api_key)
    return _free_company_lookup(company_name)


def _crunchbase_profile(company_name: str, api_key: str) -> dict:
    """Fetch company profile from Crunchbase."""
    try:
        # Search for organization
        search_url = "https://api.crunchbase.com/api/v4/autocompletes"
        resp = requests.get(
            search_url,
            params={"query": company_name, "collection_ids": "organizations"},
            headers={"X-cb-user-key": api_key},
            timeout=15,
        )
        resp.raise_for_status()
        entities = resp.json().get("entities", [])
        if not entities:
            return {"name": company_name, "error": "Not found on Crunchbase"}

        entity = entities[0]
        permalink = entity.get("identifier", {}).get("permalink", "")

        # Fetch full profile
        profile_url = f"https://api.crunchbase.com/api/v4/entities/organizations/{permalink}"
        resp = requests.get(
            profile_url,
            params={"field_ids": "short_description,num_employees_enum,founded_on,categories,location_identifiers,funding_total,last_funding_type"},
            headers={"X-cb-user-key": api_key},
            timeout=15,
        )
        resp.raise_for_status()
        props = resp.json().get("properties", {})

        return {
            "name": entity.get("identifier", {}).get("value", company_name),
            "description": props.get("short_description", ""),
            "founded": props.get("founded_on", ""),
            "employees": props.get("num_employees_enum", ""),
            "total_funding": props.get("funding_total", {}).get("value_usd", ""),
            "last_funding": props.get("last_funding_type", ""),
            "categories": [c.get("value", "") for c in props.get("categories", [])],
            "location": [l.get("value", "") for l in props.get("location_identifiers", [])],
            "source": "crunchbase",
        }
    except Exception as e:
        logger.warning("Crunchbase lookup failed for %s: %s", company_name, e)
        return {"name": company_name, "error": str(e), "source": "crunchbase"}


def _free_company_lookup(company_name: str) -> dict:
    """Free company lookup using Wikipedia API as fallback."""
    try:
        resp = requests.get(
            "https://en.wikipedia.org/api/rest_v1/page/summary/" + company_name.replace(" ", "_"),
            headers={"User-Agent": "ACBUDDY-Research/1.0"},
            timeout=10,
        )
        if resp.ok:
            data = resp.json()
            return {
                "name": data.get("title", company_name),
                "description": data.get("extract", "")[:500],
                "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                "source": "wikipedia",
            }
        return {"name": company_name, "error": "No free profile data available"}
    except Exception as e:
        logger.warning("Free company lookup failed for %s: %s", company_name, e)
        return {"name": company_name, "error": str(e)}


def compare_companies(companies: list[str], api_key: str = "") -> list[dict]:
    """Get profiles for multiple companies for comparison.

    Args:
        companies: List of company names.
        api_key: Optional Crunchbase API key.

    Returns:
        List of company profile dicts.
    """
    return [get_company_profile(c, api_key) for c in companies[:5]]
