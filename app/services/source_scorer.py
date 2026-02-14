"""Static domain-reputation scoring for source URLs.

Zero API cost — uses a dictionary of known domains to rate authority.
"""

import logging
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# High-authority domains: government, academic, major news, scientific
TIER_HIGH = {
    # Government & international orgs
    "gov", "edu", "mil",
    "who.int", "worldbank.org", "imf.org", "oecd.org", "ecb.europa.eu",
    "europa.eu", "un.org", "wto.org", "bis.org", "fed.gov",
    # Major news & wire services
    "reuters.com", "apnews.com", "bloomberg.com", "ft.com",
    "wsj.com", "nytimes.com", "washingtonpost.com", "economist.com",
    "theguardian.com", "bbc.co.uk",
    # Academic & scientific
    "nature.com", "sciencedirect.com", "pubmed.ncbi.nlm.nih.gov",
    "ncbi.nlm.nih.gov", "arxiv.org", "scholar.google.com",
    "jstor.org", "springer.com", "wiley.com", "thelancet.com",
    "bmj.com", "pnas.org", "science.org", "cell.com",
    # Data & statistics
    "data.gov", "census.gov", "bls.gov", "eurostat.ec.europa.eu",
    "statista.com",
}

# Medium-authority domains: reputable media, industry analysis, reference
TIER_MEDIUM = {
    # Reputable media
    "bbc.com", "cnbc.com", "cnn.com", "npr.org",
    "politico.com", "theatlantic.com", "newyorker.com",
    # Tech & business
    "techcrunch.com", "wired.com", "arstechnica.com", "theverge.com",
    "hbr.org", "mckinsey.com", "bcg.com", "bain.com",
    "gartner.com", "forrester.com", "deloitte.com", "pwc.com",
    "ey.com", "kpmg.com", "accenture.com",
    # Finance
    "investopedia.com", "morningstar.com", "seekingalpha.com",
    "yahoo.com", "marketwatch.com",
    # Reference & encyclopedias
    "wikipedia.org", "britannica.com",
    # Forbes etc.
    "forbes.com", "businessinsider.com", "fortune.com",
    "inc.com", "entrepreneur.com",
}

# Flags for source classification
_ACADEMIC_TLDS = {"edu", "ac.uk", "edu.au"}
_GOV_TLDS = {"gov", "mil", "gov.uk", "gov.au"}
_NEWS_DOMAINS = {
    "reuters.com", "apnews.com", "bloomberg.com", "ft.com", "wsj.com",
    "nytimes.com", "bbc.com", "bbc.co.uk", "cnbc.com", "cnn.com",
    "theguardian.com", "npr.org", "washingtonpost.com", "economist.com",
}
_BLOG_INDICATORS = {"medium.com", "substack.com", "blogspot.com", "wordpress.com"}


def _extract_domain(url: str) -> str:
    """Extract the registrable domain from a URL."""
    try:
        parsed = urlparse(url if "://" in url else f"https://{url}")
        hostname = (parsed.hostname or "").lower().strip(".")
        return hostname
    except Exception:
        return ""


def _get_tld(hostname: str) -> str:
    """Get the effective TLD from a hostname."""
    parts = hostname.split(".")
    if len(parts) >= 2:
        return parts[-1]
    return ""


def _get_root_domain(hostname: str) -> str:
    """Get root domain (e.g., 'reuters.com' from 'www.reuters.com')."""
    parts = hostname.split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return hostname


def score_url(url: str) -> dict:
    """Score a URL's authority based on domain reputation.

    Returns:
        {"authority_score": 0-10, "tier": "high"|"medium"|"low", "flags": [...]}
    """
    hostname = _extract_domain(url)
    if not hostname:
        return {"authority_score": 2, "tier": "low", "flags": ["unknown"]}

    tld = _get_tld(hostname)
    root = _get_root_domain(hostname)
    flags = []

    # Check TLD-based classifications
    if tld in _GOV_TLDS or any(hostname.endswith(f".{g}") for g in _GOV_TLDS):
        flags.append("government")
    if tld in _ACADEMIC_TLDS or any(hostname.endswith(f".{a}") for a in _ACADEMIC_TLDS):
        flags.append("academic")
    if root in _NEWS_DOMAINS or hostname in _NEWS_DOMAINS:
        flags.append("news")
    if root in _BLOG_INDICATORS or hostname in _BLOG_INDICATORS:
        flags.append("blog")

    # Check against tier dictionaries
    # Match by: full hostname, root domain, or TLD
    is_high = (
        hostname in TIER_HIGH
        or root in TIER_HIGH
        or tld in TIER_HIGH
        or any(hostname.endswith(f".{d}") for d in TIER_HIGH if "." not in d)
    )

    is_medium = (
        hostname in TIER_MEDIUM
        or root in TIER_MEDIUM
    )

    if is_high:
        tier = "high"
        score = 9
        if "academic" in flags:
            score = 10
        elif "government" in flags:
            score = 9
    elif is_medium:
        tier = "medium"
        score = 6
        if "news" in flags:
            score = 7
    else:
        tier = "low"
        score = 3
        if "blog" in flags:
            score = 2
        elif not flags:
            flags.append("commercial" if tld == "com" else "general")

    return {"authority_score": score, "tier": tier, "flags": flags}


def score_and_sort(urls: list[str]) -> list[tuple[str, dict]]:
    """Score URLs and return sorted highest-authority first."""
    scored = [(url, score_url(url)) for url in urls]
    scored.sort(key=lambda x: x[1]["authority_score"], reverse=True)
    return scored


def format_authority_tag(score: dict) -> str:
    """Format a score dict as a readable authority tag.

    Returns e.g. '[HIGH AUTHORITY: academic]' or '[LOW AUTHORITY: blog]'
    """
    tier = score.get("tier", "low").upper()
    flags = score.get("flags", [])
    flag_str = ", ".join(flags) if flags else "general"
    return f"[{tier} AUTHORITY: {flag_str}]"
