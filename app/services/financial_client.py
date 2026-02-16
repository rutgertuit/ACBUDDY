"""Financial data tools — stock data via Alpha Vantage / Yahoo Finance, SEC filings via EDGAR."""

import logging

import requests

logger = logging.getLogger(__name__)

_YAHOO_BASE = "https://query1.finance.yahoo.com/v8/finance"
_EDGAR_BASE = "https://efts.sec.gov/LATEST/search-index"
_ALPHA_VANTAGE_BASE = "https://www.alphavantage.co/query"


def get_stock_data(ticker: str, api_key: str = "") -> dict:
    """Get stock overview and recent price data.

    Uses Alpha Vantage if API key is provided, otherwise falls back to Yahoo Finance.
    Returns dict with: name, price, change, market_cap, pe_ratio, description.
    """
    if api_key:
        return _alpha_vantage_overview(ticker, api_key)
    return _yahoo_quote(ticker)


def _alpha_vantage_overview(ticker: str, api_key: str) -> dict:
    """Fetch company overview from Alpha Vantage."""
    try:
        resp = requests.get(
            _ALPHA_VANTAGE_BASE,
            params={"function": "OVERVIEW", "symbol": ticker, "apikey": api_key},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        if "Symbol" not in data:
            return {"error": f"No data for ticker {ticker}", "raw": str(data)[:200]}
        return {
            "ticker": data.get("Symbol", ticker),
            "name": data.get("Name", ""),
            "description": data.get("Description", "")[:500],
            "market_cap": data.get("MarketCapitalization", ""),
            "pe_ratio": data.get("PERatio", ""),
            "eps": data.get("EPS", ""),
            "dividend_yield": data.get("DividendYield", ""),
            "52_week_high": data.get("52WeekHigh", ""),
            "52_week_low": data.get("52WeekLow", ""),
            "sector": data.get("Sector", ""),
            "industry": data.get("Industry", ""),
        }
    except Exception as e:
        logger.warning("Alpha Vantage failed for %s: %s", ticker, e)
        return {"error": str(e)}


def _yahoo_quote(ticker: str) -> dict:
    """Fetch basic quote from Yahoo Finance (no API key needed)."""
    try:
        url = f"https://query1.finance.yahoo.com/v7/finance/quote"
        resp = requests.get(
            url,
            params={"symbols": ticker},
            headers={"User-Agent": "Luminary-Research/1.0"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("quoteResponse", {}).get("result", [])
        if not results:
            return {"error": f"No Yahoo Finance data for {ticker}"}
        q = results[0]
        return {
            "ticker": q.get("symbol", ticker),
            "name": q.get("shortName", ""),
            "price": q.get("regularMarketPrice", ""),
            "change_pct": q.get("regularMarketChangePercent", ""),
            "market_cap": q.get("marketCap", ""),
            "pe_ratio": q.get("trailingPE", ""),
            "volume": q.get("regularMarketVolume", ""),
        }
    except Exception as e:
        logger.warning("Yahoo Finance failed for %s: %s", ticker, e)
        return {"error": str(e)}


def search_sec_filings(company: str, filing_type: str = "10-K") -> list[dict]:
    """Search SEC EDGAR for company filings. Free API, no key needed.

    Args:
        company: Company name or ticker.
        filing_type: Filing type (10-K, 10-Q, 8-K, etc.).

    Returns:
        List of dicts with: title, filed_date, url, form_type.
    """
    try:
        url = "https://efts.sec.gov/LATEST/search-index"
        resp = requests.get(
            url,
            params={
                "q": company,
                "forms": filing_type,
                "dateRange": "custom",
                "startdt": "2023-01-01",
            },
            headers={"User-Agent": "Luminary-Research research@luminary.app"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        hits = data.get("hits", {}).get("hits", [])
        results = []
        for hit in hits[:10]:
            source = hit.get("_source", {})
            results.append({
                "title": source.get("display_names", [company])[0] if source.get("display_names") else company,
                "form_type": source.get("form_type", filing_type),
                "filed_date": source.get("file_date", ""),
                "url": f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&company={company}&type={filing_type}&dateb=&owner=include&count=10",
            })
        return results
    except Exception as e:
        logger.warning("SEC EDGAR search failed for %s: %s", company, e)
        # Fallback: use EDGAR full-text search
        try:
            resp = requests.get(
                "https://efts.sec.gov/LATEST/search-index",
                params={"q": f'"{company}" AND "{filing_type}"'},
                headers={"User-Agent": "Luminary-Research research@luminary.app"},
                timeout=15,
            )
            if resp.ok:
                return [{"title": company, "form_type": filing_type, "note": "Fallback search"}]
        except Exception:
            pass
        return [{"error": str(e)}]
