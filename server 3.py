"""
mcp_servers/market_intelligence/server.py
─────────────────────────────────────────
Read-only Market Intelligence MCP server.
Deploy this on Alpic at: https://your-project.alpic.ai

This server consolidates every external data fetch the specialist agents
need into one hosted, authenticated, cached endpoint. Benefits:

  • Single rate-limit budget shared across all agent sessions
  • Response caching prevents redundant API calls (TTL configurable)
  • One stable URL — agents don't care about the underlying data sources
  • Alpic's analytics dashboard shows exactly which tools are called most

TOOLS EXPOSED (all read-only):
  get_price           — live price + 24h stats via CoinGecko
  get_ohlcv           — OHLCV candlestick history for technical analysis
  get_technical       — precomputed RSI, MACD, EMA, Bollinger Bands
  get_fear_greed      — CNN Fear & Greed index (crypto)
  get_on_chain        — on-chain DEX data via GeckoTerminal
  get_trending        — CoinGecko trending coins + categories
  scrape_news         — full article text via Firecrawl
  search_news         — web search for crypto news via Firecrawl
  get_market_overview — macro snapshot: BTC dominance, total market cap, top movers

AUTHENTICATION:
  Uses x-api-key header. Set HYDRA_API_KEY env var on Alpic.
  Agents pass the key via the MCP client config (see integrations/mcp_config.json).

DEPLOY ON ALPIC:
  1. Push this file + alpic.json to a GitHub repo
  2. Import repo in Alpic dashboard
  3. Set env vars: COINGECKO_API_KEY, FIRECRAWL_API_KEY, HYDRA_API_KEY
  4. Alpic auto-detects FastMCP + streamable-http transport
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from functools import lru_cache
from typing import Any

import httpx
from mcp.server.fastmcp import Context, FastMCP
from pydantic import Field

# ── Server bootstrap ─────────────────────────────────────────────────────────
mcp = FastMCP(
    "Hydra Market Intelligence",
    stateless_http=True,
    instructions=(
        "Read-only market intelligence for the Hydra Sentinel-X trading system. "
        "All tools return JSON strings. Never execute trades — data only."
    ),
)

# ── Environment ───────────────────────────────────────────────────────────────
COINGECKO_KEY  = os.getenv("COINGECKO_API_KEY", "")
FIRECRAWL_KEY  = os.getenv("FIRECRAWL_API_KEY", "")
HYDRA_API_KEY  = os.getenv("HYDRA_API_KEY", "")
CACHE_TTL      = int(os.getenv("CACHE_TTL_SECONDS", "30"))

# Simple in-process TTL cache
_cache: dict[str, tuple[float, Any]] = {}

def _cached(key: str, ttl: int = CACHE_TTL) -> Any | None:
    if key in _cache:
        ts, val = _cache[key]
        if time.time() - ts < ttl:
            return val
    return None

def _set_cache(key: str, val: Any) -> None:
    _cache[key] = (time.time(), val)

# CoinGecko symbol → id mapping
COIN_MAP = {
    "BTC":"bitcoin","ETH":"ethereum","SOL":"solana","BNB":"binancecoin",
    "XRP":"ripple","ADA":"cardano","DOGE":"dogecoin","AVAX":"avalanche-2",
    "DOT":"polkadot","LINK":"chainlink","MATIC":"matic-network","UNI":"uniswap",
    "LTC":"litecoin","ATOM":"cosmos","NEAR":"near","ARB":"arbitrum",
    "OP":"optimism","SUI":"sui","INJ":"injective-protocol","TIA":"celestia",
    "TON":"the-open-network","PEPE":"pepe","WIF":"dogwifcoin",
}

def _coin_id(symbol: str) -> str:
    return COIN_MAP.get(symbol.upper(), symbol.lower())

def _cg_headers() -> dict:
    h = {"Accept": "application/json"}
    if COINGECKO_KEY:
        h["x-cg-pro-api-key"] = COINGECKO_KEY
    return h

def _auth(ctx: Context) -> bool:
    """Validate x-api-key header if HYDRA_API_KEY is configured."""
    if not HYDRA_API_KEY:
        return True   # open access (dev mode)
    key = ctx.request_context.request.headers.get("x-api-key", "")
    return key == HYDRA_API_KEY


# ── Tool 1: Live Price ────────────────────────────────────────────────────────

@mcp.tool(
    title="Get Crypto Price",
    description=(
        "Fetch live price, 24h change, market cap, and volume for a crypto symbol. "
        "Returns JSON with price_usd, price_change_24h, market_cap, volume_24h, "
        "ath, atl, circulating_supply."
    ),
)
async def get_price(
    ctx: Context,
    symbol: str = Field(description="Crypto symbol, e.g. BTC, ETH, SOL"),
) -> str:
    if not _auth(ctx):
        raise ValueError("Invalid API key")

    cache_key = f"price:{symbol.upper()}"
    if cached := _cached(cache_key, ttl=15):
        return cached

    coin_id = _coin_id(symbol)
    url = (
        f"https://api.coingecko.com/api/v3/coins/{coin_id}"
        "?localization=false&tickers=false&community_data=false"
        "&developer_data=false&sparkline=true"
    )
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url, headers=_cg_headers())
        r.raise_for_status()
        d = r.json()

    md = d.get("market_data", {})
    result = json.dumps({
        "symbol":          symbol.upper(),
        "name":            d.get("name"),
        "price_usd":       md.get("current_price", {}).get("usd"),
        "price_change_24h":md.get("price_change_percentage_24h"),
        "price_change_7d": md.get("price_change_percentage_7d"),
        "market_cap":      md.get("market_cap", {}).get("usd"),
        "volume_24h":      md.get("total_volume", {}).get("usd"),
        "ath":             md.get("ath", {}).get("usd"),
        "atl":             md.get("atl", {}).get("usd"),
        "circulating_supply": md.get("circulating_supply"),
        "sparkline_7d":    md.get("sparkline_7d", {}).get("price", [])[-24:],
    })
    _set_cache(cache_key, result)
    return result


# ── Tool 2: OHLCV History ─────────────────────────────────────────────────────

@mcp.tool(
    title="Get OHLCV History",
    description=(
        "Fetch OHLCV candlestick data for technical analysis. "
        "Returns a list of {timestamp, open, high, low, close, volume} objects."
    ),
)
async def get_ohlcv(
    ctx: Context,
    symbol: str = Field(description="Crypto symbol, e.g. BTC"),
    days: int   = Field(default=30, description="Number of days of history (1, 7, 14, 30, 90, 180, 365)"),
) -> str:
    if not _auth(ctx):
        raise ValueError("Invalid API key")

    cache_key = f"ohlcv:{symbol.upper()}:{days}"
    if cached := _cached(cache_key, ttl=300):  # 5 min cache for historical
        return cached

    coin_id = _coin_id(symbol)
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc?vs_currency=usd&days={days}"

    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(url, headers=_cg_headers())
        r.raise_for_status()
        raw = r.json()  # [[timestamp_ms, open, high, low, close], ...]

    candles = [
        {"timestamp": c[0], "open": c[1], "high": c[2], "low": c[3], "close": c[4]}
        for c in raw
    ]
    result = json.dumps({"symbol": symbol.upper(), "days": days, "candles": candles})
    _set_cache(cache_key, result)
    return result


# ── Tool 3: Precomputed Technical Indicators ──────────────────────────────────

@mcp.tool(
    title="Get Technical Indicators",
    description=(
        "Compute RSI(14), MACD(12/26/9), Bollinger Bands(20/2), and EMA(20/50/200) "
        "from recent OHLCV data. Returns JSON with all indicator values and "
        "human-readable signals."
    ),
)
async def get_technical(
    ctx: Context,
    symbol: str = Field(description="Crypto symbol, e.g. ETH"),
    days: int   = Field(default=90, description="Days of history to use for calculations"),
) -> str:
    if not _auth(ctx):
        raise ValueError("Invalid API key")

    cache_key = f"technical:{symbol.upper()}:{days}"
    if cached := _cached(cache_key, ttl=60):
        return cached

    # Fetch OHLCV inline — avoids double network call
    coin_id = _coin_id(symbol)
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc?vs_currency=usd&days={days}"
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(url, headers=_cg_headers())
        r.raise_for_status()
        raw = r.json()

    closes = [c[4] for c in raw]
    highs  = [c[2] for c in raw]
    lows   = [c[3] for c in raw]

    def ema(data: list[float], period: int) -> list[float]:
        k = 2 / (period + 1)
        result = [data[0]]
        for p in data[1:]:
            result.append(p * k + result[-1] * (1 - k))
        return result

    # RSI
    def rsi(data: list[float], period: int = 14) -> float:
        deltas = [data[i] - data[i-1] for i in range(1, len(data))]
        gains  = [max(d, 0) for d in deltas[-period:]]
        losses = [abs(min(d, 0)) for d in deltas[-period:]]
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return round(100 - (100 / (1 + rs)), 2)

    # MACD
    ema12 = ema(closes, 12)
    ema26 = ema(closes, 26)
    macd_line = [ema12[i] - ema26[i] for i in range(len(closes))]
    signal_line = ema(macd_line, 9)
    histogram = [macd_line[i] - signal_line[i] for i in range(len(macd_line))]

    # Bollinger Bands
    bb_period = 20
    bb_closes = closes[-bb_period:]
    bb_mid    = sum(bb_closes) / bb_period
    bb_std    = (sum((c - bb_mid) ** 2 for c in bb_closes) / bb_period) ** 0.5
    bb_upper  = bb_mid + 2 * bb_std
    bb_lower  = bb_mid - 2 * bb_std
    current   = closes[-1]
    bb_pct    = round((current - bb_lower) / (bb_upper - bb_lower) * 100, 1) if bb_upper != bb_lower else 50.0

    # EMAs
    ema20  = ema(closes, 20)[-1]
    ema50  = ema(closes, 50)[-1]
    ema200 = ema(closes, 200)[-1] if len(closes) >= 200 else None

    rsi_val = rsi(closes)
    macd_val = round(macd_line[-1], 4)
    hist_val = round(histogram[-1], 4)

    result = json.dumps({
        "symbol":   symbol.upper(),
        "price":    round(current, 4),
        "rsi":      rsi_val,
        "rsi_signal": "overbought" if rsi_val > 70 else "oversold" if rsi_val < 30 else "neutral",
        "macd":     {"macd": macd_val, "signal": round(signal_line[-1], 4), "histogram": hist_val},
        "macd_signal": "buy" if hist_val > 0 and histogram[-2] < 0 else
                        "sell" if hist_val < 0 and histogram[-2] > 0 else "neutral",
        "bollinger": {"upper": round(bb_upper, 4), "mid": round(bb_mid, 4),
                      "lower": round(bb_lower, 4), "pct_b": bb_pct},
        "bb_signal": "near_upper" if bb_pct > 80 else "near_lower" if bb_pct < 20 else "middle",
        "ema": {"ema20": round(ema20, 4), "ema50": round(ema50, 4),
                "ema200": round(ema200, 4) if ema200 else None},
        "ema_trend": "uptrend" if current > ema20 > ema50 else
                     "downtrend" if current < ema20 < ema50 else "mixed",
    })
    _set_cache(cache_key, result)
    return result


# ── Tool 4: Fear & Greed Index ────────────────────────────────────────────────

@mcp.tool(
    title="Get Fear and Greed Index",
    description="Fetch the current Crypto Fear & Greed Index (0=Extreme Fear, 100=Extreme Greed).",
)
async def get_fear_greed(ctx: Context) -> str:
    if not _auth(ctx):
        raise ValueError("Invalid API key")

    cache_key = "fear_greed"
    if cached := _cached(cache_key, ttl=300):
        return cached

    url = "https://api.alternative.me/fng/?limit=3"
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url)
        r.raise_for_status()
        d = r.json()

    data   = d.get("data", [{}])
    latest = data[0]
    result = json.dumps({
        "value":              int(latest.get("value", 50)),
        "classification":     latest.get("value_classification", "Neutral"),
        "previous_value":     int(data[1].get("value", 50)) if len(data) > 1 else None,
        "previous_class":     data[1].get("value_classification") if len(data) > 1 else None,
        "direction":          "improving" if len(data) > 1 and int(data[0]["value"]) > int(data[1]["value"]) else "worsening",
    })
    _set_cache(cache_key, result)
    return result


# ── Tool 5: On-Chain DEX Data (GeckoTerminal) ─────────────────────────────────

@mcp.tool(
    title="Get On-Chain DEX Data",
    description=(
        "Fetch on-chain DEX liquidity and trading data from GeckoTerminal. "
        "Returns top pools, 24h volume, and liquidity depth for a token. "
        "Useful for detecting wash trading (high volume, low liquidity)."
    ),
)
async def get_on_chain(
    ctx: Context,
    symbol:  str = Field(description="Token symbol, e.g. ETH, SOL, PEPE"),
    network: str = Field(default="eth", description="Network: eth, bsc, solana, base, arbitrum"),
) -> str:
    if not _auth(ctx):
        raise ValueError("Invalid API key")

    cache_key = f"onchain:{symbol.lower()}:{network}"
    if cached := _cached(cache_key, ttl=60):
        return cached

    url = f"https://api.geckoterminal.com/api/v2/search/pools?query={symbol}&network={network}&page=1"
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url, headers={"Accept": "application/json"})
        r.raise_for_status()
        d = r.json()

    pools = d.get("data", [])[:5]
    pool_data = []
    for p in pools:
        attr = p.get("attributes", {})
        pool_data.append({
            "name":        attr.get("name"),
            "dex":         attr.get("dex_id"),
            "price_usd":   attr.get("base_token_price_usd"),
            "volume_24h":  attr.get("volume_usd", {}).get("h24"),
            "liquidity":   attr.get("reserve_in_usd"),
            "price_change_24h": attr.get("price_change_percentage", {}).get("h24"),
            "txns_24h":    attr.get("transactions", {}).get("h24", {}).get("buys", 0)
                           + attr.get("transactions", {}).get("h24", {}).get("sells", 0),
        })

    # Volume/liquidity ratio — high ratio can indicate wash trading
    for p in pool_data:
        try:
            p["vol_liq_ratio"] = round(
                float(p["volume_24h"] or 0) / float(p["liquidity"] or 1), 2
            )
        except Exception:
            p["vol_liq_ratio"] = None

    result = json.dumps({
        "symbol":  symbol.upper(),
        "network": network,
        "pools":   pool_data,
        "note":    "vol_liq_ratio > 5 may indicate wash trading or low liquidity risk",
    })
    _set_cache(cache_key, result)
    return result


# ── Tool 6: Trending Coins ────────────────────────────────────────────────────

@mcp.tool(
    title="Get Trending Coins",
    description=(
        "Fetch CoinGecko's trending coins and top movers in the last 24h. "
        "Useful for identifying momentum and market rotation signals."
    ),
)
async def get_trending(ctx: Context) -> str:
    if not _auth(ctx):
        raise ValueError("Invalid API key")

    cache_key = "trending"
    if cached := _cached(cache_key, ttl=120):
        return cached

    url = "https://api.coingecko.com/api/v3/search/trending"
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url, headers=_cg_headers())
        r.raise_for_status()
        d = r.json()

    coins = [
        {
            "name":   c["item"]["name"],
            "symbol": c["item"]["symbol"],
            "rank":   c["item"]["market_cap_rank"],
            "price_change_24h": c["item"].get("data", {}).get("price_change_percentage_24h", {}).get("usd"),
        }
        for c in d.get("coins", [])[:10]
    ]
    result = json.dumps({"trending_coins": coins, "as_of": time.strftime("%Y-%m-%dT%H:%M:%SZ")})
    _set_cache(cache_key, result)
    return result


# ── Tool 7: Scrape News Article ───────────────────────────────────────────────

@mcp.tool(
    title="Scrape News Article",
    description=(
        "Fetch the full text of a news article or webpage via Firecrawl. "
        "Use this after getting headlines to read the actual content. "
        "Returns cleaned markdown text, title, and description."
    ),
)
async def scrape_news(
    ctx: Context,
    url: str = Field(description="Full URL of the article to scrape"),
) -> str:
    if not _auth(ctx):
        raise ValueError("Invalid API key")

    if not FIRECRAWL_KEY:
        return json.dumps({"error": "FIRECRAWL_API_KEY not configured on this server"})

    api_url = "https://api.firecrawl.dev/v1/scrape"
    payload = {
        "url": url,
        "formats": ["markdown"],
        "onlyMainContent": True,
        "maxLength": 3000,   # cap at 3k chars — enough for sentiment analysis
    }
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.post(
            api_url,
            json=payload,
            headers={"Authorization": f"Bearer {FIRECRAWL_KEY}", "Content-Type": "application/json"},
        )
        r.raise_for_status()
        d = r.json()

    data = d.get("data", {})
    return json.dumps({
        "url":         url,
        "title":       data.get("metadata", {}).get("title", ""),
        "description": data.get("metadata", {}).get("description", ""),
        "content":     data.get("markdown", "")[:3000],
        "scraped_at":  time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    })


# ── Tool 8: Search Crypto News ────────────────────────────────────────────────

@mcp.tool(
    title="Search Crypto News",
    description=(
        "Search the web for recent news about a crypto topic via Firecrawl. "
        "Returns a list of {title, url, description, date} results. "
        "Call scrape_news on the most relevant result to get full text."
    ),
)
async def search_news(
    ctx: Context,
    query: str  = Field(description="Search query, e.g. 'Ethereum ETF approval 2025'"),
    limit: int  = Field(default=5, description="Number of results to return"),
) -> str:
    if not _auth(ctx):
        raise ValueError("Invalid API key")

    if not FIRECRAWL_KEY:
        return json.dumps({"error": "FIRECRAWL_API_KEY not configured on this server"})

    api_url = "https://api.firecrawl.dev/v1/search"
    payload = {"query": query, "limit": limit, "scrapeOptions": {"formats": []}}

    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(
            api_url,
            json=payload,
            headers={"Authorization": f"Bearer {FIRECRAWL_KEY}", "Content-Type": "application/json"},
        )
        r.raise_for_status()
        d = r.json()

    results = [
        {
            "title":       item.get("title"),
            "url":         item.get("url"),
            "description": item.get("description"),
            "published":   item.get("publishedDate"),
        }
        for item in d.get("data", [])[:limit]
    ]
    return json.dumps({"query": query, "results": results})


# ── Tool 9: Market Overview ───────────────────────────────────────────────────

@mcp.tool(
    title="Get Market Overview",
    description=(
        "Global crypto market snapshot: total market cap, BTC dominance, "
        "ETH dominance, 24h volume, and number of active coins. "
        "Essential macro context for every analysis."
    ),
)
async def get_market_overview(ctx: Context) -> str:
    if not _auth(ctx):
        raise ValueError("Invalid API key")

    cache_key = "market_overview"
    if cached := _cached(cache_key, ttl=120):
        return cached

    url = "https://api.coingecko.com/api/v3/global"
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url, headers=_cg_headers())
        r.raise_for_status()
        d = r.json().get("data", {})

    result = json.dumps({
        "total_market_cap_usd":  d.get("total_market_cap", {}).get("usd"),
        "total_volume_24h_usd":  d.get("total_volume", {}).get("usd"),
        "btc_dominance_pct":     round(d.get("market_cap_percentage", {}).get("btc", 0), 2),
        "eth_dominance_pct":     round(d.get("market_cap_percentage", {}).get("eth", 0), 2),
        "market_cap_change_24h": d.get("market_cap_change_percentage_24h_usd"),
        "active_coins":          d.get("active_cryptocurrencies"),
        "markets":               d.get("markets"),
    })
    _set_cache(cache_key, result)
    return result


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Local dev: run with streamable-http transport on port 8001
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8001)
