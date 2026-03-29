import pandas as pd
import requests
import io
import os
import json
import time
from pydantic import BaseModel, Field, field_validator
from typing import List, Literal, Optional
from strategy_ideation_engine.config.logger import logger

class TradeableUniverse(BaseModel):
    """
    Phase 0: Rigorous definition of allowed assets and constraints.
    Prevents the engine from hallucinating assets outside the scope.
    """
    allowed_tickers: List[str] = Field(..., description="The whitelist of tickers the engine can ideate on")
    base_currency: str = Field("USD", description="The reporting and trading base currency")
    min_market_cap_millions: float = Field(500.0, description="Minimum liquidity constraint")
    asset_classes: List[Literal["Equity", "Forex", "Crypto", "Commodity"]] = Field(
        default_factory=lambda: ["Equity"]
    )
    
    @field_validator("allowed_tickers")
    @classmethod
    def force_uppercase(cls, v: List[str]) -> List[str]:
        return [t.upper() for t in v]

def get_sp500_tickers() -> List[str]:
    """Dynamically fetches the current S&P 500 tickers from Wikipedia with local caching."""
    cache_dir = ".cache"
    cache_file = os.path.join(cache_dir, "sp500_tickers.json")
    cache_expiry = 86400  # 24 hours in seconds

    # Check cache first
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
                if time.time() - cached_data.get("timestamp", 0) < cache_expiry:
                    logger.info("Using cached S&P 500 tickers.")
                    return cached_data.get("tickers", [])
        except Exception as e:
            logger.warning(f"Failed to read S&P 500 cache: {e}")

    try:
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        tables = pd.read_html(io.StringIO(response.text))
        df = tables[0]
        tickers = df['Symbol'].tolist()
        # Clean tickers (Wikipedia uses '.' instead of '-' for some symbols)
        tickers = [t.replace('.', '-') for t in tickers]
        
        # Save to cache
        try:
            with open(cache_file, 'w') as f:
                json.dump({"timestamp": time.time(), "tickers": tickers}, f)
        except Exception as e:
            logger.warning(f"Failed to write S&P 500 cache: {e}")

        logger.info(f"Successfully fetched {len(tickers)} S&P 500 tickers.")
        return tickers
    except Exception as e:
        logger.error(f"Failed to fetch S&P 500 tickers: {e}. Falling back to core mega-caps.")
        return ["AAPL", "MSFT", "GOOGL", "NVDA", "AMD", "TSLA", "META", "AMZN", "NFLX", "BRK-B"]

# Sector to SPDR ETF Mapping
SECTOR_ETF_MAPPING = {
    "Technology": "XLK",
    "Financial Services": "XLF",
    "Healthcare": "XLV",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Energy": "XLE",
    "Utilities": "XLU",
    "Industrials": "XLI",
    "Basic Materials": "XLB",
    "Real Estate": "XLRE",
    "Communication Services": "XLC"
}

def get_sector_benchmark(sector: Optional[str]) -> str:
    """Returns the relevant sector ETF for a given GICS sector, defaults to SPY."""
    return SECTOR_ETF_MAPPING.get(sector, "SPY")

# Define our S&P 500 Universe
CORE_UNIVERSE = TradeableUniverse(
    allowed_tickers=get_sp500_tickers(),
    base_currency="USD",
    min_market_cap_millions=2000.0,
    asset_classes=["Equity"]
)
