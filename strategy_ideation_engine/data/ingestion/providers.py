from typing import List, Dict, Optional
from datetime import datetime
import yfinance as yf
from fredapi import Fred
from alpha_vantage.techindicators import TechIndicators
from edgar import set_identity, Company
from strategy_ideation_engine.config import settings, logger
from strategy_ideation_engine.schemas.market_data import (
    FundamentalSnapshot, 
    MacroSnapshot, 
    MarketContext
)

import os
import json
import time

class MacroProvider:
    """Fetches macroeconomic data from FRED with local caching."""
    def __init__(self):
        self._fred = None
        self.cache_dir = ".cache"
        self.cache_file = os.path.join(self.cache_dir, "macro_snapshot.json")
        self.cache_expiry = 3600  # 1 hour

    @property
    def fred(self):
        """Lazy initializer for Fred client with environment fallback."""
        if self._fred is None:
            key = settings.FRED_API_KEY or os.getenv("FRED_API_KEY")
            if key:
                self._fred = Fred(api_key=key)
        return self._fred

    def get_macro_snapshot(self) -> MacroSnapshot:
        # Check cache
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    cached_data = json.load(f)
                    if time.time() - cached_data.get("timestamp", 0) < self.cache_expiry:
                        logger.info("Using cached Macro snapshot.")
                        return MacroSnapshot(**cached_data.get("data", {}))
            except Exception as e:
                logger.warning(f"Failed to read Macro cache: {e}")

        client = self.fred
        if not client:
            logger.warning("FRED API key missing. Returning empty MacroSnapshot.")
            return MacroSnapshot()
        
        try:
            # Series: FEDFUNDS (Fed Funds Rate), CPIAUCSL (CPI YoY), UNRATE (Unemployment)
            fed_funds = float(client.get_series('FEDFUNDS').iloc[-1])
            cpi = float(client.get_series('CPIAUCSL').pct_change(periods=12).iloc[-1] * 100)
            unrate = float(client.get_series('UNRATE').iloc[-1])
            
            snapshot = MacroSnapshot(
                fed_funds_rate=fed_funds,
                cpi_yoy_change=cpi,
                unemployment_rate=unrate
            )

            # Save to cache
            try:
                if not os.path.exists(self.cache_dir):
                    os.makedirs(self.cache_dir)
                with open(self.cache_file, 'w') as f:
                    # Use mode='json' to ensure Pydantic handles datetime correctly
                    json.dump({"timestamp": time.time(), "data": snapshot.model_dump(mode='json')}, f)
            except Exception as e:
                logger.warning(f"Failed to write Macro cache: {e}")

            return snapshot
        except Exception as e:
            logger.error(f"Error fetching FRED data: {e}")
            return MacroSnapshot()

class SECProvider:
    """Fetches official regulatory filing data from SEC EDGAR."""
    def __init__(self):
        if settings.SEC_EDGAR_USER_AGENT:
            set_identity(settings.SEC_EDGAR_USER_AGENT)
            self.active = True
        else:
            self.active = False

    def get_latest_filing_summary(self, ticker: str) -> str:
        if not self.active:
            return "SEC Data not available (User-Agent missing)."
        
        try:
            company = Company(ticker)
            filings = company.get_filings(form=["10-K", "10-Q"])
            latest = filings.latest()
            if latest:
                return f"Latest {latest.form} filed on {latest.filing_date}"
            return "No recent 10-K or 10-Q found."
        except Exception as e:
            logger.warning(f"Error fetching SEC data for {ticker}: {e}")
            return "SEC lookup failed."

import pandas_ta as ta

class FundamentalProvider:
    """Fetches fundamental and technical data with in-memory and disk caching."""
    def __init__(self):
        self.ti = TechIndicators(key=settings.ALPHA_VANTAGE_API_KEY, output_format='pandas') if settings.ALPHA_VANTAGE_API_KEY else None
        self._fund_cache = {}
        self._tech_cache = {}
        self.cache_dir = os.path.join(".cache", "tech_data")
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def get_ticker_fundamentals(self, ticker: str) -> FundamentalSnapshot:
        # Check in-memory cache
        if ticker in self._fund_cache:
            return self._fund_cache[ticker]
            
        try:
            yt = yf.Ticker(ticker)
            info = yt.info
            
            # Fetch Earnings Date
            next_earnings = "Unknown"
            try:
                calendar = yt.calendar
                if calendar is not None and not calendar.empty:
                    if hasattr(calendar, 'iloc'):
                        next_earnings = str(calendar.iloc[0, 0])
                    elif 'Earnings Date' in calendar:
                        next_earnings = str(calendar['Earnings Date'][0])
            except: pass

            # Extract key metrics with fallbacks
            ev = info.get("enterpriseValue")
            ebitda = info.get("ebitda")
            ev_ebitda = (ev / ebitda) if ev and ebitda and ebitda != 0 else None

            snapshot = FundamentalSnapshot(
                ticker=ticker,
                sector=info.get("sector"),
                industry=info.get("industry"),
                pe_ratio=info.get("forwardPE"),
                ev_to_ebitda=ev_ebitda,
                pb_ratio=info.get("priceToBook"),
                debt_to_equity=info.get("debtToEquity"),
                revenue_growth_yoy=info.get("revenueGrowth"),
                free_cash_flow=info.get("freeCashflow"),
                operating_margins=info.get("operatingMargins"),
                return_on_equity=info.get("returnOnEquity"),
                beta=info.get("beta"),
                trailing_eps=info.get("trailingEps"),
                next_earnings_date=next_earnings
            )
            self._fund_cache[ticker] = snapshot
            return snapshot
        except Exception as e:
            logger.error(f"Error fetching yfinance fundamentals for {ticker}: {e}")
            # FALLBACK: Try Alpha Vantage for Fundamentals if yfinance fails?
            # For now, return empty snapshot
            return FundamentalSnapshot(ticker=ticker)

    def get_competitors(self, ticker: str) -> List[str]:
        """Attempts to find direct competitors using simple heuristics for major tickers."""
        tech_competitors = {
            "NVDA": ["AMD", "INTC", "TSM"],
            "AAPL": ["MSFT", "GOOGL", "AMZN"],
            "MSFT": ["AAPL", "GOOGL", "ORCL"],
            "AMD": ["NVDA", "INTC"],
            "TSLA": ["RIVN", "LCID", "F"],
            "GOOGL": ["META", "MSFT", "AMZN"]
        }
        return tech_competitors.get(ticker.upper(), [])

    def get_technical_indicators(self, ticker: str) -> Dict[str, float]:
        """
        Primary technical indicator fetcher. 
        Uses yfinance + pandas_ta for local calculation to avoid Alpha Vantage limits.
        Checks disk cache first.
        """
        if ticker in self._tech_cache:
            return self._tech_cache[ticker]
        
        cache_file = os.path.join(self.cache_dir, f"{ticker.upper()}_tech.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                    # Cache valid for 4 hours for technicals
                    if time.time() - cached_data.get("timestamp", 0) < 14400:
                        logger.info(f"Using cached technicals for {ticker}")
                        self._tech_cache[ticker] = cached_data.get("indicators", {})
                        return self._tech_cache[ticker]
            except Exception as e:
                logger.warning(f"Failed to read technical cache for {ticker}: {e}")

        indicators = {}
        
        # PRIMARY: LOCAL CALCULATION VIA YFINANCE + PANDAS_TA
        try:
            yt = yf.Ticker(ticker)
            hist = yt.history(period="1y")
            if not hist.empty:
                # pandas_ta calculations
                hist.ta.rsi(length=14, append=True)
                hist.ta.sma(length=50, append=True)
                hist.ta.sma(length=200, append=True)
                hist.ta.bbands(length=20, std=2, append=True)
                hist.ta.macd(fast=12, slow=26, signal=9, append=True)
                
                # Mapping pandas_ta names to our standard internal schema
                # pandas_ta column names vary, usually like RSI_14, SMA_50, SMA_200, BBU_20_2.0, BBL_20_2.0, MACD_12_26_9, MACDs_12_26_9
                last_row = hist.iloc[-1]
                indicators['RSI_14'] = float(last_row.get('RSI_14', 0))
                indicators['SMA_50'] = float(last_row.get('SMA_50', 0))
                indicators['SMA_200'] = float(last_row.get('SMA_200', 0))
                indicators['BB_Upper'] = float(last_row.get('BBU_20_2.0', 0))
                indicators['BB_Lower'] = float(last_row.get('BBL_20_2.0', 0))
                indicators['MACD'] = float(last_row.get('MACD_12_26_9', 0))
                indicators['MACD_Signal'] = float(last_row.get('MACDs_12_26_9', 0))
                
                logger.info(f"Calculated local technicals for {ticker} using pandas_ta")
                
                # Disk Cache
                try:
                    with open(cache_file, 'w') as f:
                        json.dump({"timestamp": time.time(), "indicators": indicators}, f)
                except Exception as ce:
                    logger.warning(f"Failed to write technical cache for {ticker}: {ce}")
                
                self._tech_cache[ticker] = indicators
                return indicators
        except Exception as ex:
            logger.error(f"yfinance/pandas_ta local calculation failed for {ticker}: {ex}")

        # SECONDARY: ALPHA VANTAGE (FALLBACK)
        if self.ti:
            try:
                logger.info(f"Falling back to Alpha Vantage for {ticker} technicals")
                rsi, _ = self.ti.get_rsi(symbol=ticker, interval='daily', time_period=14)
                indicators['RSI_14'] = float(rsi.iloc[-1]['RSI'])
                sma, _ = self.ti.get_sma(symbol=ticker, interval='daily', time_period=50)
                indicators['SMA_50'] = float(sma.iloc[-1]['SMA'])
                sma200, _ = self.ti.get_sma(symbol=ticker, interval='daily', time_period=200)
                indicators['SMA_200'] = float(sma200.iloc[-1]['SMA'])
                bbands, _ = self.ti.get_bbands(symbol=ticker, interval='daily', time_period=20)
                indicators['BB_Upper'] = float(bbands.iloc[-1]['Real Upper Band'])
                indicators['BB_Lower'] = float(bbands.iloc[-1]['Real Lower Band'])
                
                self._tech_cache[ticker] = indicators
                return indicators
            except Exception as e:
                logger.error(f"Alpha Vantage fallback also failed for {ticker}: {e}")

        return indicators

class IngestionOrchestrator:
    """Orchestrates Phase 2 data gathering."""
    def __init__(self):
        self.macro = MacroProvider()
        self.fundamentals = FundamentalProvider()
        self.sec = SECProvider()

    def get_market_context(self, tickers: List[str]) -> MarketContext:
        # Expand tickers to include competitors for better context/pairs trades
        expanded_tickers = list(tickers)
        for t in tickers:
            comps = self.fundamentals.get_competitors(t)
            for c in comps:
                if c not in expanded_tickers:
                    expanded_tickers.append(c)
        
        # Limit expansion to prevent API bloat (e.g. max 5 total)
        expanded_tickers = expanded_tickers[:5]
        
        logger.info(f"Ingesting market context for: {expanded_tickers}")
        
        # 1. Macro
        macro_snap = self.macro.get_macro_snapshot()
        
        # 2. Fundamentals, Technicals, and SEC
        fundamental_snaps = {}
        technical_data = {}
        sec_summaries = {}
        
        for ticker in expanded_tickers:
            fundamental_snaps[ticker] = self.fundamentals.get_ticker_fundamentals(ticker)
            technical_data[ticker] = self.fundamentals.get_technical_indicators(ticker)
            sec_summaries[ticker] = self.sec.get_latest_filing_summary(ticker)
            
        return MarketContext(
            macro_data=macro_snap,
            fundamentals=fundamental_snaps,
            technical_indicators=technical_data,
            filing_summaries=sec_summaries
        )
