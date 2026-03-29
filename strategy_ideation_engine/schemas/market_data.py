from typing import Dict, Optional
from datetime import datetime
from pydantic import BaseModel, Field

class FundamentalSnapshot(BaseModel):
    """Snapshot of a single asset's fundamentals (SEC/YFinance)."""
    ticker: str
    sector: Optional[str] = Field(None, description="GICS Sector")
    industry: Optional[str] = Field(None, description="Specific Industry")
    pe_ratio: Optional[float] = None
    ev_to_ebitda: Optional[float] = Field(None, description="Enterprise Value to EBITDA")
    pb_ratio: Optional[float] = None
    debt_to_equity: Optional[float] = None
    revenue_growth_yoy: Optional[float] = None
    free_cash_flow: Optional[float] = None
    operating_margins: Optional[float] = None
    return_on_equity: Optional[float] = None
    beta: Optional[float] = None
    trailing_eps: Optional[float] = None
    next_earnings_date: Optional[str] = None
    last_updated: datetime = Field(default_factory=datetime.utcnow)

class MacroSnapshot(BaseModel):
    """Snapshot of macroeconomic indicators (FRED)."""
    fed_funds_rate: Optional[float] = None
    cpi_yoy_change: Optional[float] = None
    unemployment_rate: Optional[float] = None
    gdp_growth_qoq: Optional[float] = None
    last_updated: datetime = Field(default_factory=datetime.utcnow)

class MarketContext(BaseModel):
    """
    Phase 2: Master payload containing unified fundamental and macro data.
    """
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    macro_data: MacroSnapshot = Field(default_factory=MacroSnapshot)
    fundamentals: Dict[str, FundamentalSnapshot] = Field(
        default_factory=dict, 
        description="Keyed by ticker"
    )
    technical_indicators: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Keyed by ticker, then indicator (e.g., {'AAPL': {'RSI_14': 65.2}})"
    )
    filing_summaries: Dict[str, str] = Field(
        default_factory=dict,
        description="Keyed by ticker, summary of latest SEC filings"
    )
