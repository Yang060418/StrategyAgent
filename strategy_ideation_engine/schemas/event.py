from enum import Enum
from typing import Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field

class EventSource(str, Enum):
    MACRO_CALENDAR = "MACRO_CALENDAR"
    EARNINGS_RELEASE = "EARNINGS_RELEASE"
    NEWS_ALERT = "NEWS_ALERT"
    PRICE_SHOCK = "PRICE_SHOCK"
    SYSTEMATIC_RUN = "SYSTEMATIC_RUN"

class MarketEvent(BaseModel):
    """
    Phase 1: Defines the trigger that wakes up the Ideation Engine.
    """
    event_id: str = Field(..., description="Unique identifier for the event")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the event occurred")
    source: EventSource = Field(..., description="The category/source of the event")
    affected_tickers: Optional[List[str]] = Field(default_factory=list, description="Tickers directly implicated by the event")
    raw_payload: Dict = Field(default_factory=dict, description="Raw data from the event source (e.g., news headline, macro print)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "event_id": "evt_12345",
                "timestamp": "2023-10-27T14:30:00Z",
                "source": "MACRO_CALENDAR",
                "affected_tickers": ["SPY", "QQQ"],
                "raw_payload": {"indicator": "CPI", "actual": 3.2, "expected": 3.3}
            }
        }
