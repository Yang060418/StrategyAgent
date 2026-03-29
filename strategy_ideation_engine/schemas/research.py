from enum import Enum
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field, HttpUrl

class SourceCredibility(str, Enum):
    ACADEMIC_PAPER = "ACADEMIC_PAPER"       # ArXiv
    FINANCIAL_NEWS = "FINANCIAL_NEWS"       # Bloomberg, Reuters via Tavily
    INSTITUTIONAL_REPORT = "INSTITUTIONAL_REPORT" # SEC Filings, Bank Research
    SOCIAL_SENTIMENT = "SOCIAL_SENTIMENT"   # Twitter/Reddit (if added later)

class ResearchInsight(BaseModel):
    """A synthesized insight extracted from a single source."""
    title: str = Field(..., description="Title of the paper or article")
    source_url: Optional[HttpUrl] = Field(None, description="Link to original source")
    credibility: SourceCredibility = Field(..., description="Classification of the source")
    authors_or_publisher: Optional[str] = None
    published_date: Optional[datetime] = None
    key_findings: List[str] = Field(
        ..., 
        description="Concise, bulleted findings extracted by the LLM from the raw text"
    )

class LiteratureContext(BaseModel):
    """
    Phase 3: The complete literature review payload.
    Passed to the hypothesis generator as academic/news backing.
    """
    query_topic: str = Field(..., description="The subject being researched")
    insights: List[ResearchInsight] = Field(default_factory=list)
    synthesis_summary: Optional[str] = Field(
        None, 
        description="A brief summary of the combined insights"
    )
