from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from strategy_ideation_engine.schemas.hypothesis import TradingHypothesis

class StrategyScratchpad(BaseModel):
    """
    Internal state for a single ideation cycle.
    Tracks the evolution of the strategy from raw data to validated hypothesis.
    """
    event_id: str
    
    # Phase 2 & 3: Raw Context
    market_context_str: str = ""
    research_summary_str: str = ""
    
    # Phase 4.0: Ground Truth
    fact_sheet: str = ""
    discrepancy_log: str = ""
    
    # Phase 4a: Economic Thesis
    strategy_name: str = ""
    strategy_type: str = ""
    economic_rationale: str = ""
    catalysts: List[str] = []
    key_risks: str = ""
    
    # Phase 5: Critiques
    critique_history: List[str] = Field(default_factory=list)
    
    # Phase 4b: Trade Structure
    current_hypothesis: Optional[TradingHypothesis] = None
    
    def add_critique(self, critique: str):
        self.critique_history.append(critique)
        
    def get_full_thesis(self) -> str:
        return (
            f"NAME: {self.strategy_name}\n"
            f"TYPE: {self.strategy_type}\n"
            f"RATIONALE: {self.economic_rationale}\n"
            f"CATALYSTS: {', '.join(self.catalysts)}\n"
            f"RISKS: {self.key_risks}"
        )
