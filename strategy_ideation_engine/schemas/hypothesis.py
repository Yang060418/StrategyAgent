from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
import uuid

class LegType(str, Enum):
    LONG_EQUITY = "LONG_EQUITY"
    SHORT_EQUITY = "SHORT_EQUITY"
    CALL_OPTION = "CALL_OPTION"
    PUT_OPTION = "PUT_OPTION"
    CASH = "CASH"

class StrategyType(str, Enum):
    DIRECTIONAL = "DIRECTIONAL"
    PAIRS_TRADE = "PAIRS_TRADE"
    STATISTICAL_ARBITRAGE = "STATISTICAL_ARBITRAGE"
    MACRO_HEDGE = "MACRO_HEDGE"
    VOLATILITY_ARBITRAGE = "VOLATILITY_ARBITRAGE"
    FACTOR_TILT = "FACTOR_TILT"
    MEAN_REVERSION = "MEAN_REVERSION"

class StrategyArchetype(str, Enum):
    STRUCTURAL = "STRUCTURAL" # Long-term, fundamental, wider stops
    TACTICAL = "TACTICAL"     # Short-term, event-driven, tight stops
    MEAN_REVERSION = "MEAN_REVERSION" # Technical/statistical overshoot

class TradeLeg(BaseModel):
    """Defines a single component of a multi-leg strategy."""
    ticker: str = Field(..., description="The asset symbol")
    leg_type: LegType = Field(..., description="The instrument and direction")
    relative_weight: float = Field(
        ..., 
        description="Weight of this leg relative to the total strategy allocation (e.g., 1.0 for long, -1.0 for short pairs trade leg)"
    )
    entry_condition: Optional[str] = Field(
        None, 
        description="Specific technical or fundamental condition to execute this leg (e.g., 'RSI < 30')"
    )
    exit_condition: Optional[str] = Field(
        None, 
        description="Specific condition to close this leg (e.g., 'Target Price = $150')"
    )

class RiskMetrics(BaseModel):
    """Expected risk parameters for the strategy."""
    expected_volatility_annualized: Optional[float] = None
    max_drawdown_tolerance: float = Field(
        ..., 
        description="The maximum acceptable drawdown percentage before forced liquidation"
    )
    sharpe_ratio_target: Optional[float] = None

class TradingHypothesis(BaseModel):
    """
    Phases 4-6: The ultimate, rigorously structured output.
    Supports complex, multi-leg strategies.
    """
    hypothesis_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    strategy_name: str = Field(..., description="A clear, descriptive name for the strategy")
    strategy_type: StrategyType = Field(..., description="The classification of the strategy architecture")
    strategy_archetype: StrategyArchetype = Field(
        StrategyArchetype.TACTICAL, 
        description="The overarching strategy style (STRUCTURAL vs TACTICAL)"
    )
    
    # Core Rationale
    economic_rationale: str = Field(
        ..., 
        description="The underlying economic or behavioral thesis driving the strategy"
    )
    catalysts: List[str] = Field(
        ..., 
        description="Specific upcoming events or conditions that will trigger the alpha"
    )
    
    # Execution Architecture
    legs: List[TradeLeg] = Field(
        ..., 
        description="The components of the trade. E.g., a Pairs Trade needs a Long leg and a Short leg."
    )
    time_horizon_days: int = Field(..., description="Expected holding period in days")
    
    # Risk & Validation
    invalidation_criteria: List[str] = Field(
        ..., 
        description="Fundamental or macro data points that, if realized, prove the thesis wrong (e.g., 'Fed hikes rates instead of pausing')"
    )
    risk_metrics: RiskMetrics = Field(..., description="Pre-defined risk limits")
    
    # Gatekeeper Scoring
    adversarial_score: float = Field(
        0.0, 
        description="Confidence score (0.0 to 1.0) from Phase 5 (Adversarial Filter). Must be > 0.8 for handoff."
    )
    adversarial_feedback: Optional[str] = Field(
        None,
        description="Critique from the adversarial filter regarding look-ahead bias or logic flaws"
    )
    backtest_score: float = Field(
        0.0,
        description="Empirical score from Phase 5b (Historical Backtest)"
    )
    backtest_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured metrics from the historical backtest (e.g., win_rate, sharpe)"
    )
    validation_logs: Optional[str] = Field(
        None,
        description="Logs from technical validation and backtesting"
    )
    
    @field_validator("adversarial_score")
    @classmethod
    def check_score_range(cls, v: float) -> float:
        if v < 0.0 or v > 1.0:
            raise ValueError("Adversarial score must be between 0.0 and 1.0")
        return v
