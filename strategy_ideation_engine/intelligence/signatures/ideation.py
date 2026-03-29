import dspy
from typing import List, Optional
from pydantic import BaseModel, Field
from strategy_ideation_engine.schemas.hypothesis import StrategyType, StrategyArchetype, LegType

class ResearchSummarizer(dspy.Signature):
    """
    Phase 3: The 'Research Analyst'.
    Condense raw financial research into high-signal insights.
    
    CRITICAL RULES:
    1. DISTILLATION: Be extremely dense. Focus ONLY on catalysts, quantitative anomalies, and risks.
    2. NO FLUFF: Do NOT use introductory phrases. Start directly with the signal.
    3. REPETITION: Do NOT repeat the same insight for different tickers.
    """
    raw_insights = dspy.InputField(desc="List of ResearchInsight objects.")
    query_topic = dspy.InputField(desc="The specific topic or ticker.")
    
    summarized_findings = dspy.OutputField(desc="Bulleted summary of relevant trading signals.")

class DataSynthesizer(dspy.Signature):
    """
    Phase 4.0: The 'Data Miner'.
    Extract numerical facts and specific data points from raw context.
    Your output becomes the 'Ground Truth' for the strategy architects.
    
    CRITICAL RULES:
    1. NEGATIVE EARNINGS: If a ticker has negative trailing_eps, explicitly flag P/E as 'INVALID (Negative EPS)' and prioritize 'EV/EBITDA' or 'Price/Book' in your summary.
    2. CLEAN NUMBERS: Provide exact values from the context. No rounding.
    3. NO PROSE: Only bulleted numerical facts.
    """
    market_context = dspy.InputField(desc="Macro, fundamental, technical, and SEC data.")
    research_summary = dspy.InputField(desc="Condensed insights from ArXiv and news.")
    
    numerical_fact_sheet = dspy.OutputField(desc="A bulleted list of verified numerical facts (e.g., 'MSFT RSI_14 is 28', 'NVDA EV/EBITDA is 25x').")

class ThesisOutput(BaseModel):
    strategy_name: str = Field(..., description="Clear and descriptive name.")
    strategy_archetype: StrategyArchetype = Field(..., description="The overarching strategy style.")
    strategy_type: StrategyType = Field(..., description="The classification of the strategy architecture.")
    horizon: int = Field(..., description="Intended holding period in days.")
    economic_rationale: str = Field(..., description="The 'Why' backed by NUMERICAL evidence. MUST BE A PLAIN STRING, NO NESTED OBJECTS.")
    catalysts: List[str] = Field(..., description="Upcoming events or macro conditions. List of strings.")
    risks: List[str] = Field(..., description="Structural or macro risks. List of strings.")

class ThesisArchitect(dspy.Signature):
    """
    Phase 4a: The 'Economic Brain'. 
    Generate a RIGOROUS economic and behavioral thesis for a trading strategy.
    
    CRITICAL RULES:
    1. NUMERICAL INTERPRETATION: Growth metrics are ratios. E.g., '0.73' means 73% growth.
    2. NO NESTING: Every field in your JSON output must be a simple string, integer, or list of strings. Do NOT use objects/dictionaries for rationale or risks.
    3. NUMERICAL MANDATE: Every economic claim MUST be supported by at least TWO specific metrics from the provided Fact Sheet.
    4. QUALITY FILTER: If the fact sheet shows conflicting data, explain the discrepancy as part of the risk.
    """

    event = dspy.InputField(desc="The triggering market event.")
    market_context = dspy.InputField(desc="Macro, fundamental, technical, and SEC data.")
    research_summary = dspy.InputField(desc="Condensed insights from ArXiv and news.")
    fact_sheet = dspy.InputField(desc="Numerical fact sheet and identified discrepancies from Step 4.0.")
    refinement_feedback = dspy.InputField(desc="CRITICAL: Feedback from a rejected prior attempt.", default="")
    
    output: ThesisOutput = dspy.OutputField(desc="Structured economic thesis.")

class IndicatorCoder(dspy.Signature):
    """
    Phase 5c: The 'Quant Programmer'.
    Write a single line of Python/Pandas code to calculate a technical indicator.
    Input is a dataframe 'df' with standard yfinance columns: [Open, High, Low, Close, Volume].
    
    CRITICAL RULES:
    1. PERIOD MANDATE: If the indicator name doesn't specify a period (e.g., just 'EMA'), default to 20 or 50 as appropriate.
    2. PURE PANDAS/NUMPY: Use ONLY 'pd' and 'np'. No external libraries.
    3. MODERN PANDAS: Use 'clip(lower=0)' instead of deprecated methods.
    4. OUTPUT ONLY CODE: No preambles.
    """
    indicator_name = dspy.InputField(desc="The name of the indicator (e.g., 'EMA_20', 'SMA_50').")
    context = dspy.InputField(desc="The strategy context.")
    
    python_code = dspy.OutputField(desc="A single valid line of Python code.")

class TradeLegOutput(BaseModel):
    ticker: str = Field(..., description="The asset symbol")
    leg_type: LegType = Field(..., description="The instrument and direction")
    relative_weight: float = Field(..., description="Weight of this leg")
    entry_condition: str = Field(..., description="Technical or fundamental trigger (e.g., 'RSI_14 < 30')")
    exit_condition: str = Field(..., description="Condition to close the leg")

class TradeArchitectOutput(BaseModel):
    legs: List[TradeLegOutput] = Field(..., description="Trade components.")
    time_horizon_days: int = Field(..., description="Integer holding period.")
    invalidation_criteria: List[str] = Field(..., description="Conditions that prove the thesis wrong.")
    max_drawdown_pct: float = Field(..., description="Max drawdown target.")

class TradeArchitect(dspy.Signature):
    """
    Phase 4b: The 'Technical Architect'.
    Translate an approved economic thesis into concrete trade legs.
    
    CRITICAL RULES:
    1. LOGIC INTEGRITY: Your trade legs MUST mathematically fulfill the logic of the thesis.
    2. NUMERICAL GROUNDING: Entry conditions MUST use specific indicator levels (e.g., 'RSI_14 < 30', 'SMA_50 > SMA_200') found in the Fact Sheet.
    3. MEAN_REVERSION: 
       - FORBIDDEN: Do NOT go LONG on an asset with RSI > 70.
       - FORBIDDEN: Do NOT go SHORT on an asset with RSI < 30.
    4. INDICATOR MAPPING: Use standard indicators: RSI_14, SMA_50, SMA_200, BB_Upper, BB_Lower, MACD.
    """
    strategy_name = dspy.InputField()
    strategy_type = dspy.InputField()
    strategy_archetype = dspy.InputField(desc="MEAN_REVERSION, TACTICAL, or STRUCTURAL.")
    economic_rationale = dspy.InputField()
    market_context = dspy.InputField()
    refinement_feedback = dspy.InputField(desc="Critical fixes required for the trade structure.", default="None")
    
    output: TradeArchitectOutput = dspy.OutputField(desc="Structured trade architecture.")
