import dspy
from typing import List
from pydantic import BaseModel, Field

class AdversarialOutput(BaseModel):
    analysis: List[str] = Field(..., description="Bullet points identifying flaws and logical leaps.")
    counter_facts: List[str] = Field(..., description="Data points or trends that contradict the hypothesis.")
    
    logic_score: float = Field(..., description="Score 0.0-1.0: Strength of economic reasoning.")
    execution_score: float = Field(..., description="Score 0.0-1.0: Specificity of trade rules.")
    math_score: float = Field(..., description="Score 0.0-1.0: Mathematical validity.")
    refinement_score: float = Field(..., description="Score 0.0-1.0: How well did they fix previous complaints?")
    
    fixes: List[str] = Field(..., description="How to make the strategy more robust.")

class AdversarialCritic(dspy.Signature):
    """
    Act as a cynical but REALISTIC senior quantitative researcher. 
    Critique a trading hypothesis for logical fallacies and execution risks.

    CRITICAL EVALUATION RULES:
    1. ARCHETYPE-LEG CONTRADICTION: If the strategy is 'MEAN_REVERSION' but goes Long on an overbought asset (RSI > 70) or Short on an oversold asset (RSI < 30), REJECT immediately with a score < 0.3.
    2. SECTOR LOGIC: Check for macro contradictions (e.g., Long Airlines during an Oil Spike).
    3. BACKTEST GROUNDING: Analyze the 'backtest_results'. If win rate < 45% or Sharpe < 0.2, penalize Execution and Math scores heavily.
    4. MEMORY INTEGRITY: Compare the 'current_hypothesis' against the 'previous_hypothesis'. If they didn't implement your 'previous_critique', penalize heavily.
    5. NO HALLUCINATION: Do NOT assume they did something (like a straddle) if it's not in the current_hypothesis JSON.
    6. INSTRUMENT CONSTRAINT: Do NOT demand data or models that are not provided in the context.
    7. HORIZON-SPECIFIC REASONING: For < 30 day trades, stop penalizing for long-term fundamentals like YoY Revenue.
    """
    current_hypothesis = dspy.InputField(desc="The proposed trading strategy in JSON format.")
    market_context = dspy.InputField(desc="Current market conditions.")
    backtest_results = dspy.InputField(desc="Historical performance metrics from Phase 5b.", default="None")
    previous_hypothesis = dspy.InputField(desc="The strategy from the previous attempt (if any).", default="None")
    previous_critique = dspy.InputField(desc="Your previous feedback (if any).", default="None")
    
    output: AdversarialOutput = dspy.OutputField(desc="Structured adversarial critique.")
