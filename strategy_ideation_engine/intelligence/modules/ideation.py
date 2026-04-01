import dspy
import re
import string
from typing import List, Optional, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential

from strategy_ideation_engine.intelligence.signatures.ideation import (
    ResearchSummarizer, DataSynthesizer, ThesisArchitect, TradeArchitect, IndicatorCoder
)
from strategy_ideation_engine.schemas.hypothesis import (
    TradingHypothesis, StrategyType, StrategyArchetype, TradeLeg, LegType, RiskMetrics
)
from strategy_ideation_engine.intelligence.utils import GroundingValidator
from strategy_ideation_engine.config import logger, settings

# Retry decorator for API Rate Limits
provider_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=5, max=30),
    before_sleep=lambda retry_state: logger.warning(
        f"Retrying API call (attempt {retry_state.attempt_number}) due to: {retry_state.outcome.exception()}"
    )
)

class StrategyIdeationModule(dspy.Module):
    """
    Refactored Strategy Ideation Engine as a compilable DSPy Module.
    Uses dspy.Assert and dspy.Suggest to enforce numerical grounding and prevent 'Density Hacking'.
    """
    def __init__(self, lm: Optional[dspy.LM] = None):
        super().__init__()
        self.lm = lm
        
        # Initialize Signatures as TypedPredictors for structured output validation
        self.summarizer = dspy.TypedChainOfThought(ResearchSummarizer)
        self.synthesizer = dspy.TypedChainOfThought(DataSynthesizer)
        self.thesis_architect = dspy.TypedChainOfThought(ThesisArchitect)
        self.trade_architect = dspy.TypedChainOfThought(TradeArchitect)
        self.indicator_coder = dspy.TypedPredictor(IndicatorCoder)

    def _sanitize_string(self, text: str) -> str:
        """Strips non-ASCII characters and standardizes punctuation."""
        if not text: return ""
        printable = set(string.printable)
        text = ''.join(filter(lambda x: x in printable, text))
        return text

    @provider_retry
    def summarize_literature(self, literature) -> str:
        """Scout Task: Condense raw research."""
        # Convert list of Pydantic objects to a dense string for the LLM
        insights_str = "\n".join([
            f"TITLE: {i.title}\nFINDINGS: {'; '.join(i.key_findings)}" 
            for i in literature.insights
        ])
        
        with dspy.settings.context(lm=self.lm or dspy.settings.lm):
            prediction = self.summarizer(
                raw_insights=insights_str, 
                query_topic=literature.query_topic
            )
            return self._sanitize_string(prediction.summarized_findings)

    @provider_retry
    def synthesize_facts(self, market_context_str: str, research_summary_str: str) -> Dict[str, str]:
        """Scout Task: Extract numerical fact sheet."""
        with dspy.settings.context(lm=self.lm or dspy.settings.lm):
            prediction = self.synthesizer(
                market_context=market_context_str, 
                research_summary=research_summary_str
            )
            return {"fact_sheet": self._sanitize_string(prediction.numerical_fact_sheet)}

    def forward(self, event, market_context_str: str, research_summary: str, fact_sheet: str, refinement_feedback: str = "") -> TradingHypothesis:
        """
        The main optimization path. 
        Chains the Economic Thesis generation with Technical Trade Architecture.
        Includes DSPy Suggest for numerical grounding.
        """
        with dspy.settings.context(lm=self.lm or dspy.settings.lm):
            # 1. Architect the Economic Thesis
            thesis_prediction = self.thesis_architect(
                event=event,
                market_context=market_context_str,
                research_summary=research_summary,
                fact_sheet=fact_sheet,
                refinement_feedback=refinement_feedback
            )
            thesis = thesis_prediction.output

            # --- NUMERICAL GROUNDING ASSERTION ---
            # Prevents 'Density Hacking' by ensuring numbers exist in the Fact Sheet.
            grounding_score = GroundingValidator.calculate_grounding_score(
                fact_sheet, thesis.economic_rationale
            )
            
            dspy.Suggest(
                grounding_score >= 0.7,
                f"Your rationale cited numbers that were not in the Fact Sheet or had poor grounding (Score: {grounding_score:.2f}). "
                f"Rewrite the rationale using ONLY the EXACT verified metrics from the Fact Sheet: {fact_sheet}"
            )

            # 2. Architect the Technical Trade Structure (chained from Thesis)
            trade_prediction = self.trade_architect(
                strategy_name=thesis.strategy_name,
                strategy_type=thesis.strategy_type,
                strategy_archetype=thesis.strategy_archetype,
                economic_rationale=thesis.economic_rationale,
                market_context=market_context_str,
                refinement_feedback=refinement_feedback
            )
            trade = trade_prediction.output

            # 3. Assemble and Return the Pydantic-validated TradingHypothesis
            return TradingHypothesis(
                strategy_name=thesis.strategy_name,
                strategy_type=thesis.strategy_type,
                strategy_archetype=thesis.strategy_archetype,
                economic_rationale=thesis.economic_rationale,
                catalysts=thesis.catalysts,
                legs=[
                    TradeLeg(
                        ticker=leg.ticker.upper(),
                        leg_type=leg.leg_type,
                        relative_weight=leg.relative_weight,
                        entry_condition=leg.entry_condition,
                        exit_condition=leg.exit_condition
                    ) for leg in trade.legs
                ],
                time_horizon_days=trade.time_horizon_days,
                invalidation_criteria=trade.invalidation_criteria,
                risk_metrics=RiskMetrics(max_drawdown_tolerance=trade.max_drawdown_pct)
            )
