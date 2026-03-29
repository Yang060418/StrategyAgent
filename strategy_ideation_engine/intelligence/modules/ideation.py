import dspy
import re
import string
from typing import List, Optional, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from strategy_ideation_engine.intelligence.signatures.ideation import (
    ResearchSummarizer, DataSynthesizer, ThesisArchitect, TradeArchitect, IndicatorCoder
)
from strategy_ideation_engine.schemas.hypothesis import (
    TradingHypothesis, StrategyType, StrategyArchetype, TradeLeg, LegType, RiskMetrics
)
from strategy_ideation_engine.schemas.market_data import MarketContext
from strategy_ideation_engine.schemas.research import LiteratureContext
from strategy_ideation_engine.schemas.event import MarketEvent
from strategy_ideation_engine.schemas.state import StrategyScratchpad
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
    Stateful Four-Stage Brain for complex strategy formulation.
    Upgraded with ASCII sanitization and Multi-Line Re-Stitching for formatting resilience.
    """
    def __init__(self, lm: Optional[dspy.LM] = None):
        super().__init__()
        self.lm = lm
        self.summarizer = dspy.ChainOfThought(ResearchSummarizer)
        self.synthesizer = dspy.ChainOfThought(DataSynthesizer)
        self.thesis_architect = dspy.ChainOfThought(ThesisArchitect)
        self.trade_architect = dspy.ChainOfThought(TradeArchitect)
        self.indicator_coder = dspy.Predict(IndicatorCoder)
        
        # Force sub-modules to use the provided LM
        if self.lm:
            self.summarizer.lm = self.lm
            self.synthesizer.lm = self.lm
            self.thesis_architect.lm = self.lm
            self.trade_architect.lm = self.lm
            self.indicator_coder.lm = self.lm

    def _sanitize_string(self, text: str) -> str:
        """Strips non-ASCII characters and standardizes punctuation."""
        if not text: return ""
        printable = set(string.printable)
        text = ''.join(filter(lambda x: x in printable, text))
        text = text.replace('鈥', '-').replace('鈥', '-').replace('鈥', "'").replace('鈥', "'")
        return text

    @provider_retry
    def summarize_literature(self, literature: LiteratureContext) -> str:
        with dspy.settings.context(lm=self.lm or dspy.settings.lm):
            raw_text = "\n".join([f"- {i.title}: {'; '.join(i.key_findings)}" for i in literature.insights])
            summary_out = self.summarizer(
                raw_insights=raw_text, 
                query_topic=literature.query_topic,
                config={"max_tokens": 1024}
            )
            return self._sanitize_string(summary_out.summarized_findings)

    @provider_retry
    def synthesize_facts(self, market_str: str, research_summary: str) -> Dict[str, str]:
        with dspy.settings.context(lm=self.lm or dspy.settings.lm):
            fact_out = self.synthesizer(
                market_context=market_str, 
                research_summary=research_summary,
                config={"max_tokens": 2048}
            )
            return {
                "fact_sheet": self._sanitize_string(fact_out.numerical_fact_sheet)
            }

    @provider_retry
    def architect_thesis(self, event: MarketEvent, market_str: str, research_summary: str, fact_sheet: str, refinement_feedback: str = "") -> Dict[str, Any]:
        with dspy.settings.context(lm=self.lm or dspy.settings.lm):
            thesis_out = self.thesis_architect(
                event=f"{event.source}: {event.raw_payload}",
                market_context=market_str,
                research_summary=research_summary,
                fact_sheet=fact_sheet,
                refinement_feedback=refinement_feedback,
                config={"max_tokens": settings.DSPY_MAX_TOKENS} 
            )
            
            # Access structured output directly
            out = thesis_out.output
            
            data = {
                "strategy_name": out.strategy_name,
                "strategy_archetype": out.strategy_archetype,
                "strategy_type": out.strategy_type,
                "horizon": str(out.horizon),
                "economic_rationale": out.economic_rationale,
                "catalysts": out.catalysts,
                "risks": out.risks
            }
            return {k: self._sanitize_string(v) if isinstance(v, str) else v for k, v in data.items()}

    @provider_retry
    def architect_trade(self, thesis_data: Dict[str, Any], market_str: str, refinement_feedback: str = "None") -> TradingHypothesis:
        with dspy.settings.context(lm=self.lm or dspy.settings.lm):
            trade_out = self.trade_architect(
                strategy_name=thesis_data["strategy_name"],
                strategy_type=thesis_data["strategy_type"],
                strategy_archetype=thesis_data.get("strategy_archetype", "TACTICAL"),
                economic_rationale=thesis_data["economic_rationale"],
                market_context=market_str,
                refinement_feedback=refinement_feedback,
                config={"max_tokens": 4096}
            )
            return self._assemble_hypothesis(thesis_data, trade_out)

    def _assemble_hypothesis(self, thesis_data: Dict[str, Any], trade: dspy.Prediction) -> TradingHypothesis:
        """Assembles the structured output into a TradingHypothesis."""
        out = trade.output
        
        # 1. Map legs directly from structured output
        legs = []
        for leg_out in out.legs:
            legs.append(TradeLeg(
                ticker=leg_out.ticker.upper(),
                leg_type=leg_out.leg_type,
                relative_weight=leg_out.relative_weight,
                entry_condition=leg_out.entry_condition,
                exit_condition=leg_out.exit_condition
            ))

        if not legs:
            raise ValueError("CRITICAL: No legs provided in structured output.")

        # 2. Rationale Quality Check
        rationale = thesis_data.get("economic_rationale", "")
        numbers_found = re.findall(r"\d+%?|\d+\.\d+%?", rationale)
        if len(numbers_found) < 2:
            logger.warning(f"Low numerical density ({len(numbers_found)}) in rationale.")

        return TradingHypothesis(
            strategy_name=thesis_data["strategy_name"],
            strategy_archetype=thesis_data["strategy_archetype"],
            strategy_type=thesis_data["strategy_type"],
            economic_rationale=rationale,
            catalysts=thesis_data.get("catalysts", []),
            legs=legs,
            time_horizon_days=out.time_horizon_days,
            invalidation_criteria=out.invalidation_criteria,
            risk_metrics=RiskMetrics(max_drawdown_tolerance=out.max_drawdown_pct),
            adversarial_score=0.0
        )
