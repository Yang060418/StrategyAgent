import dspy
import re
from typing import Tuple, Any, Optional
from tenacity import retry, stop_after_attempt, wait_exponential

from strategy_ideation_engine.intelligence.signatures.adversarial import AdversarialCritic
from strategy_ideation_engine.schemas.hypothesis import TradingHypothesis
from strategy_ideation_engine.schemas.market_data import MarketContext
from strategy_ideation_engine.config import logger

# Reuse logic or define local retry for simplicity in this module
provider_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=5, max=30),
    before_sleep=lambda retry_state: logger.warning(
        f"Retrying Adversarial Critic call (attempt {retry_state.attempt_number}) due to: {retry_state.outcome.exception()}"
    )
)

class AdversarialFilterModule(dspy.Module):
    """
    Phase 5: The 'Internal Critic' of the Engine.
    Uses Chain-of-Thought to analyze a hypothesis for logical and market-realism flaws.
    """
    def __init__(self, lm: Optional[dspy.LM] = None):
        super().__init__()
        self.lm = lm
        self.critic = dspy.ChainOfThought(AdversarialCritic)
        # Force the critic to use the provided LM
        if self.lm:
            self.critic.lm = self.lm

    def _parse_subscore(self, val: Any) -> float:
        try:
            match = re.search(r"([01]\.?\d*)", str(val))
            return float(match.group(1)) if match else 0.5
        except: return 0.5

    def _to_bullet_string(self, items: Any) -> str:
        if isinstance(items, list):
            return "\n".join([f"- {i}" for i in items])
        return str(items)

    @provider_retry
    def forward(self, hypothesis: TradingHypothesis, market_context: MarketContext, backtest_results: str = "None", previous_hypothesis: str = "None", previous_feedback: str = "None") -> Tuple[float, str]:
        """
        Evaluates a hypothesis and returns (adversarial_score, adversarial_feedback).
        """
        logger.info(f"Phase 5: Critiquing hypothesis '{hypothesis.strategy_name}'")
        
        market_str = f"Macro: {market_context.macro_data.model_dump(exclude={'last_updated'})}\nFundamentals: {market_context.fundamentals}"
        
        # Invoke the critic with actual memory and backtest data
        with dspy.settings.context(lm=self.lm or dspy.settings.lm):
            response = self.critic(
                current_hypothesis=hypothesis.model_dump_json(),
                market_context=market_str,
                backtest_results=backtest_results,
                previous_hypothesis=previous_hypothesis,
                previous_critique=previous_feedback
            )
        
        # Access structured output
        out = response.output
        
        s_logic = out.logic_score
        s_exec = out.execution_score
        s_math = out.math_score
        s_refine = out.refinement_score
        
        # Weighted Final Score
        # 40% Logic, 30% Execution, 20% Risk Math, 10% Refinement
        final_score = (s_logic * 0.4) + (s_exec * 0.3) + (s_math * 0.2) + (s_refine * 0.1)
            
        feedback = (
            f"SCORECARD: Logic={s_logic}, Exec={s_exec}, Math={s_math}, Refine={s_refine}\n\n"
            f"ANALYSIS:\n{self._to_bullet_string(out.analysis)}\n\n"
            f"COUNTER-FACTS:\n{self._to_bullet_string(out.counter_facts)}\n\n"
            f"FIXES:\n{self._to_bullet_string(out.fixes)}"
        )
        
        logger.info(f"Adversarial Feedback for '{hypothesis.strategy_name}':\n{feedback}")
        
        return float(final_score), feedback
