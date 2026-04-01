import dspy
from dspy.teleprompter import BootstrapFewShot
import json
import os
from typing import List, Optional
from strategy_ideation_engine.intelligence.modules.ideation import StrategyIdeationModule
from strategy_ideation_engine.data.persistence import HypothesisLedger
from strategy_ideation_engine.config import logger, settings
from strategy_ideation_engine.intelligence.utils import setup_lm

class EngineCompiler:
    """
    Phase 5: The 'Learning Layer'.
    Optimizes the Strategist and Critic prompts using historical successes.
    """
    def __init__(self, model_name: Optional[str] = None):
        self.ledger = HypothesisLedger()
        self.model_name = model_name or settings.COMPILER_MODEL
        self.teacher_lm = setup_lm(self.model_name)

    def _prepare_dataset(self, min_adversarial_score: float = 0.8) -> List[dspy.Example]:
        """
        Extracts high-quality (Input, Output) pairs from the ledger.
        """
        raw_data = self.ledger.load_all()
        dataset = []
        
        for record in raw_data:
            hypo = record.get("hypothesis", {})
            inputs = record.get("inputs", {})
            
            # Filter for 'Gold Standard' examples
            if hypo.get("adversarial_score", 0) >= min_adversarial_score and inputs:
                # Construct dspy.Example
                example = dspy.Example(
                    event=record.get("trigger_event_id"),
                    market_context_str=inputs.get("market_context_str"),
                    research_summary=inputs.get("research_summary"),
                    fact_sheet=inputs.get("fact_sheet"),
                    # The 'Target' is the final hypothesis
                    # Note: Depending on your forward() return type, you might need to wrap this
                    answer=hypo.get("economic_rationale") 
                ).with_inputs('event', 'market_context_str', 'research_summary', 'fact_sheet')
                
                dataset.append(example)
        
        logger.info(f"Compiler: Prepared {len(dataset)} Gold Standard examples.")
        return dataset

    def compile_strategist(self, trainset: List[dspy.Example]):
        """
        Uses BootstrapFewShot to optimize the Strategist module.
        """
        if not trainset:
            logger.warning("No training data found. Skipping compilation.")
            return

        # Define the 'Metric' for optimization
        # For bootstrapping, we just need the teacher to approve the trace
        teleprompter = BootstrapFewShot(
            metric=lambda gold, pred: True, # Simplest metric for bootstrapping
            max_bootstrapped_demos=3,
            max_labeled_demos=3,
            teacher_settings=dict(lm=self.teacher_lm)
        )
        
        # The module to optimize
        student = StrategyIdeationModule()
        
        logger.info("Compiler: Starting BootstrapFewShot optimization...")
        optimized_app = teleprompter.compile(student, trainset=trainset)
        
        # Save the optimized program
        save_path = os.path.join(settings.EXPORT_DIR, "optimized_strategist.json")
        optimized_app.save(save_path)
        logger.info(f"Compiler: Optimized Strategist saved to {save_path}")
        
        return optimized_app

if __name__ == "__main__":
    # Example usage
    compiler = EngineCompiler()
    gold_set = compiler._prepare_dataset()
    if gold_set:
        compiler.compile_strategist(gold_set)
