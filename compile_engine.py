import dspy
from dspy.teleprompt import BootstrapFewShot
import json
import os
from typing import Any

from strategy_ideation_engine.intelligence.modules.ideation import StrategyIdeationModule
from strategy_ideation_engine.intelligence.utils import GroundingValidator, setup_lm
from strategy_ideation_engine.schemas.hypothesis import TradingHypothesis
from strategy_ideation_engine.config import logger, settings

def load_gold_standard_trainset():
    """
    Loads the Gold Standard traces generated via the CLI.
    Maps them into DSPy Example objects with designated inputs and labels.
    """
    ledger_path = os.path.join(settings.EXPORT_DIR, "hypothesis_ledger.jsonl")
    trainset = []
    
    if not os.path.exists(ledger_path):
        logger.error(f"Ledger not found at {ledger_path}")
        return trainset

    with open(ledger_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                # Filter strictly for the Gold Standard traces that contain 'market_event'
                if "market_event" in data and "hypothesis" in data:
                    hypo_dict = data["hypothesis"]
                    
                    # 1. Define the Inputs
                    kwargs = {
                        "event": data["market_event"],
                        "market_context_str": data["market_context"],
                        "research_summary": "", # Gold standards don't rely on a separate scout summary
                        "fact_sheet": data["fact_sheet"]
                    }
                    
                    # 2. Flatten the expected outputs (Labels) into the Example
                    # This allows DSPy to know exactly what a "perfect" answer looks like
                    for key, value in hypo_dict.items():
                        kwargs[key] = value
                        
                    # 3. Create Example and explicitly declare the input fields
                    ex = dspy.Example(**kwargs).with_inputs('event', 'market_context_str', 'research_summary', 'fact_sheet')
                    trainset.append(ex)
            except Exception as e:
                pass
                
    return trainset

def deterministic_metric(gold: dspy.Example, pred: Any) -> float:
    """
    100% Deterministic Custom Metric. Zero LLM calls.
    - Schema Validation (0.4)
    - Numerical Grounding (0.6)
    """
    score = 0.0
    
    # 1. Schema Validation (0.4)
    # Ensure the prediction perfectly matches the Pydantic TradingHypothesis model
    try:
        if isinstance(pred, TradingHypothesis):
            hypo = pred
            score += 0.4
        else:
            # If pred is somehow a dict or another structure, attempt validation
            hypo = TradingHypothesis.model_validate(pred)
            score += 0.4
    except Exception as e:
        return 0.0 # Fail fast if the structure is broken
        
    # 2. Numerical Grounding (0.6)
    # Ensure the economic rationale ONLY uses numbers provided in the fact sheet
    if not hasattr(hypo, 'economic_rationale') or not hypo.economic_rationale:
        return score
        
    grounding_score = GroundingValidator.calculate_grounding_score(
        gold.fact_sheet, hypo.economic_rationale
    )
    score += (grounding_score * 0.6)
    
    return score

class EngineCompiler:
    def __init__(self):
        self.trainset = load_gold_standard_trainset()
        
    def compile_strategist(self):
        logger.info(f"### COMPILING STRATEGIST (Model: {settings.STRATEGIST_MODEL}) ###")
        
        # Limit trainset size to 5 to avoid Groq Rate Limits (TPD/TPM)
        limited_trainset = self.trainset[:5]
        
        if len(limited_trainset) < 3:
            logger.warning(f"Insufficient Gold Standard examples found ({len(limited_trainset)}). Need at least 3.")
            return

        # 1. Setup the Student Model (No Teacher required since we have perfect GS labels)
        lm = setup_lm("llama-3.1-8b-instant")
        dspy.settings.configure(lm=lm)
        
        # 2. Configure Optimizer
        # We use BootstrapFewShot to find the best CoT traces without RandomSearch overhead
        teleprompter = BootstrapFewShot(
            metric=deterministic_metric,
            max_bootstrapped_demos=3,   # Number of examples to bake into the prompt
            max_labeled_demos=1         # Number of direct label examples to include
        )
        
        # 3. Instantiate base module
        strategist = StrategyIdeationModule()
        
        # 4. Compile
        logger.info("Starting DSPy compilation process. This will generate CoT paths for your Gold Standards...")
        optimized = teleprompter.compile(strategist, trainset=limited_trainset)
        
        # 5. Export for production engine.py
        save_path = os.path.join(settings.EXPORT_DIR, "optimized_strategist.json")
        optimized.save(save_path)
        logger.info(f"SUCCESS: Strategist optimization saved to {save_path}.")

if __name__ == "__main__":
    compiler = EngineCompiler()
    logger.info(f"Loaded {len(compiler.trainset)} Gold Standard traces for compilation.")
    compiler.compile_strategist()
