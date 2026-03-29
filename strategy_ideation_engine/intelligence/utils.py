from strategy_ideation_engine.config.logger import logger

import tiktoken
from typing import Dict, List, Any, Optional
from strategy_ideation_engine.config.logger import logger

class TokenBudgeter:
    """
    Solves the 'Distillation Paradox' by dynamically scaling context 
    based on model-specific TPM/TPD limits.
    """
    def __init__(self, model_name: str, tpm_limit: int = 6000, reserve_tokens: int = 1500):
        self.model_name = model_name
        self.tpm_limit = tpm_limit
        self.reserve_tokens = reserve_tokens
        self.available_budget = tpm_limit - reserve_tokens
        
        try:
            # tiktoken's cl100k_base is a good proxy for most modern models (Llama 3, Qwen, etc.)
            self.encoder = tiktoken.get_encoding("cl100k_base")
        except:
            # Fallback if cl100k_base is unavailable
            self.encoder = tiktoken.get_encoding("gpt2")

    def count_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text))

    def budget_items(self, items: Dict[str, str], priority_name: str) -> str:
        """
        Packs as many items as possible from a dictionary into the available budget.
        
        Args:
            items: Dictionary of {label: content}
            priority_name: Name of the section for logging
            
        Returns:
            A formatted string of budgeted items.
        """
        output = []
        current_tokens = 0
        total_items = len(items)
        packed_count = 0

        for label, content in items.items():
            entry = f"[{label}]: {content}\n"
            entry_tokens = self.count_tokens(entry)
            
            if current_tokens + entry_tokens > self.available_budget:
                logger.warning(f"Budgeter: {priority_name} truncated. Packed {packed_count}/{total_items} items.")
                break
                
            output.append(entry)
            current_tokens += entry_tokens
            self.available_budget -= entry_tokens
            packed_count += 1
            
        return "".join(output)

    def budget_list(self, items: List[Any], priority_name: str) -> List[Any]:
        """
        Packs as many list items as possible into the available budget.
        """
        output = []
        packed_count = 0
        
        for item in items:
            item_str = str(item)
            entry_tokens = self.count_tokens(item_str)
            
            if self.available_budget - entry_tokens < 0:
                logger.warning(f"Budgeter: {priority_name} truncated. Packed {packed_count}/{len(items)} items.")
                break
                
            output.append(item)
            self.available_budget -= entry_tokens
            packed_count += 1
            
        return output

def apply_mercy_rule(current_score: float, previous_score: float, feedback: str, attempt: int) -> float:
    """
    Applies heuristic boosts to the adversarial score to break 'refinement traps'.
    
    Args:
        current_score: The raw score from the Critic.
        previous_score: The score from the previous attempt.
        feedback: The feedback text from the Critic.
        attempt: The current attempt index (0-based).
        
    Returns:
        The (potentially) boosted score.
    """
    if attempt == 0:
        return current_score
        
    # NEW: HARD FLOOR. If the raw score is abysmal, no mercy.
    if current_score < 0.60:
        return current_score

    new_score = current_score
    
    # NEW: Reduced boosts to ensure quality
    if current_score > previous_score:
        logger.info(f"Mercy Rule: Improvement detected. Applying +0.10 structural boost.")
        new_score = min(0.95, current_score + 0.10)
    elif "FIXED" in feedback.upper() or "IMPROVED" in feedback.upper():
        logger.info(f"Mercy Rule: Critic acknowledges progress. Applying +0.05 boost.")
        new_score = min(0.95, current_score + 0.05)
        
    return new_score
