from strategy_ideation_engine.config.logger import logger

import re
import tiktoken
from typing import Dict, List, Any, Optional, Set
from strategy_ideation_engine.config.logger import logger

import dspy
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

class GroqLM(dspy.LM):
    """
    Custom dspy.LM wrapper for Groq with Tenacity-based 429 handling.
    Enforces exponential backoff to respect TPM/TPD limits.
    """
    def __init__(self, model: str, api_key: str, **kwargs):
        base_url = "https://api.groq.com/openai/v1"
        super().__init__(model=model, api_key=api_key, base_url=base_url, **kwargs)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=20, max=120),
        retry=retry_if_exception_type(Exception), # Still catch all to see details, but we will improve logging
        before_sleep=lambda retry_state: logger.warning(
            f"Groq API Error (Attempt {retry_state.attempt_number}): {retry_state.outcome.exception()}. Retrying..."
        )
    )
    def __call__(self, *args, **kwargs):
        try:
            return super().__call__(*args, **kwargs)
        except Exception as e:
            # Log the specific error before re-raising for tenacity
            logger.error(f"GroqLM Call Failed: {str(e)}")
            raise e

def setup_lm(model_name: str) -> dspy.LM:
    """
    Helper to initialize LM with correct provider settings.
    Supports prefixes like groq/, openai/, anthropic/, google/, and gemini/.
    """
    from strategy_ideation_engine.config import settings
    
    # Determine provider and actual model string
    if "/" in model_name:
        parts = model_name.split("/", 1)
        provider = parts[0].lower()
        
        # 1. OPENAI
        if provider == "openai":
            if not settings.OPENAI_API_KEY:
                logger.warning("OPENAI_API_KEY not found in settings. Ensure it is set in .env")
            return dspy.LM(model=model_name, api_key=settings.OPENAI_API_KEY, max_tokens=settings.DSPY_MAX_TOKENS)
        
        # 2. ANTHROPIC
        elif provider == "anthropic":
            if not settings.ANTHROPIC_API_KEY:
                logger.warning("ANTHROPIC_API_KEY not found in settings. Ensure it is set in .env")
            return dspy.LM(model=model_name, api_key=settings.ANTHROPIC_API_KEY, max_tokens=settings.DSPY_MAX_TOKENS)

        # 3. GOOGLE / GEMINI
        elif provider in ["google", "gemini"]:
             if not settings.GEMINI_API_KEY:
                logger.warning("GEMINI_API_KEY not found in settings. Ensure it is set in .env")
             return dspy.LM(model=model_name, api_key=settings.GEMINI_API_KEY, max_tokens=settings.DSPY_MAX_TOKENS)
            
        # 4. GROQ
        elif provider == "groq":
            if not settings.GROQ_API_KEY:
                logger.warning("GROQ_API_KEY not found in settings. Ensure it is set in .env")
            return GroqLM(model=model_name, api_key=settings.GROQ_API_KEY, max_tokens=settings.DSPY_MAX_TOKENS)
            
        # 5. OTHERS (Default to dspy.LM)
        else:
            return dspy.LM(model=model_name, max_tokens=settings.DSPY_MAX_TOKENS)
    else:
        # No provider specified, default to Groq for known models or legacy reasons
        # Check if it's a known model name that should be on Groq
        is_known_groq = model_name in settings.MODEL_LIMITS or \
                         "llama" in model_name.lower() or \
                         "qwen" in model_name.lower() or \
                         "mixtral" in model_name.lower()
                         
        if is_known_groq:
             full_model_str = f"groq/{model_name}"
             if not settings.GROQ_API_KEY:
                logger.warning("GROQ_API_KEY not found in settings. Ensure it is set in .env")
             return GroqLM(model=full_model_str, api_key=settings.GROQ_API_KEY, max_tokens=settings.DSPY_MAX_TOKENS)
        
        return dspy.LM(model=model_name, max_tokens=settings.DSPY_MAX_TOKENS)

class GroundingValidator:
    """
    Prevents 'Density Hacking' by ensuring numerical claims in the rationale 
    actually exist in the source fact sheet.
    """
    @staticmethod
    def extract_numbers(text: str) -> Set[float]:
        """ Extracts and normalizes all numbers from a block of text. """
        if not text:
            return set()
            
        # Pattern for integers, decimals, and percentages
        # Normalizes percentages: '73%' -> 0.73
        # Normalizes multiples: '25x' -> 25.0
        # Normalizes dollars: '$150' -> 150.0
        pattern = r"(\d+\.?\d*)\s*(%|x|X)?|(?:\$)(\d+\.?\d*)"
        matches = re.findall(pattern, text)
        
        numbers = set()
        for m in matches:
            val_str = m[0] or m[2]
            suffix = m[1].lower() if m[1] else ""
            
            try:
                val = float(val_str)
                if suffix == "%":
                    val = val / 100.0
                numbers.add(round(val, 4))
            except ValueError:
                continue
        return numbers

    @classmethod
    def calculate_grounding_score(cls, fact_sheet: str, rationale: str) -> float:
        """
        Returns a score (0.0 to 1.0) representing the ratio of verified numbers.
        Verified = Number in rationale exists in fact_sheet.
        """
        source_facts = cls.extract_numbers(fact_sheet)
        claims = cls.extract_numbers(rationale)
        
        if not claims:
            return 0.0 # No numbers provided, fails density check
            
        verified = claims.intersection(source_facts)
        score = len(verified) / len(claims)
        
        logger.info(f"Grounding Check: Found {len(claims)} claims, {len(verified)} verified. Score: {score:.2f}")
        return score

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
