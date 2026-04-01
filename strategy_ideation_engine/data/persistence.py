import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from strategy_ideation_engine.config import settings, logger
from strategy_ideation_engine.schemas.hypothesis import TradingHypothesis

class HypothesisLedger:
    """
    Handles persistence of generated hypotheses to ensure state and auditability.
    Uses JSON-Line format for append-only efficiency.
    """
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path or settings.EXPORT_DIR) / "hypothesis_ledger.jsonl"
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, hypothesis: TradingHypothesis, event_id: str, inputs: Optional[dict] = None):
        """
        Saves a hypothesis with its triggering event metadata and optional inputs
        for DSPy compilation/training.
        """
        record = {
            "saved_at": datetime.utcnow().isoformat(),
            "trigger_event_id": event_id,
            "inputs": inputs or {},
            "hypothesis": hypothesis.model_dump()
        }
        
        try:
            with open(self.storage_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            logger.info(f"Hypothesis {hypothesis.hypothesis_id} saved to ledger.")
        except Exception as e:
            logger.error(f"Failed to save hypothesis to ledger: {e}")

    def load_all(self) -> List[dict]:
        """Loads all historical hypotheses for analysis or replay."""
        if not self.storage_path.exists():
            return []
        
        history = []
        with open(self.storage_path, "r", encoding="utf-8") as f:
            for line in f:
                history.append(json.loads(line))
        return history
