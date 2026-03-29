import unittest
from typing import Any, Dict
import dspy
from strategy_ideation_engine.intelligence.modules.ideation import StrategyIdeationModule
from strategy_ideation_engine.schemas.hypothesis import StrategyType, LegType

class MockPrediction:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class TestParsing(unittest.TestCase):
    def setUp(self):
        self.module = StrategyIdeationModule()

    def test_assemble_hypothesis_standard(self):
        thesis_data = {
            "strategy_name": "Test Strategy",
            "strategy_type": "DIRECTIONAL",
            "economic_rationale": "The market is up 10% and RSI is 30.",
            "catalysts": ["Earnings", "Product Launch"]
        }
        trade_pred = MockPrediction(
            legs="AAPL: LONG_EQUITY (weight 1.0) [Entry: Price > 150] [Exit: Price < 140]\nMSFT: SHORT_EQUITY (weight -0.5) [Entry: RSI > 70] [Exit: RSI < 50]",
            days="14",
            exit="Target reached",
            risk="Max DD 5%"
        )
        
        hypo = self.module._assemble_hypothesis(thesis_data, trade_pred)
        
        self.assertEqual(hypo.strategy_name, "Test Strategy")
        self.assertEqual(hypo.strategy_type, StrategyType.DIRECTIONAL)
        self.assertEqual(len(hypo.legs), 2)
        self.assertEqual(hypo.legs[0].ticker, "AAPL")
        self.assertEqual(hypo.legs[0].leg_type, LegType.LONG_EQUITY)
        self.assertEqual(hypo.legs[1].ticker, "MSFT")
        self.assertEqual(hypo.legs[1].leg_type, LegType.SHORT_EQUITY)
        self.assertEqual(hypo.time_horizon_days, 14)
        self.assertEqual(hypo.risk_metrics.max_drawdown_tolerance, 5.0)

    def test_assemble_hypothesis_messy_legs(self):
        thesis_data = {
            "strategy_name": "Messy Strategy",
            "strategy_type": "PAIRS_TRADE",
            "economic_rationale": "Rationale with 123 numbers 456.",
            "catalysts": "Bullet 1\n* Bullet 2"
        }
        # Messy formatting with extra spaces and missing brackets
        trade_pred = MockPrediction(
            legs="NVDA : LONG (weight 2) Entry: MKT Exit: SL 10%\nAMD: SHORT (weight -1) [Entry: MKT]",
            days="One week",
            exit="Exit cond",
            risk="10% risk"
        )
        
        hypo = self.module._assemble_hypothesis(thesis_data, trade_pred)
        
        self.assertEqual(hypo.strategy_type, StrategyType.PAIRS_TRADE)
        self.assertEqual(len(hypo.legs), 2)
        self.assertEqual(hypo.legs[0].ticker, "NVDA")
        self.assertEqual(hypo.legs[0].relative_weight, 2.0)
        self.assertEqual(hypo.legs[1].ticker, "AMD")
        self.assertEqual(hypo.time_horizon_days, 7) # Default/fallback from "One week" logic or re-parse
        self.assertEqual(hypo.risk_metrics.max_drawdown_tolerance, 10.0)

    def test_fallback_parsing(self):
        thesis_data = {
            "strategy_name": "Fallback Strategy",
            "strategy_type": "DIRECTIONAL",
            "economic_rationale": "Some rationale 100.",
            "catalysts": []
        }
        # Completely non-standard legs string
        trade_pred = MockPrediction(
            legs="I think we should buy TSLA because it looks good.",
            days="30",
            exit="",
            risk=""
        )
        
        hypo = self.module._assemble_hypothesis(thesis_data, trade_pred)
        self.assertEqual(len(hypo.legs), 1)
        self.assertEqual(hypo.legs[0].ticker, "TSLA")
        self.assertEqual(hypo.legs[0].leg_type, LegType.LONG_EQUITY)

if __name__ == "__main__":
    unittest.main()
