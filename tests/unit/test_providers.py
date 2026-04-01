import unittest
import os
import json
import time
from unittest.mock import MagicMock, patch
from strategy_ideation_engine.data.ingestion.providers import MacroProvider, FundamentalProvider
from strategy_ideation_engine.schemas.market_data import MacroSnapshot

class TestProviders(unittest.TestCase):
    def setUp(self):
        # Ensure .cache exists
        if not os.path.exists(".cache"):
            os.makedirs(".cache")
        self.macro_cache_file = ".cache/macro_snapshot.json"
        if os.path.exists(self.macro_cache_file):
            os.remove(self.macro_cache_file)

    def test_macro_provider_caching(self):
        provider = MacroProvider()
        
        # Mock FRED to return some data
        mock_series = MagicMock()
        mock_series.iloc = [-1]
        mock_series.__getitem__.return_value = mock_series # For .iloc[-1]
        
        # Better mock for series
        class MockSeries:
            def __init__(self, val): self.val = val
            @property
            def iloc(self): return [self.val]
            def pct_change(self, **kwargs): return self
            def __mul__(self, *args): return self
            def __getitem__(self, idx): return self.val

        mock_fred = MagicMock()
        provider._fred = mock_fred
        
        mock_fred.get_series.side_effect = [
            MockSeries(5.33), # FEDFUNDS
            MockSeries(0.03), # CPI (pct_change)
            MockSeries(3.9)   # UNRATE
        ]
        
        # First call - should fetch from mock
        snap1 = provider.get_macro_snapshot()
        self.assertEqual(snap1.fed_funds_rate, 5.33)
        self.assertTrue(os.path.exists(self.macro_cache_file))
        
        # Second call - should use cache (mock_fred shouldn't be called again)
        mock_fred.get_series.reset_mock()
        snap2 = provider.get_macro_snapshot()
        self.assertEqual(snap2.fed_funds_rate, 5.33)
        mock_fred.get_series.assert_not_called()

    def test_fundamental_provider_caching(self):
        provider = FundamentalProvider()
        
        with patch('yfinance.Ticker') as mock_ticker:
            mock_instance = mock_ticker.return_value
            mock_instance.info = {"forwardPE": 20.0, "beta": 1.2}
            mock_instance.calendar = None
            
            # First call for AAPL
            snap1 = provider.get_ticker_fundamentals("AAPL")
            self.assertEqual(snap1.pe_ratio, 20.0)
            self.assertEqual(mock_ticker.call_count, 1)
            
            # Second call for AAPL - should be cached in memory
            snap2 = provider.get_ticker_fundamentals("AAPL")
            self.assertEqual(snap2.pe_ratio, 20.0)
            self.assertEqual(mock_ticker.call_count, 1) # Still 1

if __name__ == "__main__":
    unittest.main()
