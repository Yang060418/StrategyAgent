import unittest
from unittest.mock import patch, MagicMock, mock_open
import pandas as pd
import numpy as np
import json
import time
import os

from strategy_ideation_engine.data.ingestion.providers import FundamentalProvider
from strategy_ideation_engine.intelligence.modules.validator import TechnicalValidator
from strategy_ideation_engine.schemas.hypothesis import TradingHypothesis, TradeLeg, LegType, StrategyType, StrategyArchetype, RiskMetrics

class TestIngestion(unittest.TestCase):
    def setUp(self):
        # Create dummy OHLCV data - enough for SMA_200
        dates = pd.date_range(start="2024-01-01", periods=300)
        self.dummy_df = pd.DataFrame({
            'Open': np.random.uniform(100, 200, 300),
            'High': np.random.uniform(100, 200, 300),
            'Low': np.random.uniform(100, 200, 300),
            'Close': np.random.uniform(100, 200, 300),
            'Volume': np.random.uniform(1000, 5000, 300)
        }, index=dates)

    @patch('yfinance.Ticker')
    def test_indicator_verification_provider(self, mock_ticker):
        """Test that FundamentalProvider successfully appends indicators via pandas_ta."""
        mock_instance = mock_ticker.return_value
        mock_instance.history.return_value = self.dummy_df
        
        provider = FundamentalProvider()
        # Ensure we don't accidentally write to real .cache during tests
        # Use a more surgical mock for open in the provider's specific module
        with patch('os.path.exists', return_value=False), \
             patch('strategy_ideation_engine.data.ingestion.providers.open', mock_open(), create=True):
            indicators = provider.get_technical_indicators("AAPL")
            
        # Verify standard indicators are present
        expected_keys = ['RSI_14', 'SMA_50', 'SMA_200', 'BB_Upper', 'BB_Lower', 'MACD', 'MACD_Signal']
        for key in expected_keys:
            self.assertIn(key, indicators, f"{key} not found in {indicators.keys()}")
            self.assertIsInstance(indicators[key], float)

    def test_indicator_verification_validator(self):
        """Test that TechnicalValidator._prepare_dataframe appends correct columns."""
        validator = TechnicalValidator()
        # TechnicalValidator._prepare_dataframe takes (df, indicators_list, context_string)
        prepared_df = validator._prepare_dataframe(self.dummy_df, ["RSI_14", "SMA_50"], "Test Context")
        
        # Verify columns added by pandas_ta and mapping
        self.assertIn('RSI_14', prepared_df.columns)
        self.assertIn('SMA_50', prepared_df.columns)
        self.assertIn('BB_Upper', prepared_df.columns)
        self.assertIn('MACD', prepared_df.columns)
        self.assertIn('Price', prepared_df.columns)

    @patch('yfinance.download')
    def test_error_handling_bulk_download(self, mock_download):
        """Test that TechnicalValidator handles yfinance failures gracefully."""
        # Scenario: yfinance raises an exception
        mock_download.side_effect = Exception("Connection Timeout")
        
        validator = TechnicalValidator()
        hypo = TradingHypothesis(
            strategy_name="Failure Test",
            strategy_type=StrategyType.DIRECTIONAL,
            strategy_archetype=StrategyArchetype.TACTICAL,
            economic_rationale="Testing failure",
            catalysts=["Earnings"],
            time_horizon_days=5,
            legs=[TradeLeg(ticker="AAPL", leg_type=LegType.LONG_EQUITY, relative_weight=1.0, entry_condition="Price > 100", exit_condition="Price < 90")],
            invalidation_criteria=["Market Crash"],
            risk_metrics=RiskMetrics(max_drawdown_tolerance=5.0)
        )
        
        score, logs, metrics = validator.validate_and_backtest(hypo)
        
        self.assertEqual(score, 0.0)
        self.assertIn("Backtest Engine Error", logs)
        self.assertEqual(metrics["win_rate"], 0.0)

    @patch('yfinance.Ticker')
    def test_provider_fallback_handling(self, mock_ticker):
        """Test that FundamentalProvider handles empty data frames gracefully."""
        mock_instance = mock_ticker.return_value
        mock_instance.history.return_value = pd.DataFrame() # Empty
        
        provider = FundamentalProvider()
        with patch('os.path.exists', return_value=False):
            indicators = provider.get_technical_indicators("INVALID")
            
        self.assertEqual(indicators, {})

    @patch('yfinance.Ticker')
    def test_cache_verification(self, mock_ticker):
        """Verify that system reads from cache if it exists and is within 4-hour window."""
        provider = FundamentalProvider()
        ticker = "MSFT"
        cache_data = {
            "timestamp": time.time() - 1000, # ~16 mins ago (well within 4 hours)
            "indicators": {"RSI_14": 45.0, "SMA_50": 150.0}
        }
        
        # Mock os.path.exists to return True for the cache file
        # Mock open to return the json data
        json_str = json.dumps(cache_data)
        
        with patch('os.path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data=json_str)):
            
            indicators = provider.get_technical_indicators(ticker)
            
            # Assertions
            self.assertEqual(indicators["RSI_14"], 45.0)
            # Verify yfinance was NOT called because data was in cache
            mock_ticker.assert_not_called()

    @patch('yfinance.Ticker')
    def test_cache_expiry(self, mock_ticker):
        """Verify that system ignores expired cache (older than 4 hours)."""
        provider = FundamentalProvider()
        ticker = "MSFT"
        # 14400 seconds = 4 hours. Set to 5 hours ago.
        cache_data = {
            "timestamp": time.time() - 18000, 
            "indicators": {"RSI_14": 45.0}
        }
        
        mock_instance = mock_ticker.return_value
        mock_instance.history.return_value = self.dummy_df
        
        json_str = json.dumps(cache_data)
        
        with patch('os.path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data=json_str)) as mocked_file:
            
            # We also need to mock the WRITE because it will attempt to refresh the cache
            # The second call to open() will be for writing the new cache
            indicators = provider.get_technical_indicators(ticker)
            
            # Verify yfinance WAS called because cache was expired
            mock_ticker.assert_called_once()
            self.assertNotEqual(indicators["RSI_14"], 45.0)

if __name__ == '__main__':
    unittest.main()
