import pandas as pd
import pandas_ta as ta
import numpy as np
import yfinance as yf
import re
import dspy
from typing import Tuple, Dict, List, Any, Optional
from strategy_ideation_engine.schemas.hypothesis import TradingHypothesis, LegType
from strategy_ideation_engine.intelligence.signatures.ideation import IndicatorCoder
from strategy_ideation_engine.config import logger

from strategy_ideation_engine.config.universe import get_sector_benchmark
from strategy_ideation_engine.schemas.market_data import MarketContext

class TechnicalValidator:
    """
    Phase 5b: The 'Reality Check' Module.
    Upgraded with a 'Dynamic Indicator Engine' to calculate custom technicals,
    Transaction Cost modeling, Unified Portfolio Aggregation, and Sector-Aware Benchmarking.
    """
    def __init__(self, transaction_cost_pct: float = 0.001, slippage_pct: float = 0.0005):
        self.indicator_coder = dspy.Predict(IndicatorCoder)
        self.transaction_cost_pct = transaction_cost_pct
        self.slippage_pct = slippage_pct
    
    def validate_and_backtest(self, hypothesis: TradingHypothesis, market_context: Optional[MarketContext] = None) -> Tuple[float, str, Dict[str, Any]]:
        """Runs a historical simulation and returns score, logs, and structured metrics."""
        logger.info(f"Phase 5b: Portfolio Backtest for '{hypothesis.strategy_name}'")
        
        metrics = {
            "win_rate": 0.0,
            "avg_return": 0.0,
            "sharpe": 0.0,
            "num_triggers": 0,
            "alpha": 0.0,
            "beta": 0.0
        }

        tickers = list(set([leg.ticker for leg in hypothesis.legs]))
        if not tickers:
            return 0.0, "No tickers found in hypothesis.", metrics

        try:
            # 1. Discover needed indicators from leg conditions
            needed_indicators = self._discover_needed_indicators(hypothesis)
            
            # 2. Determine Benchmarks (Sector-Aware)
            ticker_benchmarks = {}
            all_fetch_tickers = list(tickers)
            if "SPY" not in all_fetch_tickers:
                all_fetch_tickers.append("SPY")
            
            if market_context:
                for t in tickers:
                    if t in market_context.fundamentals:
                        sector = market_context.fundamentals[t].sector
                        bench = get_sector_benchmark(sector)
                        ticker_benchmarks[t] = bench
                        if bench not in all_fetch_tickers:
                            all_fetch_tickers.append(bench)
            
            logger.info(f"Bulk downloading data for: {all_fetch_tickers}")
            bulk_data = yf.download(all_fetch_tickers, period="2y", group_by='ticker', progress=False)
            
            data_map = {}
            ticker_betas = {}
            
            market_bench_df = bulk_data["SPY"] if "SPY" in bulk_data else pd.DataFrame()
            
            for t in tickers:
                df = bulk_data[t].copy() if t in bulk_data else pd.DataFrame()
                df = df.dropna(subset=['Close'])
                
                if not df.empty:
                    # Determine relevant benchmark for this specific ticker
                    bench_ticker = ticker_benchmarks.get(t, "SPY")
                    bench_df = bulk_data[bench_ticker] if bench_ticker in bulk_data else market_bench_df
                    
                    if not bench_df.empty:
                        bench_aligned = bench_df.reindex(df.index, method='ffill')
                        t_ret = df['Close'].pct_change().dropna()
                        
                        # Calculate Beta relative to MARKET (SPY) for systemic risk
                        if not market_bench_df.empty:
                            m_bench_aligned = market_bench_df.reindex(df.index, method='ffill')
                            m_ret = m_bench_aligned['Close'].pct_change().dropna()
                            common_idx = t_ret.index.intersection(m_ret.index)
                            if len(common_idx) > 30:
                                cov = np.cov(t_ret.loc[common_idx], m_ret.loc[common_idx])[0,1]
                                var = np.var(m_ret.loc[common_idx])
                                ticker_betas[t] = cov / var if var != 0 else 1.0
                            else:
                                ticker_betas[t] = 1.0
                        
                        df['BENCH_Ret'] = bench_aligned['Close'].pct_change()
                        df['BENCH_Price'] = bench_aligned['Close']
                        df['BENCH_Ticker'] = bench_ticker
                    
                    data_map[t] = self._prepare_dataframe(df, needed_indicators, hypothesis.economic_rationale)
            
            if not data_map:
                return 0.0, "Failed to fetch data for any tickers.", metrics

            logs = []
            leg_results = []
            
            # 3. Individual Leg Evaluation (with friction)
            for leg in hypothesis.legs:
                if leg.ticker not in data_map:
                    logs.append(f"[{leg.ticker}] Data unavailable.")
                    continue
                
                df = data_map[leg.ticker]
                entry_cond = str(leg.entry_condition)
                exit_cond = str(leg.exit_condition)
                
                perf_score, detail, fwd_rets, bench_fwd_rets = self._evaluate_leg_condition_with_bench(
                    df, entry_cond, hypothesis.time_horizon_days, leg.leg_type, exit_cond
                )
                
                bench_name = df['BENCH_Ticker'].iloc[0] if 'BENCH_Ticker' in df.columns else "SPY"
                leg_results.append((leg.relative_weight, fwd_rets, bench_fwd_rets, ticker_betas.get(leg.ticker, 1.0)))
                logs.append(f"[{leg.ticker} {leg.leg_type.value}] {detail} (vs {bench_name}, Beta: {ticker_betas.get(leg.ticker, 1.0):.2f})")

            # 4. Portfolio Aggregation (Unified Co-integration)
            portfolio_win_rate, portfolio_avg_return, portfolio_summary, metrics = self._aggregate_portfolio_with_alpha(
                leg_results, hypothesis.time_horizon_days
            )
            
            logs.append("\n=== STRATEGY PORTFOLIO SUMMARY (NET OF FRICTION) ===")
            logs.append(portfolio_summary)
            
            # Score: Win Rate (40%), Sharpe (30%), Alpha (30%)
            alpha_score = min(max(metrics['alpha'] * 10, 0), 0.3) 
            sharpe_score = min(max(metrics['sharpe'] / 3.0, 0), 0.3) 
            win_score = metrics['win_rate'] * 0.4
            
            final_score = win_score + sharpe_score + alpha_score
            return float(max(0, final_score)), "\n".join(logs), metrics
            
        except Exception as e:
            logger.error(f"Backtest Critical Failure: {e}")
            return 0.0, f"Backtest Engine Error: {str(e)}", metrics

    def _evaluate_leg_condition_with_bench(self, df: pd.DataFrame, entry_cond: str, horizon: int, leg_type: LegType, exit_cond: str = None) -> Tuple[float, str, pd.Series, pd.Series]:
        """Evaluates a single leg's condition relative to benchmark with frictions."""
        try:
            clean_entry = str(entry_cond).strip().upper()
            if not clean_entry or clean_entry in ['MKT', 'NULL', 'NONE', 'MARKET', 'AT MARKET', 'CURRENT PRICE']:
                mask = pd.Series(True, index=df.index)
            else:
                query_str = str(entry_cond).replace("AND", "&").replace("OR", "|").replace("and", "&").replace("or", "|")
                query_str = re.sub(r"\bMARKET\b", "True", query_str, flags=re.I)
                query_str = re.sub(r"\bMKT\b", "True", query_str, flags=re.I)
                mask = df.eval(query_str)
            
            triggers = df[mask]
            fwd_returns = pd.Series(dtype=float)
            bench_fwd_returns = pd.Series(dtype=float)
            
            # Friction Modeling: Apply costs to entry and exit
            total_friction = self.transaction_cost_pct + self.slippage_pct
            
            for idx in triggers.index:
                try:
                    loc = df.index.get_loc(idx)
                    if loc + horizon < len(df):
                        # Ticker Return
                        entry_price = df.iloc[loc]['Close']
                        exit_price = df.iloc[loc + horizon]['Close']
                        raw_ret = (exit_price - entry_price) / entry_price
                        
                        # Apply friction (approx raw_ret - 2*friction)
                        net_raw_ret = raw_ret - (2 * total_friction)
                        
                        # Benchmark Return
                        bench_entry = df.iloc[loc]['BENCH_Price'] if 'BENCH_Price' in df.columns else entry_price
                        bench_exit = df.iloc[loc+horizon]['BENCH_Price'] if 'BENCH_Price' in df.columns else exit_price
                        bench_raw_ret = (bench_exit - bench_entry) / bench_entry if bench_entry != 0 else 0
                        
                        ret_val = 0.0
                        if leg_type == LegType.LONG_EQUITY: ret_val = net_raw_ret
                        elif leg_type == LegType.SHORT_EQUITY: ret_val = -raw_ret - (2 * total_friction)
                        elif leg_type == LegType.CALL_OPTION: ret_val = max(-1.0, min(5.0, net_raw_ret * 5))
                        elif leg_type == LegType.PUT_OPTION: ret_val = max(-1.0, min(5.0, -raw_ret * 5 - (2 * total_friction)))
                        else: ret_val = net_raw_ret
                        
                        fwd_returns.at[idx] = ret_val
                        bench_fwd_returns.at[idx] = bench_raw_ret 
                except: continue
            
            if fwd_returns.empty: return 0.1, "No triggers found.", fwd_returns, bench_fwd_returns
            win_rate = len(fwd_returns[fwd_returns > 0]) / len(fwd_returns)
            return 0.5, f"Win Rate: {win_rate:.1%}, Avg Net Ret: {fwd_returns.mean():.2%}", fwd_returns, bench_fwd_returns
        except Exception as e:
            return 0.0, f"Parse Error: {str(e)}", pd.Series(), pd.Series()



    def _aggregate_portfolio_with_alpha(self, leg_results: List[Tuple[float, pd.Series, pd.Series, float]], horizon: int) -> Tuple[float, float, str, Dict[str, Any]]:
        metrics = {"win_rate": 0.0, "avg_return": 0.0, "sharpe": 0.0, "num_triggers": 0, "portfolio_beta": 0.0, "alpha": 0.0}
        if not leg_results: return 0.0, 0.0, "Portfolio empty.", metrics
        
        # Calculate Weighted Beta
        total_abs_weight = sum(abs(w) for w, _, _, _ in leg_results)
        weighted_beta = sum(w * b for w, _, _, b in leg_results) / total_abs_weight if total_abs_weight > 0 else 0.0
        
        all_dates = pd.Index([])
        for _, s, _, _ in leg_results: all_dates = all_dates.union(s.index)
        if all_dates.empty: return 0.0, 0.0, f"No valid strategy trigger dates found. (Portfolio Beta: {weighted_beta:.2f})", metrics

        portfolio_returns = []
        benchmark_returns = []
        for date in all_dates:
            daily_weighted_ret, daily_bench_ret, current_abs_weight = 0.0, 0.0, 0.0
            for weight, series, bench_series, _ in leg_results:
                if date in series.index:
                    daily_weighted_ret += series[date] * weight
                    # Benchmark for the leg's direction
                    # If we are short the ticker, alpha is (-TickerRet) - (-BenchRet)? 
                    # Simpler: Alpha = LegRet - (Weight * BenchRet)
                    daily_bench_ret += bench_series[date] * weight
                    current_abs_weight += abs(weight)
            
            if current_abs_weight > 0: 
                portfolio_returns.append(daily_weighted_ret / current_abs_weight)
                benchmark_returns.append(daily_bench_ret / current_abs_weight)

        if not portfolio_returns: return 0.0, 0.0, "Aggregation failed.", metrics
        
        win_rate = len([r for r in portfolio_returns if r > 0]) / len(portfolio_returns)
        avg_return = float(np.mean(portfolio_returns))
        avg_bench_return = float(np.mean(benchmark_returns))
        alpha = avg_return - avg_bench_return
        
        std_dev = np.std(portfolio_returns)
        sharpe = float((avg_return / std_dev) * np.sqrt(252/horizon)) if std_dev > 0 else 0.0
        
        metrics = {
            "win_rate": float(win_rate),
            "avg_return": avg_return,
            "sharpe": sharpe,
            "num_triggers": len(portfolio_returns),
            "portfolio_beta": float(weighted_beta),
            "alpha": float(alpha)
        }
        
        summary = f"Win Rate: {win_rate:.1%}, Alpha: {alpha:.2%}, Sharpe: {sharpe:.2f}, Net Beta: {weighted_beta:.2f}, Triggers: {len(portfolio_returns)}"
        return win_rate, avg_return, summary, metrics


    def _discover_needed_indicators(self, hypothesis: TradingHypothesis) -> List[str]:
        """Scans conditions for indicator names."""
        all_text = " ".join([str(l.entry_condition) + " " + str(l.exit_condition) for l in hypothesis.legs])
        # Replace hyphens with underscores for discovery
        all_text = all_text.replace("-", "_")
        found = re.findall(r"\b[A-Z][A-Z0-9_]{2,}\b", all_text)
        tickers = {leg.ticker.upper() for leg in hypothesis.legs}
        exclude = {'AND', 'OR', 'NOT', 'TRUE', 'FALSE', 'Price', 'Close', 'Open', 'High', 'Low', 'Volume'}.union(tickers)
        return list(set([f for f in found if f not in exclude]))

    def _prepare_dataframe(self, df: pd.DataFrame, indicators: List[str], context: str) -> pd.DataFrame:
        """Calculates needed indicators using pandas_ta for professional fidelity."""
        df = df.copy()
        df['Price'] = df['Close']
        
        # 1. CORE INDICATOR CALCULATION VIA PANDAS_TA
        # Moving Averages
        df.ta.sma(length=20, append=True)
        df.ta.sma(length=50, append=True)
        df.ta.sma(length=200, append=True)
        df.ta.ema(length=20, append=True)
        
        # Rename pandas_ta columns to our standard names for backward compatibility if needed, 
        # or just ensure we have the ones we use.
        df['SMA_20'] = df['SMA_20']
        df['SMA_50'] = df['SMA_50']
        df['SMA_200'] = df['SMA_200']
        df['EMA_20'] = df['EMA_20']
        
        # Volatility
        df.ta.atr(length=14, append=True)
        # Handle variations like ATRr_14 or ATR_14
        atr_col = next((c for c in df.columns if c.startswith('ATR')), None)
        if atr_col:
            df['ATR_14'] = df[atr_col]
            df['ATR'] = df[atr_col]

        # Bollinger Bands
        df.ta.bbands(length=20, std=2, append=True)
        df['BB_Mid'] = df.get('BBM_20_2.0', df.get('BBM_20_2', pd.Series(dtype=float)))
        df['BB_Upper'] = df.get('BBU_20_2.0', df.get('BBU_20_2', pd.Series(dtype=float)))
        df['BB_Lower'] = df.get('BBL_20_2.0', df.get('BBL_20_2', pd.Series(dtype=float)))

        # RSI
        df.ta.rsi(length=14, append=True)
        df['RSI_14'] = df.get('RSI_14', pd.Series(dtype=float))
        df['RSI'] = df['RSI_14']

        # MACD
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        df['MACD'] = df.get('MACD_12_26_9', pd.Series(dtype=float))
        df['MACD_Signal'] = df.get('MACDs_12_26_9', pd.Series(dtype=float))
        
        # 2. Safety Check: Log and skip unknown indicators instead of using exec()
        for ind in indicators:
            if ind not in df.columns:
                # Try to calculate on the fly with pandas_ta if it follows a pattern like RSI_21
                try:
                    parts = ind.split('_')
                    if len(parts) == 2 and parts[1].isdigit():
                        name, length = parts[0].lower(), int(parts[1])
                        func = getattr(df.ta, name, None)
                        if func:
                            func(length=length, append=True)
                            logger.info(f"Dynamically calculated {ind} via pandas_ta")
                        else:
                            logger.warning(f"Unsupported indicator: {ind}")
                    else:
                        logger.warning(f"Unknown indicator format: {ind}")
                except Exception as e:
                    logger.warning(f"Failed to dynamically calculate {ind}: {e}")
            
        return df

    def _evaluate_leg_condition(self, df: pd.DataFrame, entry_cond: str, horizon: int, leg_type: LegType, exit_cond: str = None) -> Tuple[float, str, pd.Series]:
        """Evaluates a single leg's condition with robust parsing for Market triggers."""
        try:
            # Handle common 'Market' or 'None' triggers
            clean_entry = str(entry_cond).strip().upper()
            if not clean_entry or clean_entry in ['MKT', 'NULL', 'NONE', 'MARKET', 'AT MARKET', 'CURRENT PRICE']:
                mask = pd.Series(True, index=df.index)
            else:
                # Prepare query string for pandas eval
                query_str = str(entry_cond).replace("AND", "&").replace("OR", "|").replace("and", "&").replace("or", "|")
                # Handle cases where the LLM might still use 'Market' inside a larger condition
                query_str = re.sub(r"\bMARKET\b", "True", query_str, flags=re.I)
                query_str = re.sub(r"\bMKT\b", "True", query_str, flags=re.I)
                mask = df.eval(query_str)
            
            # Similar handling for exit condition (though less critical for the current backtest logic)
            triggers = df[mask]
            fwd_returns = pd.Series(dtype=float)
            
            for idx in triggers.index:
                try:
                    loc = df.index.get_loc(idx)
                    if loc + horizon < len(df):
                        entry_price = df.iloc[loc]['Close']
                        exit_price = df.iloc[loc + horizon]['Close']
                        raw_ret = (exit_price - entry_price) / entry_price
                        
                        ret_val = 0.0
                        if leg_type == LegType.LONG_EQUITY: ret_val = raw_ret
                        elif leg_type == LegType.SHORT_EQUITY: ret_val = -raw_ret
                        elif leg_type == LegType.CALL_OPTION: ret_val = max(-1.0, min(5.0, raw_ret * 5))
                        elif leg_type == LegType.PUT_OPTION: ret_val = max(-1.0, min(5.0, -raw_ret * 5))
                        else: ret_val = raw_ret
                        
                        fwd_returns.at[idx] = ret_val
                except: continue
            
            if fwd_returns.empty: return 0.1, "No triggers found.", fwd_returns
            win_rate = len(fwd_returns[fwd_returns > 0]) / len(fwd_returns)
            return 0.5, f"Win Rate: {win_rate:.1%}, Avg Ret: {fwd_returns.mean():.2%}", fwd_returns
        except Exception as e:
            return 0.0, f"Parse Error: {str(e)}", pd.Series()

    def _aggregate_portfolio(self, leg_results: List[Tuple[float, pd.Series, float]], horizon: int) -> Tuple[float, float, str, Dict[str, Any]]:
        metrics = {"win_rate": 0.0, "avg_return": 0.0, "sharpe": 0.0, "num_triggers": 0, "portfolio_beta": 0.0}
        if not leg_results: return 0.0, 0.0, "Portfolio empty.", metrics
        
        # Calculate Weighted Beta
        total_abs_weight = sum(abs(w) for w, _, _ in leg_results)
        weighted_beta = sum(w * b for w, _, b in leg_results) / total_abs_weight if total_abs_weight > 0 else 0.0
        
        all_dates = pd.Index([])
        for _, s, _ in leg_results: all_dates = all_dates.union(s.index)
        if all_dates.empty: return 0.0, 0.0, f"No valid strategy trigger dates found. (Portfolio Beta: {weighted_beta:.2f})", metrics

        portfolio_returns = []
        for date in all_dates:
            daily_weighted_ret, current_abs_weight = 0.0, 0.0
            for weight, series, _ in leg_results:
                if date in series.index:
                    daily_weighted_ret += series[date] * weight
                    current_abs_weight += abs(weight)
            if current_abs_weight > 0: portfolio_returns.append(daily_weighted_ret / current_abs_weight)

        if not portfolio_returns: return 0.0, 0.0, "Aggregation failed.", metrics
        win_rate = len([r for r in portfolio_returns if r > 0]) / len(portfolio_returns)
        avg_return = float(np.mean(portfolio_returns))
        std_dev = np.std(portfolio_returns)
        sharpe = float((avg_return / std_dev) * np.sqrt(252/horizon)) if std_dev > 0 else 0.0
        
        metrics = {
            "win_rate": float(win_rate),
            "avg_return": avg_return,
            "sharpe": sharpe,
            "num_triggers": len(portfolio_returns),
            "portfolio_beta": float(weighted_beta)
        }
        
        summary = f"Win Rate: {win_rate:.1%}, Avg Return: {avg_return:.2%}, Sharpe: {sharpe:.2f}, Net Beta: {weighted_beta:.2f}, Triggers: {len(portfolio_returns)}"
        return win_rate, avg_return, summary, metrics
