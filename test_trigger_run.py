import uuid
from datetime import datetime
from strategy_ideation_engine.engine import StrategyEngine
from strategy_ideation_engine.schemas.event import MarketEvent, EventSource
from strategy_ideation_engine.config import logger

def run_test_case():
    # 1. Map user's sample input to our MarketEvent schema
    raw_input = {
        "event_id": "idx_shock_03202026",
        "timestamp": "2026-03-20T16:00:00Z",
        "source": "MARKET_ANOMALY_ENGINE",
        "affected_index": "SPX",
        "consolidated_metrics": {
            "index_price_impact_pct": -1.50,
            "aggregated_sentiment_score": -0.85,
            "advancing_declining_ratio": 0.18,
            "vix_level": 24.50
        },
        "raw_payload": {
            "headline": "S&P 500 drops 1.5% to close fourth losing week as oil prices surge past $100 on Strait of Hormuz closures",
            "trigger_type": "Geopolitical Supply Shock",
            "primary_driver": "Energy Prices",
            "refinement_needed": "Assess transport and logistics sector exposure"
        }
    }

    # Since we need to pick specific tickers to avoid scanning 500+ stocks
    # and the payload asks to assess transport/logistics vs energy:
    test_tickers = ["XOM", "CVX", "FDX", "UPS", "DAL"]

    # Combine metrics into raw_payload for the LLM to see it
    combined_payload = raw_input["raw_payload"].copy()
    combined_payload["metrics"] = raw_input["consolidated_metrics"]
    combined_payload["affected_index"] = raw_input["affected_index"]

    event = MarketEvent(
        event_id=raw_input["event_id"],
        timestamp=datetime.fromisoformat(raw_input["timestamp"].replace("Z", "+00:00")),
        source=EventSource.PRICE_SHOCK, # Mapping "MARKET_ANOMALY_ENGINE"
        affected_tickers=test_tickers,
        raw_payload=combined_payload
    )

    print(f"\n🧪 [TEST CASE] Initiating research on: {test_tickers}")
    print(f"Trigger: {event.raw_payload['headline']}\n")

    engine = StrategyEngine()
    hypothesis = engine.run_on_event(event)

    if hypothesis:
        print("\n✅ TEST SUCCESSFUL: Hypothesis generated and validated.")
        print(f"Strategy Name: {hypothesis.strategy_name}")
        print(f"Adversarial Score: {hypothesis.adversarial_score}")
        print(f"Backtest Score: {hypothesis.backtest_score}")
        print(f"Report Location: exports/HYPO_{hypothesis.strategy_name.replace(' ', '_')}_*.md")
    else:
        print("\n❌ TEST FAILED: Engine failed to generate a hypothesis.")

if __name__ == "__main__":
    run_test_case()
