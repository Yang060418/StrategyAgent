import argparse
import sys
import uuid
import time
from datetime import datetime, timezone
from pprint import pprint

from strategy_ideation_engine.engine import StrategyEngine
from strategy_ideation_engine.schemas.event import MarketEvent, EventSource
from strategy_ideation_engine.config import logger, CORE_UNIVERSE

def main():
    parser = argparse.ArgumentParser(description="Strategy Ideation Engine CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: run (Manual Initiation)
    run_parser = subparsers.add_parser("run", help="Trigger a manual strategy ideation cycle")
    run_parser.add_argument("--ticker", "-t", type=str, help="Specify a ticker")
    run_parser.add_argument("--reason", "-r", type=str, default="Manual Run", help="Why is this engine being triggered?")

    # Command: watch (Market Watcher Bot Simulation)
    watch_parser = subparsers.add_parser("watch", help="Simulate a market watch bot scanning for triggers")
    watch_parser.add_argument("--interval", "-i", type=int, default=3600, help="Scan interval in seconds")

    # Command: history (view the ledger)
    subparsers.add_parser("history", help="Show all historical hypotheses from the ledger")

    args = parser.parse_args()

    engine = StrategyEngine()

    if args.command == "run":
        # Create a mock market event
        event = MarketEvent(
            event_id=f"evt_{uuid.uuid4().hex[:8]}",
            timestamp=datetime.now(timezone.utc),
            source=EventSource.SYSTEMATIC_RUN if args.reason == "Manual Run" else EventSource.NEWS_ALERT,
            affected_tickers=[args.ticker] if args.ticker else [],
            raw_payload={"reason": args.reason}
        )
        
        print(f"\n🚀 [MANUAL] Triggering Engine for: {args.ticker if args.ticker else 'Core Universe'}")
        engine.run_on_event(event)

    elif args.command == "watch":
        print(f"\n🤖 [WATCHER] Starting Market Watcher Bot (Interval: {args.interval}s)...")
        print(f"Scanning Universe: {CORE_UNIVERSE.allowed_tickers}\n")
        
        try:
            while True:
                for ticker in CORE_UNIVERSE.allowed_tickers:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Scanning for news/data catalysts on {ticker}...")
                    
                    # Simulate a trigger if news is found or data anomaly (In reality, this would be a real API check)
                    event = MarketEvent(
                        event_id=f"evt_{uuid.uuid4().hex[:8]}",
                        timestamp=datetime.now(timezone.utc),
                        source=EventSource.NEWS_ALERT,
                        affected_tickers=[ticker],
                        raw_payload={"trigger": "Automated Scanner Signal"}
                    )
                    engine.run_on_event(event)
                    print(f"--- Completed scan for {ticker}. Waiting for next cycle... ---\n")
                    
                print(f"Cycle complete. Sleeping for {args.interval} seconds...")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nWatcher stopped by user.")

    elif args.command == "history":
        history = engine.ledger.load_all()
        if not history:
            print("Ledger is empty.")
        else:
            print(f"\n📜 Historical Ledger ({len(history)} entries):")
            for entry in history:
                hypo = entry['hypothesis']
                print(f"- [{entry['saved_at'][:19]}] {hypo['strategy_name']} (Adv: {hypo.get('adversarial_score', 0)}, BT: {hypo.get('backtest_score', 0)})")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
