import dspy
from typing import List, Optional, Dict, Any
import uuid
import re
import time
from datetime import datetime

from strategy_ideation_engine.config import logger, settings, CORE_UNIVERSE
from strategy_ideation_engine.data.ingestion import IngestionOrchestrator
from strategy_ideation_engine.data.research import ResearchOrchestrator
from strategy_ideation_engine.data.persistence import HypothesisLedger
from strategy_ideation_engine.intelligence.modules.ideation import StrategyIdeationModule
from strategy_ideation_engine.intelligence.modules.adversarial import AdversarialFilterModule
from strategy_ideation_engine.intelligence.modules.validator import TechnicalValidator
from strategy_ideation_engine.data.exports import MarkdownExporter
from strategy_ideation_engine.schemas.event import MarketEvent, EventSource
from strategy_ideation_engine.schemas.hypothesis import (
    TradingHypothesis, StrategyType, StrategyArchetype, TradeLeg, LegType, RiskMetrics
)
from strategy_ideation_engine.schemas.state import StrategyScratchpad

from strategy_ideation_engine.intelligence.utils import apply_mercy_rule, TokenBudgeter

class StrategyEngine:
    """
    The central orchestrator of the Strategy Ideation Engine.
    Uses a Three-Tier Intelligence Architecture (Scout, Strategist, Critic).
    """
    def __init__(self):
        self.ingestor = IngestionOrchestrator()
        self.researcher = ResearchOrchestrator()
        self.ledger = HypothesisLedger()
        self.exporter = MarkdownExporter()
        
        # Initialize Three-Tier Intelligence
        self._setup_tiered_brains()
        self.brain = StrategyIdeationModule(lm=self.strategist_lm)
        self.scout_brain = StrategyIdeationModule(lm=self.scout_lm)
        self.gatekeeper = AdversarialFilterModule(lm=self.critic_lm)
        self.validator = TechnicalValidator()

    def _setup_tiered_brains(self):
        """Configure Scout, Strategist, and Critic backends with provider-aware routing."""
        try:
            self.scout_lm = self._setup_lm(settings.SCOUT_MODEL)
            self.strategist_lm = self._setup_lm(settings.STRATEGIST_MODEL)
            self.critic_lm = self._setup_lm(settings.CRITIC_MODEL)
            
            # Configure global dspy settings
            dspy.settings.configure(lm=self.strategist_lm)
            
            logger.info(f"Three-Tier Engine Ready: Scout={settings.SCOUT_MODEL}, Strategist={settings.STRATEGIST_MODEL}, Critic={settings.CRITIC_MODEL}")
        except Exception as e:
            logger.error(f"Failed to configure Tiered Brains: {e}")

    def _setup_lm(self, model_name: str) -> dspy.LM:
        """Helper to initialize dspy.LM with the correct provider settings."""
        api_key = settings.GROQ_API_KEY
        base_url = "https://api.groq.com/openai/v1"
        full_model_str = model_name

        if "/" in model_name:
            provider = model_name.split("/")[0].lower()
            
            # 1. ACTUAL OPENAI
            if provider == "openai" and settings.OPENAI_API_KEY:
                api_key = settings.OPENAI_API_KEY
                base_url = None
            
            # 2. ANTHROPIC
            elif provider == "anthropic":
                api_key = settings.ANTHROPIC_API_KEY
                base_url = None
                
            # 3. GROQ / OPENAI-Compatible providers
            elif provider in ["llama", "groq", "meta-llama", "qwen"]:
                api_key = settings.GROQ_API_KEY
                base_url = "https://api.groq.com/openai/v1"
        else:
            # Default to Groq
            api_key = settings.GROQ_API_KEY
            base_url = "https://api.groq.com/openai/v1"

        return dspy.LM(
            model=full_model_str,
            api_key=api_key,
            base_url=base_url,
            max_tokens=settings.DSPY_MAX_TOKENS
        )

    def run_on_event(self, event: MarketEvent) -> Optional[TradingHypothesis]:
        """Executes the Tiered workflow."""
        logger.info(f"Starting THREE-TIER ideation cycle for event: {event.event_id}")
        
        # Configure Caching
        dspy.configure(cache=True)
        
        # 1. Scope and Ingest
        tickers = event.affected_tickers if event.affected_tickers else CORE_UNIVERSE.allowed_tickers
        market_context = self.ingestor.get_market_context(tickers)
        
        # 2. Research
        query = f"trading opportunities for {', '.join(tickers)}"
        literature = self.researcher.get_literature_review(query, tickers)
        
        # 3. Initialize Scratchpad with Dynamic Token Budgeting
        # Strategist Budget: 12K TPM, 1.5K reserve = 10.5K available
        budgeter = TokenBudgeter(settings.STRATEGIST_MODEL, tpm_limit=12000, reserve_tokens=1500)
        
        # Priority 1: Macro & Fundamentals (High Density)
        macro_str = str(market_context.macro_data.model_dump(exclude={'last_updated'}))
        fund_data = {t: f.model_dump(exclude={'last_updated', 'next_earnings_date', 'beta'}) 
                    for t, f in market_context.fundamentals.items()}
        fund_str = str(fund_data)
        
        budgeter.available_budget -= budgeter.count_tokens(macro_str)
        budgeter.available_budget -= budgeter.count_tokens(fund_str)
        
        # Priority 2: Technicals
        tech_str = budgeter.budget_items({t: str(v) for t, v in market_context.technical_indicators.items()}, "Technicals")
        
        # Priority 3: SEC Filings (Bulkier, lower priority)
        sec_str = budgeter.budget_items(market_context.filing_summaries, "SEC Filings")
        
        scratchpad = StrategyScratchpad(event_id=event.event_id)
        scratchpad.market_context_str = (
            f"MACRO: {macro_str}\n"
            f"FUNDAMENTALS: {fund_str}\n"
            f"TECHNICALS:\n{tech_str}"
            f"SEC FILINGS:\n{sec_str}"
        )

        # --- TIER 1: SCOUT (Reading & Distillation) ---
        logger.info("Tier 1: Scout summarizing literature and synthesizing facts...")
        # Scout Budget: 6K TPM, 1.5K reserve = 4.5K available
        scout_budgeter = TokenBudgeter(settings.SCOUT_MODEL, tpm_limit=6000, reserve_tokens=1500)
        literature.insights = scout_budgeter.budget_list(literature.insights, "Literature Insights")
        
        scratchpad.research_summary_str = self.scout_brain.summarize_literature(literature)
        facts = self.scout_brain.synthesize_facts(scratchpad.market_context_str, scratchpad.research_summary_str)
        scratchpad.fact_sheet = facts["fact_sheet"]
        
        # --- TIER 2 & 3: STRATEGIST vs CRITIC DEBATE ---
        final_hypothesis = None
        refinement_feedback = ""
        adv_score = 0.0
        prev_score = 0.0
        
        previous_hypo_json = "None"
        previous_feedback = "None"
        full_critique_history = []
        
        for attempt in range(3):
            if attempt > 0:
                logger.info(f"Rate Limit Mitigation: Sleeping for 15s before attempt {attempt+1}...")
                time.sleep(15)

            logger.info(f"Strategist Architecting Thesis (Attempt {attempt+1})...")
            with dspy.settings.context(lm=self.strategist_lm):
                try:
                    thesis_data = self.brain.architect_thesis(
                        event=event,
                        market_str=scratchpad.market_context_str,
                        research_summary=scratchpad.research_summary_str,
                        fact_sheet=scratchpad.fact_sheet,
                        refinement_feedback=refinement_feedback
                    )
                except Exception as e:
                    logger.error(f"Strategist failed: {e}")
                    continue

            # Architect Trade Structure WITHIN the loop to allow backtest feedback
            logger.info(f"Strategist Architecting Trade Structure (Attempt {attempt+1})...")
            with dspy.settings.context(lm=self.strategist_lm):
                try:
                    temp_hypo = self.brain.architect_trade(
                        thesis_data, 
                        scratchpad.market_context_str, 
                        refinement_feedback=refinement_feedback
                    )
                except Exception as e:
                    logger.error(f"Trade Architecture failed: {e}")
                    continue

            # Phase 5b: Reality Check (Backtest) WITHIN the loop
            logger.info(f"Scout Executing Backtest (Attempt {attempt+1})...")
            with dspy.settings.context(lm=self.scout_lm):
                bt_score, bt_logs, bt_metrics = self.validator.validate_and_backtest(temp_hypo, market_context=market_context)
            
            backtest_summary = f"Backtest Score: {bt_score:.2f}\nMetrics: {bt_metrics}\nLogs: {bt_logs[:500]}..."

            # Internal Critique using separate CRITIC model
            logger.info(f"Critic Analyzing Strategy (Attempt {attempt+1})...")
            with dspy.settings.context(lm=self.critic_lm):
                try:
                    adv_score, adv_feedback = self.gatekeeper(
                        temp_hypo, 
                        market_context,
                        backtest_results=backtest_summary,
                        previous_hypothesis=previous_hypo_json,
                        previous_feedback=previous_feedback
                    )
                    
                    # Update memory for next turn
                    previous_hypo_json = temp_hypo.model_dump_json()
                    previous_feedback = adv_feedback
                    full_critique_history.append(f"Attempt {attempt+1}: {adv_feedback}")
                    
                    # Apply Mercy Rule (Boost score based on progress)
                    adv_score = apply_mercy_rule(adv_score, prev_score, adv_feedback, attempt)

                    if adv_score >= 0.7:
                        logger.info(f"Passed! Critic Score: {adv_score:.2f}")
                        scratchpad.add_critique(f"PASS (Attempt {attempt+1}): {adv_feedback}")
                        temp_hypo.backtest_score = bt_score
                        temp_hypo.backtest_metrics = bt_metrics
                        temp_hypo.validation_logs = bt_logs
                        temp_hypo.adversarial_score = adv_score
                        final_hypothesis = temp_hypo
                        break
                    else:
                        logger.warning(f"Rejected. Critic Score: {adv_score:.2f}")
                        # Combine backtest failure into feedback for next attempt
                        refinement_feedback = (
                            f"REJECTED (Score: {adv_score:.2f}).\n"
                            f"CRITICAL CORRECTIONS REQUIRED:\n{adv_feedback}\n\n"
                            f"EMPIRICAL BACKTEST RESULTS:\n{backtest_summary}"
                        )
                        scratchpad.add_critique(f"FAIL (Attempt {attempt+1}): {adv_feedback}")
                        prev_score = adv_score
                except Exception as e:
                    logger.error(f"Critic failed: {e}")
                    continue

        if not final_hypothesis: return None

        hypothesis = final_hypothesis
        hypothesis.adversarial_feedback = "\n".join(scratchpad.critique_history)
        hypothesis.hypothesis_id = str(uuid.uuid4())
        
        # 8. Persist and Export
        self.ledger.save(hypothesis, event.event_id)
        self.exporter.export(hypothesis)
        
        return hypothesis
