# StrategyIdeationEngine: Development Roadmap & Technical Debt

This document tracks the current drawbacks, technical debt, and planned improvements for the StrategyIdeationEngine. These items were identified during the Phase 1-3 stabilization refactor (March 2026).

## Current Drawbacks & Technical Debt

### 1. The "Distillation Paradox" (Information Loss)
*   **Status:** Resolved (March 2026)
*   **Description:** Implemented a dynamic `TokenBudgeter` using `tiktoken` that scales context based on model-specific TPM limits.
*   **Impact:** Maximize available signal (SEC/News/Research) while staying within API safety thresholds.

### 2. Provider Fragility (Data Bottlenecks)
*   **Status:** Resolved (March 2026)
*   **Description:** Migrated to `yfinance` bulk downloads and local `pandas_ta` processing. Implemented `.cache/tech_data` disk caching.
*   **Impact:** Eliminated Alpha Vantage rate-limit bottlenecks for technicals; significantly improved backtest speed via bulk data fetching.

### 3. Numerical Density & Grounding
*   **Status:** Quality Gap
*   **Description:** Despite strict Pydantic schemas, models (especially 8B variants) struggle to carry specific metrics from the Fact Sheet into the final Economic Rationale.
*   **Impact:** Strategy reports can still feel "prose-heavy" rather than "data-driven."

### 4. Backtest Simulation Fidelity
*   **Status:** Resolved (March 2026)
*   **Description:** The `TechnicalValidator` now includes:
    *   **Transaction Costs/Slippage:** Configurable frictions applied to entry/exit.
    *   **Unified Portfolio State:** Daily return aggregation across all co-integrated legs.
*   **Impact:** Higher confidence in theoretical Alpha before live deployment.

### 5. Deterministic Indicator Rigidity
*   **Status:** Mitigated (March 2026)
*   **Description:** Transitioned to `pandas_ta`, enabling the engine to dynamically call any of the 130+ professional indicators in the library without requiring `exec()`.
*   **Impact:** High innovation ceiling; model can now request non-standard indicators (e.g., ADX, SuperTrend) by name.

### 6. Homogeneous Benchmarking
*   **Status:** Resolved (March 2026)
*   **Description:** Implemented **Sector-Aware Benchmarking**. The engine automatically maps tickers to relevant Sector ETFs (XLK, XLE, etc.) for Alpha and Beta calculations.
*   **Impact:** Correctly identifies Alpha vs. Sector-specific Beta.

### 7. Zero-Shot Prompt Dependency (Lack of DSPy Optimization)
*   **Status:** Resolved (March 2026)
*   **Description:** Refactored `StrategyIdeationModule` into a unified `dspy.Module`. Implemented `EngineCompiler` to bootstrap and optimize prompts using 'Gold Standard' examples from the `HypothesisLedger`.
*   **Impact:** The system now programmatically learns from historical successes, moving beyond static prompts.

---

## Future Improvement Phases

### Phase 4: Analytical Depth & Institutional Benchmarking
- [x] **Sector-Aware Benchmarking:** Automatically identify and fetch the relevant Sector ETF for every strategy's Alpha calculation. (Completed March 2026)
- [x] **Advanced Portfolio Simulation:** Implement a unified backtest engine that accounts for slippage and co-integration of trade legs. (Completed March 2026)
- [x] **Indicator Registry Expansion:** Integrated `pandas_ta` allowing access to 130+ indicators. (Completed March 2026)

### Phase 5: Programmatic Optimization & Learning (DSPy "Compilation")
- [x] **Refactor to dspy.Module:** Unified the Strategist and Architect into a single compilable execution path. (Completed March 2026)
- [x] **Numerical Grounding (Anti-Hallucination):** Implemented `GroundingValidator` and `dspy.Suggest` to ensure all rationales are factually anchored to the Fact Sheet. (Completed March 2026)
- [ ] **Establish "Gold Standard" Dataset:** Build a training set of 50+ "Perfect" strategy-event pairs. (Captured via enhanced `HypothesisLedger` tracing)
- [ ] **Implement DSPy MIPRO/BootstrapFewShot:** Fully automate the instruction and few-shot optimization using the `EngineCompiler`.

### Phase 6: Reliability & Model Orchestration
- [x] **Dynamic Context Scaling:** Implemented `TokenBudgeter` to scale context (SEC/News) based on model limits. (Completed March 2026)
- [ ] **Structured Output "Self-Correction":** Add a small DSPy "Repair" module specifically for fixing 8B model JSON formatting errors before they hit Pydantic validation.
- [ ] **Hybrid Model Routing:** Use 70B models only for the final "Architect" and "Critic" steps, while using 8B models for all data-cleaning tasks to maximize TPD efficiency.

### Phase 6: Operational Robustness
- [x] **Tiered API Fallbacks:** Implemented yfinance-primary with Alpha Vantage-secondary fallback architecture. (Completed March 2026)
- [ ] **PDF/Professional Export:** Upgrade the `MarkdownExporter` to generate high-fidelity, client-ready PDF reports with embedded charts.
