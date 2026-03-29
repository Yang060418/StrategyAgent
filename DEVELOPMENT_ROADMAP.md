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
*   **Status:** Architectural Debt
*   **Description:** The engine currently relies on static "Zero-Shot" prompts defined in the signatures. It does not use DSPy's `Teleprompter` or `Compiler` to optimize prompts based on historical successes.
*   **Impact:** The system does not "learn" which reasoning patterns lead to higher backtest scores or lower Critic rejection rates over time.

---

## Future Improvement Phases

### Phase 4: Analytical Depth & Institutional Benchmarking
- [x] **Sector-Aware Benchmarking:** Automatically identify and fetch the relevant Sector ETF for every strategy's Alpha calculation. (Completed March 2026)
- [x] **Advanced Portfolio Simulation:** Implement a unified backtest engine that accounts for slippage and co-integration of trade legs. (Completed March 2026)
- [x] **Indicator Registry Expansion:** Integrated `pandas_ta` allowing access to 130+ indicators. (Completed March 2026)

### Phase 5: Programmatic Optimization & Learning (DSPy "Compilation")
- [ ] **Establish "Gold Standard" Dataset:** Build a training set of 50+ "Perfect" strategy-event pairs that passed both the Critic and Backtest with high scores.
- [ ] **Implement DSPy MIPRO/BootstrapFewShot:** Use a teleprompter to optimize the `Strategist` and `Critic` prompts based on the Gold Standard dataset.
- [ ] **Automated Prompt Versioning:** Track how prompt optimizations affect "Alpha" generation across different market regimes.

### Phase 6: Reliability & Model Orchestration
- [x] **Dynamic Context Scaling:** Implemented `TokenBudgeter` to scale context (SEC/News) based on model limits. (Completed March 2026)
- [ ] **Structured Output "Self-Correction":** Add a small DSPy "Repair" module specifically for fixing 8B model JSON formatting errors before they hit Pydantic validation.
- [ ] **Hybrid Model Routing:** Use 70B models only for the final "Architect" and "Critic" steps, while using 8B models for all data-cleaning tasks to maximize TPD efficiency.

### Phase 6: Operational Robustness
- [x] **Tiered API Fallbacks:** Implemented yfinance-primary with Alpha Vantage-secondary fallback architecture. (Completed March 2026)
- [ ] **PDF/Professional Export:** Upgrade the `MarkdownExporter` to generate high-fidelity, client-ready PDF reports with embedded charts.
