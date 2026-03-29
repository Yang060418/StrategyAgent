# StrategyIdeationEngine (StrategyAgent)
### *Institutional-Grade AI Agent for Quantitative Strategy Research*

The **StrategyIdeationEngine** is a sophisticated, self-correcting AI agent designed to generate, critique, and empirically validate mid-term (days to weeks) trading hypotheses. Unlike standard LLM "narrative" generators, this engine uses a **Tiered "Scout & Strategist" Architecture** to force every idea through a rigorous internal "Red Team" and a 2-year portfolio-level historical backtest.

---

## 🚀 Core Architecture: The Tiered Pipeline

The engine operates on a multi-agent orchestration designed to reduce hallucinations and maximize alpha quality:

1.  **Phase 1: Event Triggering:** Manual ticker research or an automated scanner bot.
2.  **Phase 2: Ground Truth Ingestion:** Unified data gathering from **yfinance**, **Alpha Vantage**, and **FRED**. 
3.  **Phase 3: The Scout (Tier 1):** Powered by fast models (e.g., Llama-4 17B). Condenses thousands of tokens from **ArXiv papers**, **Tavily news**, and **SEC filings** into a distilled **Numerical Fact Sheet**.
4.  **Phase 4: The Strategist (Tier 2):** Powered by high-reasoning models (e.g., Llama-3.3 70B).
    *   **Thesis Architect:** Develops a multi-leg economic rationale with citations.
    *   **Trade Architect:** Translates the thesis into concrete math rules (RSI, Bollinger Bands, etc.) supporting **Equity** and **Options** (Calls/Puts).
5.  **Phase 5: Double Validation & Self-Correction:**
    *   **Adversarial Critique:** An internal Red Team LLM analyzes the strategy for logical flaws and horizon-specific risks.
    *   **High-Fidelity Backtest:** A quantitative module that simulates the **entire strategy portfolio** as a co-integrated unit. Now includes **transaction costs and slippage** modeling for institutional-grade accuracy.
    *   **Sector-Aware Benchmarking:** Automatically identifies the relevant sector ETF (e.g., XLK, XLE) for every ticker to isolate true Alpha from Sector Beta.
    *   **Refinement Loop:** If scores are low (< 0.7), the engine passes a **Self-Correction Plan** back to the Strategist for up to 3 polish cycles.
6.  **Phase 6: Professional Export:** Generates detailed Markdown reports.

---

## 🚦 Getting Started

### 1. Prerequisites
*   Python 3.10+
*   API Keys for: Groq/OpenRouter (Primary), Tavily (Search), FRED (optional), Alpha Vantage (optional).

### 2. Configuration
Create a `.env` file in the root directory:
```env
PROVIDER_API_KEY=your_groq_or_openrouter_key
TAVILY_API_KEY=your_key
SCOUT_MODEL=meta-llama/llama-4-scout-17b-16e-instruct
STRATEGIST_MODEL=llama-3.3-70b-versatile
```

### 3. Usage

**Manual Run:**
```bash
python main.py run --ticker NVDA --reason "Semiconductor supply chain shift"
```

**Watcher Mode:**
```bash
python main.py watch --interval 3600
```

**View History:**
```bash
python main.py history
```

---

## 🛡 Security & Risk Disclaimer
This software is a research tool and **not financial advice**. Trading involves significant risk of loss. Backtest results are historical and do not guarantee future performance. Option simulation uses a simplified 5x leverage model and should be independently verified.

## 🛠 Maintenance & Performance
The engine includes several features for performance and robustness:
*   **Dynamic Context Scaling:** A custom `TokenBudgeter` uses `tiktoken` to maximize the research signal from SEC and news data while staying within model TPM limits.
*   **Rate-Limit Aware Research:** The academic research module (`ArxivClient`) includes global request throttling and exponential backoff to handle strict API usage guidelines.
*   **Caching Layer:** S&P 500 tickers, FRED Macro data, and technical indicators are cached in the `.cache/` directory to reduce API latency and credit usage.
*   **Robust Parsing:** Upgraded leg parsing and fallback mechanisms to handle various LLM formatting styles.
*   **Mercy Rule:** Intelligent score boosting to prevent the engine from getting stuck in infinite refinement loops.

## 🧪 Testing
A suite of unit tests is available to verify core logic:
```bash
# Run all tests
python -m unittest discover tests

# Run specific test modules
python -m unittest tests/unit/test_parsing.py
python -m unittest tests/unit/test_providers.py
```

---
*Developed for Quantitative Strategy Research v2.0*
