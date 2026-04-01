# StrategyIdeationEngine (StrategyAgent)

### *Institutional-Grade AI Agent for Quantitative Strategy Research*

The **StrategyIdeationEngine** is a sophisticated, self-correcting AI agent designed to generate, critique, and empirically validate mid-term (days to weeks) trading hypotheses. Unlike standard LLM "narrative" generators, this engine uses a **Three-Tier Intelligence Architecture** to force every idea through a rigorous internal "Red Team" and a high-fidelity historical backtest.

---

## 🚀 Core Architecture: The Tiered Pipeline

The engine operates on a multi-agent orchestration designed to reduce hallucinations and maximize alpha quality:

1.  **Phase 1: Event Triggering:** Manual ticker research or an automated scanner bot.
2.  **Phase 2: Ground Truth Ingestion:** Unified data gathering from **yfinance**, **Alpha Vantage**, and **FRED**. 
3.  **Phase 3: The Scout (Tier 1):** Powered by fast models (e.g., Llama-4 17B). Condenses thousands of tokens from **ArXiv papers**, **Tavily news**, and **SEC filings** into a distilled **Numerical Fact Sheet**.
4.  **Phase 4: The Strategist (Tier 2):** Powered by high-reasoning models (e.g., Llama-3.3 70B).
    *   **Thesis Architect:** Develops a multi-leg economic rationale with citations.
    *   **Trade Architect:** Translates the thesis into concrete math rules (using 130+ indicators via `pandas_ta`).
5.  **Phase 5: Double Validation & Self-Correction:**
    *   **Adversarial Critique (Tier 3):** An internal Red Team LLM analyzes the strategy for logical flaws and horizon-specific risks.
    *   **High-Fidelity Backtest:** A quantitative module that simulates the **entire strategy portfolio** as a co-integrated unit, including **transaction costs and slippage**.
    *   **Sector-Aware Benchmarking:** Automatically identifies the relevant sector ETF (e.g., XLK, XLE) to isolate true Alpha from Sector Beta.
    *   **Refinement Loop:** If scores are low (< 0.7), the engine passes a **Self-Correction Plan** back to the Strategist for up to 3 polish cycles.
6.  **Phase 6: Professional Export:** Generates detailed Markdown reports.

---

## 🚦 Getting Started

### 1. Prerequisites
*   Python 3.10+
*   API Keys for your chosen LLM providers: **Groq**, **OpenAI**, **Anthropic**, or **Google Gemini**.
*   Optional Data APIs: **Tavily** (Search), **FRED** (Macro), **Alpha Vantage** (Technicals).

### 2. Configuration
Create a `.env` file in the root directory (see `.env.example`). You can now mix and match models from different providers for each tier:

```env
# API Keys (Set the ones you intend to use)
GROQ_API_KEY=your_groq_key
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GEMINI_API_KEY=your_gemini_key

# Model Tiering (Supports prefixes: groq/, openai/, anthropic/, gemini/)
SCOUT_MODEL=groq/llama-3.1-8b-instant
STRATEGIST_MODEL=openai/gpt-4o
CRITIC_MODEL=anthropic/claude-3-5-sonnet

# Data APIs
TAVILY_API_KEY=your_key
FRED_API_KEY=your_key
ALPHA_VANTAGE_API_KEY=your_key
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

## 🛠 Operational Robustness
*   **Dynamic Context Scaling:** Uses a custom `TokenBudgeter` with `tiktoken` to maximize research signal while staying within model TPM limits.
*   **Caching Layer:** S&P 500 tickers, FRED Macro data, and technical indicators are cached in the `.cache/` directory.
*   **DSPy Optimization:** Refactored into a unified `dspy.Module` for programmatic prompt optimization and self-correction.
*   **Sector-Aware Alpha:** Automatically maps tickers to their respective Sector ETFs to calculate true Alpha.

## 🧪 Testing
```bash
# Run all tests
python -m pytest tests
```

## 🛡 Security & Risk Disclaimer
This software is a research tool and **not financial advice**. Trading involves significant risk of loss. Backtest results are historical and do not guarantee future performance.

---
*Developed for Quantitative Strategy Research v2.0*
