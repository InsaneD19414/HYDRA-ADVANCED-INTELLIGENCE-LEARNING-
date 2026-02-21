# 🚀 Crypto Multi-Agent Trading System

A professional-grade multi-agent AI system built with **LangGraph** and **Claude** for cryptocurrency market analysis, technical analysis, risk management, and trading strategy recommendations.

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│              ORCHESTRATOR (Claude Opus)                  │
│  • Parses intent & extracts target symbol               │
│  • Routes to relevant specialist agents                 │
│  • Synthesises all outputs → final response             │
└──────────┬──────────────────────────────────────────────┘
           │  fan-out (parallel)
    ┌──────┴───────────────────────────┐
    │                                  │
    ▼                                  ▼
┌────────────────┐         ┌─────────────────────┐
│ MARKET ANALYST │         │ TECHNICAL ANALYST   │
│ (Claude Sonnet)│         │ (Claude Sonnet)     │
│                │         │                     │
│ • Live prices  │         │ • RSI (14)          │
│ • Fear & Greed │         │ • MACD (12/26/9)    │
│ • Market cap   │         │ • Bollinger Bands   │
│ • Volume data  │         │ • EMA 20/50         │
│ • Sentiment    │         │ • Support/Resistance│
└───────┬────────┘         └──────────┬──────────┘
        │                             │
        ▼                             ▼
┌────────────────┐         ┌─────────────────────┐
│  RISK MANAGER  │         │  STRATEGY ADVISOR   │
│ (Claude Sonnet)│         │  (Claude Sonnet)    │
│                │         │                     │
│ • Position size│         │ • Entry/exit zones  │
│ • Kelly Crit.  │         │ • DCA plans         │
│ • Stop-losses  │         │ • Cycle analysis    │
│ • Portfolio exp│         │ • Final rec (BUY/   │
│ • Concentration│         │   SELL/HOLD)        │
└───────┬────────┘         └──────────┬──────────┘
        │                             │
        └─────────────┬───────────────┘
                      │  fan-in
                      ▼
          ┌───────────────────────┐
          │   ORCHESTRATOR        │
          │   (Synthesis)         │
          │                       │
          │  Formats final report │
          └───────────┬───────────┘
                      │
                      ▼
          ┌───────────────────────┐
          │   PERSISTENT MEMORY   │
          │                       │
          │  • LangGraph SQLite   │
          │    checkpointer       │
          │  • Custom memory DB   │
          │    (trades, prefs,    │
          │    session summaries) │
          └───────────────────────┘
```

---

## 📦 Installation

```bash
# 1. Clone / unzip the project
cd crypto_agents

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your API keys:
#   ANTHROPIC_API_KEY  (required)
#   COINGECKO_API_KEY  (optional — free tier works without)
#   TAVILY_API_KEY     (optional — for richer web search)
```

---

## 🚀 Usage

### Interactive Mode (recommended)
```bash
python main.py
python main.py --symbol ETH --portfolio 50000 --risk aggressive
```

### Single Query Mode
```bash
python main.py --query "Should I buy BTC right now?" --portfolio 10000
python main.py --query "Analyse SOL technically" --symbol SOL
```

### In-session commands
| Command | Effect |
|---|---|
| `symbol ETH` | Switch active coin to ETH |
| `portfolio 25000` | Update portfolio size |
| `risk aggressive` | Change risk tolerance |
| `history` | Show trade history & P&L |
| `memory` | Show current memory context |
| `quit` | Exit |

---

## 💬 Example Queries

```
Should I buy BTC right now?
Give me a full technical analysis on ETH/USD
What's the risk of putting 20% of my portfolio into SOL?
The market looks bearish, should I hedge?
Set up a DCA plan for Solana over 3 months
What's the current Fear & Greed index telling us?
Analyse AVAX for a swing trade setup
```

---

## 🗂️ Project Structure

```
crypto_agents/
├── main.py                    # CLI entry point
├── graph.py                   # LangGraph assembly & checkpointing
├── state.py                   # Shared AgentState TypedDict
├── requirements.txt
├── .env.example
│
├── agents/
│   ├── orchestrator.py        # Parse + Synthesise nodes
│   └── specialists.py         # 4 specialist agent nodes
│
├── tools/
│   └── crypto_tools.py        # All @tool functions
│
└── memory/
    └── memory_manager.py      # SQLite persistent memory
```

---

## 🔧 Supported Coins

BTC, ETH, SOL, BNB, XRP, ADA, DOGE, AVAX, DOT, LINK, MATIC, UNI, LTC, ATOM, NEAR, ARB, OP, SUI, INJ, TIA — and any CoinGecko-listed coin by its ID.

---

## 🔐 Risk Disclaimer

This system is for **educational and research purposes only**. Cryptocurrency markets are highly volatile. Nothing here constitutes financial advice. Always do your own research (DYOR) and never invest more than you can afford to lose.

---

## 🛠️ Extending the System

### Add a new specialist agent
1. Define a new node function in `agents/specialists.py`
2. Add the node to `graph.py` with `builder.add_node()`
3. Add routing in `route_to_specialists()`
4. Add tools in `tools/crypto_tools.py`

### Add on-chain data
Install `web3` and add a tool that calls The Graph or Dune Analytics.

### Add exchange integration
Add a tool using `ccxt` for live order book data and paper trading.
```python
pip install ccxt
```

### Enable LangSmith tracing
```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_key
```
