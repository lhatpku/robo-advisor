# 🧠 Robo Advisor with Risk, Portfolio, Investment & Trading Agents

This repository implements a **complete 5-step** intelligent, modular robo-advising platform built on
LLM-powered agents orchestrated with **LangGraph**.  
The system integrates conversational intent detection, questionnaire-based risk profiling,
portfolio optimization, fund selection, trading execution workflows, and a modern **Streamlit web interface**.

---

## 🏗 Architecture Overview

```
User
 └──> Entry Agent (ChatOpenAI)
       ├─ natural conversation
       ├─ detects user intent (risk, portfolio, investment, trading)
       ├─ directs to:
       │    ├─ Risk Agent  → equity setting OR questionnaire-based guidance
       │    ├─ Portfolio Agent → mean-variance optimizer
       │    ├─ Investment Agent → fund selection & analysis
       │    └─ Trading Agent → executable trading requests
       ↓
 ├──> Risk Agent (ChatOpenAI + Tool)
 │      ├─ handles direct equity setting commands
 │      ├─ runs 7 risk-profiling questions (when guidance requested)
 │      ├─ produces {"equity": x, "bond": 1-x}
 │      └─ writes recommendation to shared state
 │
 ├──> Portfolio Agent (ChatOpenAI + Tool)
 │      ├─ reads equity/bond split from risk output
 │      ├─ expands into detailed asset-class sleeves via **mean/variance optimization**
 │      ├─ allows user edits to λ (risk-aversion) and cash-reserve inputs
 │      ├─ outputs an **asset-class portfolio dictionary**
 │      └─ routes to Investment Agent for fund selection
 │
 ├──> Investment Agent (ChatOpenAI + Fund Analysis)
 │      ├─ selects ETFs/funds for each asset class
 │      ├─ provides 4 selection criteria (Balanced, Low Cost, High Performance, Low Risk)
 │      ├─ analyzes funds using Yahoo Finance API
 │      ├─ allows user review and editing of selections
 │      └─ outputs **investment portfolio with tickers**
 │
 ├──> Trading Agent (ChatOpenAI + Rebalancing Engine)
 │      ├─ generates executable trading requests
 │      ├─ uses demo scenarios for realistic testing
 │      ├─ implements tax-aware rebalancing optimization
 │      ├─ outputs **simple trading table** (ticker, action, price, shares)
 │      └─ provides execution summary
 │
 └──> Reviewer Agent (ChatOpenAI)
        ├─ reviews completed phases and user progress
        ├─ provides final recommendations and next steps
        ├─ orchestrates flow between agents
        └─ handles completion and routing decisions
```

---

## 🧩 Key Components

| Module | File | Purpose |
|--------|------|----------|
| **Entry** | `entry_agent.py` | Main orchestrator, intent detection, routing |
| **Risk** | `risk/risk_agent.py` | Risk profiling questionnaire and guidance |
| | `risk/risk_manager.py` | Risk calculation tools and question management |
| **Portfolio** | `portfolio/portfolio_agent.py` | Portfolio optimization conversation |
| | `portfolio/portfolio_manager.py` | Mean-variance optimization tools |
| **Investment** | `investment/investment_agent.py` | Fund selection and analysis |
| | `investment/fund_analyzer.py` | Yahoo Finance API integration |
| **Trading** | `trading/trading_agent.py` | Trading request generation |
| | `trading/portfolio_trading.py` | Portfolio-to-trades conversion |
| | `trading/rebalance.py` | Tax-aware rebalancing optimization |
| | `trading/config.py` | Configuration and assumptions |
| | `trading/demo_scenarios.json` | Demo trading scenarios |
| **Reviewer** | `reviewer_agent.py` | Final review, recommendations, and flow orchestration |
| **UI** | `streamlit_app.py` | Modern web interface with real-time visualization |
| **Core** | `state.py` | Shared TypedDict state |
| | `app.py` | Main LangGraph orchestration |

---

## ⚙️ Setup & Run

### 1. Environment Setup
```bash
# Create conda environment
conda create -n roboadvisor python=3.11
conda activate roboadvisor

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Variables
Create a `.env` file in the project root:
```
OPENAI_API_KEY=sk-your-key
OPENAI_MODEL=gpt-4o-mini
OPENAI_TEMPERATURE=0.2
```

### 3. Run the Application

#### Option A: Streamlit Web Interface (Recommended)
```bash
streamlit run streamlit_app.py
```
The app will open in your browser at `http://localhost:8501`

#### Option B: Command Line Interface
```bash
python app.py
```

#### Option C: Gradio Interface (Legacy)
```bash
python gradio_app.py
```

---

## 🎨 Streamlit Web Interface Features

### **Interactive Chat Interface**
- Real-time back-and-forth communication with the AI robo-advisor
- Clean message input with automatic form submission
- Displays latest AI response prominently

### **Real-time Data Visualization**
- **Risk Assessment**: Bar charts for equity/bond allocation + collapsible questionnaire answers
- **Portfolio**: Interactive pie charts + detailed allocation tables with weights
- **Investment**: Comprehensive fund selection table with tickers, weights, and selection criteria
- **Trading**: Clean table format for trading requests with execution details

### **Process Status Tracking**
- Visual progress indicators for each of the 4 main phases
- Color-coded status indicators (complete/pending/not started)
- Progress bar in sidebar showing completion percentage

### **Advanced UI Features**
- **Collapsible Message History**: Expandable conversation log (shows last 15 messages)
- **Reset Functionality**: One-click reset button to clear state and restart
- **Reactive Design**: Sections only appear when relevant data exists
- **Modern Styling**: Professional interface with custom CSS and responsive layout

---

## 🧠 Agent Behaviors

### Entry Agent
- Initializes with a greeting and clear user choices
- Routes users to appropriate agents based on their intent
- Routes dynamically:
  - → **Risk Agent** for any risk-related requests (equity setting or guidance)
  - → **Portfolio Agent** if ready to invest
  - → **Investment Agent** after portfolio optimization
  - → **Trading Agent** after fund selection
- Always preserves existing state when moving forward

### Risk Agent
- **Handles all risk-related functionality** including:
  - Direct equity setting commands: *"set equity 0.6"*, *"set equity to 60%"*
  - Risk guidance through 7-question questionnaire
  - Equity reset commands: *"reset equity"*, *"clear equity"*
- Conducts a comprehensive 7-question risk-profiling interview when guidance requested
- Supports "why" clarifications per question
- On completion:
  - Writes equity/bond mix into `state["risk"]`
  - Sets `done=True` and `awaiting_input=False`
  - Routes to **Reviewer Agent** for next steps

### Portfolio Agent ✅
- Reads mean/covariance data from `portfolio/config/asset_stats.xlsx`
- Runs a **mean-variance optimizer** producing 12 asset classes:
  - Equity sleeves (large/small growth/value, developed/emerging)
  - Bond sleeves (short/mid/long-term treasuries, corporates, TIPS, cash)
- Lets user adjust λ (5–20 typical) and cash reserve (3–6%)
- Outputs **asset-class portfolio dictionary**
- Routes to **Reviewer Agent** for next steps

### Investment Agent ✅
- Presents 4 fund selection criteria:
  - **Balanced**: Best Sharpe ratio
  - **Low Cost**: Lowest expense ratio
  - **High Performance**: Highest returns
  - **Low Risk**: Lowest volatility
- Analyzes funds using Yahoo Finance API:
  - Performance metrics (returns, volatility, Sharpe ratio, beta)
  - Management metrics (expense ratio, AUM, fund family)
- Allows user review and editing of fund selections
- Excludes cash from fund analysis (uses "sweep_cash")
- Outputs **investment portfolio with tickers**
- Routes to **Reviewer Agent** when user says "proceed"

### Trading Agent ✅
- Generates executable trading requests from investment portfolio
- Uses demo scenarios for realistic testing:
  - 6 predefined scenarios with different account values and holdings
  - Conservative, Balanced, Growth, Young Professional, Wealthy Conservative, Active Trader
- Implements sophisticated rebalancing:
  - Tax-aware optimization (short/long-term capital gains)
  - Full covariance risk model
  - Soft tax cap with increasing penalties
  - Cash sweep band management
  - Two-stage integerization for whole shares
- Outputs **simple trading table**:
  ```
  | Ticker | Action | Unit Price | Shares |
  |--------|--------|------------|--------|
  | VUG    | BUY    | $245.50    | 100    |
  | VTV    | SELL   | $180.25    | 50     |
  ```

### Reviewer Agent ✅
- **Central orchestrator** that manages flow between all agents
- Reviews completed phases and provides recommendations
- Handles routing decisions based on user progress
- Provides final completion options and next steps
- Manages the overall conversation flow and state transitions

---

## 🧭 Complete User Flow

```
┌───────────────────────────────┐
│ User launches conversation    │
└──────────────┬────────────────┘
               ▼
        Entry Agent
         │ greet + choices
         ▼
 ┌───────────────────────────────┐
 │ "set equity 0.6" → Risk Agent │
 │ "use guidance" → Risk Agent   │
 │ "proceed" → Portfolio Agent  │
 └───────────────────────────────┘
               ▼
        Risk Agent
         │ handles equity setting OR
         │ 7-question questionnaire
         │ computes 60% / 40%
         │ sets done=True
         ▼
     Reviewer Agent
         │ reviews progress
         │ routes to Portfolio
         ▼
     Portfolio Agent
         │ asks λ & cash reserve
         │ runs mean-variance optimization
         │ outputs asset-class portfolio
         │ sets done=True
         ▼
     Reviewer Agent
         │ reviews progress
         │ routes to Investment
         ▼
     Investment Agent
         │ presents fund selection criteria
         │ analyzes funds via Yahoo Finance
         │ allows review/edit of selections
         │ outputs investment portfolio
         │ sets done=True when user says "proceed"
         ▼
     Reviewer Agent
         │ reviews progress
         │ routes to Trading
         ▼
     Trading Agent
         │ shows demo scenarios
         │ generates trading requests
         │ outputs simple trading table
         │ sets done=True
         ▼
     Reviewer Agent
         │ final review and completion
         │ provides next steps
         ▼
      (Ready for execution)
```

---

## 🧪 Example Complete Flow

### Option A: Direct Equity Setting
> **User:** set equity 0.6  
> **AI:** Perfect! I've set your allocation to **60% equity / 40% bonds**...  
> *(Risk Agent directly sets equity allocation)*  
> **User:** proceed  
> *(Reviewer Agent routes to Portfolio Agent)*

### Option B: Risk Guidance
> **User:** use guidance  
> **AI:** I'll help you determine your optimal asset allocation through a risk assessment...  
> *(Risk Agent runs full 7-question flow → recommends 60% equity / 40% bonds)*  
> **User:** proceed  
> *(Reviewer Agent routes to Portfolio Agent)*  
> **AI:** Defaults λ = 1.0, cash = 0.05 — say "set lambda to 1 and cash to 0.03 run"  
> **User:** set lambda as 1 and cash as 0.03 run  
> *(Optimizer builds portfolio)*  

| Asset Class | Weight |
|--------------|-------:|
| Mid-term Treasury | 29.72% |
| TIPS | 29.72% |
| Corporate Bond | 22.56% |
| Emerging Market Equity | 11.00% |
| Cash | 3.00% |
| Large Cap Value | 1.90% |
| Small Cap Growth | 1.32% |
| **Total** | **100%** |

> **User:** proceed  
> *(Reviewer Agent routes to Investment Agent)*  
> **AI:** Choose your fund selection criteria: Balanced, Low Cost, High Performance, or Low Risk  
> **User:** balanced  
> *(Fund analysis and selection)*  

| Asset Class | Weight | Ticker | Selection Reason |
|-------------|--------|--------|------------------|
| Mid-term Treasury | 29.72% | VGIT | Low cost index fund |
| TIPS | 29.72% | VTEB | Diversified bond exposure |
| Corporate Bond | 22.56% | VCIT | Best Sharpe ratio |
| Emerging Market Equity | 11.00% | VWO | Emerging market exposure |
| Cash | 3.00% | sweep_cash | Sweep Account |

> **User:** proceed  
> *(Reviewer Agent routes to Trading Agent)*  
> **AI:** Welcome to the Trading Module! Select a demo scenario (1-6)...  
> **User:** 1  
> *(Trading requests generated)*  

| Ticker | Action | Unit Price | Shares |
|--------|--------|------------|--------|
| VGIT | BUY | $50.35 | 100 |
| VTEB | BUY | $45.20 | 150 |
| VCIT | SELL | $48.75 | 50 |

**Total Trades:** 3  
**Buy Orders:** 2  
**Sell Orders:** 1  
**Net Cash Flow:** $15,000

---

## 🔮 Implementation Status

| Phase | Status | Description |
|--------|--------|-------------|
| ✅ **Step 1 – Risk Onboarding** | **Complete** | Risk-profiling and allocation summary finished |
| ✅ **Step 2 – Portfolio Agent** | **Complete** | Asset-class optimizer with mean-variance optimization |
| ✅ **Step 3 – Investment Agent** | **Complete** | Fund selection with Yahoo Finance analysis |
| ✅ **Step 4 – Trading Agent** | **Complete** | Tax-aware rebalancing with demo scenarios |
| ✅ **Step 5 – Reviewer Agent** | **Complete** | Central orchestrator and flow management |
| ✅ **Step 6 – Streamlit UI** | **Complete** | Modern web interface with real-time visualization |
| 🚀 **Step 7 – Production Ready** | **Vision** | Real market data, custodian integration, monitoring |

---

## 🧰 Technical Features

### Configuration System
- **Centralized config** (`trading/config.py`) with all assumptions and parameters
- **Easy toggle** between assumed data and real market data
- **Comprehensive documentation** of all assumptions and production recommendations

### State Management
- **Clean AgentState** with only shared fields
- **Internal state** for agent-specific data (demo scenarios, preferences)
- **Type-safe** state management with TypedDict
- **Status tracking** for each agent phase (done, awaiting_input)

### Fund Analysis
- **Yahoo Finance integration** for real-time fund data
- **Performance metrics**: returns, volatility, Sharpe ratio, beta, max drawdown
- **Management metrics**: expense ratio, AUM, fund family, inception date
- **Data quality assessment** and fallback handling

### Trading Optimization
- **Tax-aware rebalancing** with lot-aware cost calculation
- **Full covariance risk model** for accurate tracking error
- **Soft tax cap** with increasing penalty functions
- **Cash sweep band** management
- **Two-stage integerization** for whole-share constraints
- **Demo scenarios** for realistic testing

### Web Interface
- **Real-time visualization** with Plotly charts and tables
- **Reactive design** that shows/hides sections based on data availability
- **Message history** with collapsible conversation log
- **Status tracking** with visual progress indicators
- **Reset functionality** for easy testing and restart

### Error Handling
- **Comprehensive error handling** with user-friendly messages
- **Debug information** for development (removed from production output)
- **Graceful fallbacks** for missing data or API failures
- **Unicode encoding fixes** for cross-platform compatibility

---

## 🚀 Production Roadmap

### Immediate (Next Steps)
1. **Replace synthetic covariance** with real market data
2. **Implement real-time fund data** feeds
3. **Add custodian integration** for trade execution
4. **Implement portfolio monitoring** and rebalancing triggers

### Future Enhancements
1. **ESG and sustainability** criteria
2. **Factor models** for better risk attribution
3. **Currency hedging** for international assets
4. **Tax-loss harvesting** optimization
5. **Multi-account management** and aggregation
6. **Performance attribution** and reporting

---

## 🛠 Developer Notes

- **Modular architecture**: Each agent is self-contained with clear interfaces
- **Easy extension**: Add new agents by updating routing in `app.py`
- **Configuration-driven**: All assumptions centralized in config files
- **Type safety**: Full typing with Pydantic models and TypedDict
- **Testing**: Demo scenarios provide realistic testing without real data
- **Documentation**: Comprehensive docstrings and configuration notes
- **UI/UX**: Modern Streamlit interface with real-time updates and visualization

---

## 📊 Asset Classes Supported

| Category | Asset Classes |
|----------|---------------|
| **Equity** | Large Cap Growth, Large Cap Value, Small Cap Growth, Small Cap Value, Emerging Market Equity, Developed Market Equity |
| **Fixed Income** | Mid-term Treasury, Long-term Treasury, Short-term Treasury, TIPS, Corporate Bond |
| **Cash** | Sweep Account (for trading reserve) |

---

## 🎯 Key Metrics Tracked

### Performance Metrics
- Total Return (1Y, 3Y, 5Y)
- Annualized Return
- Volatility (Annualized)
- Sharpe Ratio
- Maximum Drawdown
- Beta (vs S&P 500)

### Management Metrics
- Expense Ratio
- Assets Under Management (AUM)
- Fund Family
- Management Company
- Inception Date
- Minimum Investment

### Trading Metrics
- Execution Priority
- Tax Implications
- Risk Metrics
- Cash Flow Impact
- Tracking Error

---

## 🧪 Testing

The repository includes comprehensive user flow testing:

```bash
# Run all user flow tests
python userflowtesting/test_suite.py

# Run individual tests
python userflowtesting/test_comprehensive_risk_flow.py
python userflowtesting/test_portfolio_to_investment.py
python userflowtesting/test_simple_completion.py
```

**Note**: Tests may show Unicode encoding warnings on Windows - this is a display issue and doesn't affect functionality.

---

*This robo-advisor represents a complete end-to-end wealth management solution, from risk assessment to trade execution, built with modern AI, optimization techniques, and a beautiful web interface.*