## ✅ Robo Advisor with Advice and Investment Agent

# 🧠 Robo-Advising Agent Framework — Step 2: Investment Flow Extension

This repository now implements **2 steps** of an intelligent, modular robo-advising platform built on
LLM-powered agents orchestrated with **LangGraph**.  
The system integrates conversational intent detection, questionnaire-based risk profiling,
and a partially completed investment-optimization workflow.

---

## 🏗 Architecture Overview

```
User
 └──> Robo-Advisor Agent (ChatOpenAI)
       ├─ natural conversation
       ├─ detects if user wants to set equity manually, request guidance, or proceed to invest
       ├─ directs to:
       │    ├─ Advice Agent  → questionnaire-based guidance
       │    └─ Investment Agent → mean-variance optimizer
       ↓
 ├──> Advice Agent (ChatOpenAI + Tool)
 │      ├─ runs 7 risk-profiling questions
 │      ├─ produces {"equity": x, "bond": 1-x}
 │      └─ writes recommendation to shared state
 │
 └──> Investment Agent (ChatOpenAI + Tool)
        ├─ reads equity/bond split from Advice output
        ├─ expands into detailed asset-class sleeves via **mean/variance optimization**
        ├─ allows user edits to λ (risk-aversion) and cash-reserve inputs
        ├─ outputs an **asset-class portfolio dictionary**
        └─ (Next step → ETF/fund analysis to map sleeves to real securities)
```

---

## 🧩 Key Components

| File | Purpose |
|------|----------|
| **`advice/questions.py`** | Questionnaire text and “why” guidance. |
| **`advice/general_investing.py`** | Tool to compute base equity/bond allocation. |
| **`advice/advice_agent.py`** | Advice Agent logic and tool invocation. |
| **`investment/optimizer_tool.py`** | `@tool("mean_variance_optimizer")` that reads mean & covariance from Excel and solves the optimization. |
| **`investment/investment_agent.py`** | Conversational Investment Agent that manages λ / cash reserve and triggers optimization. |
| **`state.py`** | Shared TypedDict state accessible to all agents. |
| **`app.py`** | Entry graph (Robo-Advisor → Advice → Investment). |

---

## ⚙️ Setup & Run

### 1. Env Set up
```bash
conda create -n roboadvisor python=3.11
conda activate roboadvisor
pip install langchain langgraph langchain-openai python-dotenv numpy pandas openpyxl
python app.py
```
### 2. Environment Variables
Create a `.env` file in the project root:
```
OPENAI_API_KEY=sk-your-key
OPENAI_MODEL=gpt-4o-mini
OPENAI_TEMPERATURE=0.2
```

### 3. Run
```bash
python app.py
```
---

## 🧠 Agent Behaviors

### Robo-Advisor Agent
- Initializes with a greeting and clear user choices.  
- Understands commands such as:  
  *“set equity 0.6”*, *“use guidance”*, *“reset equity”*, *“proceed to invest”*.  
- Delegates dynamically:
  - → **Advice Agent** if guidance requested.  
  - → **Investment Agent** if ready to invest.  
- Always preserves existing equity in state when moving forward.

### Advice Agent
- Conducts a 7-question risk-profiling interview.  
- Supports “why” clarifications per question.  
- On completion:
  - Writes equity/bond mix into `state["advice"]`.  
  - Clears `intent_to_advise` so routing returns to entry.  
  - User can then review or proceed to investment.

### Investment Agent  (🧩 In-Progress)
- Completed:  
  - Reads mean / covariance data from `investment/config/mu_cov.xlsx`.  
  - Runs a **mean-variance optimizer** producing 12 asset classes:  
    - Equity sleeves (large/small growth/value, developed/emerging)  
    - Bond sleeves (short/mid/long-term treasuries, corporates, TIPS, cash)  
  - Lets user adjust λ (5–20 typical) and cash reserve (3–6 %).  
- Pending:  
  - ETF/fund selection analysis (fund-level optimizer + LLM due-diligence).  
  - Portfolio-to-security mapping and monitoring.  

---

## 🧭 Example User Flow

```
┌───────────────────────────────┐
│ User launches conversation    │
└──────────────┬────────────────┘
               ▼
        Robo-Advisor Agent
         │ greet + choices
         ▼
 ┌───────────────────────────────┐
 │ “set equity” → Review prompt  │
 │ “use guidance” → Advice Agent │
 │ “proceed” → Investment Agent  │
 └───────────────────────────────┘
               ▼
        Advice Agent
         │ 7 questions
         │ computes 15 % / 85 %
         ▼
   Returns to Robo-Advisor
         │ review or proceed
         ▼
     Investment Agent
         │ asks λ & cash reserve
         │ runs mean-variance optimization
         │ outputs asset-class portfolio
         ▼
      (next step: ETF selection)
```

---

### 🧪 Example Transcript (from recent test)

> **User:** set equity  
> **AI:** Please provide your desired equity weight to set.  
> **User:** 0.6  
> **AI:** Updated: 60 % equity / 40 % bonds. Would you like to review or proceed?  
> **User:** review  
> **AI:** Current mix … Type `use guidance` for questionnaire.  
> **User:** guidance  
> *(Advice Agent runs full 7-question flow → recommends 15 % equity / 85 % bonds)*  
> **User:** proceed  
> *(Investment Agent engages)*  
> **AI:** Defaults λ = 1.0, cash = 0.05 — say “set lambda to 1 and cash to 0.03 run”  
> **User:** set lambda as 1 and cash as 0.03 run  
> *(Optimizer builds portfolio — sample output below)*  

| Asset Class | Weight |
|--------------|-------:|
| Mid-term Treasury | 29.72 % |
| TIPS | 29.72 % |
| Corporate Bond | 22.56 % |
| Emerging Market Equity | 11.00 % |
| Cash | 3.00 % |
| Large Cap Value | 1.90 % |
| Small Cap Growth | 1.32 % |
| ... | ... |
| **Total** | **100 %** |

---

## 🔮 Roadmap

| Phase | Focus | Description |
|--------|--------|-------------|
| ✅ **Step 1 – Advice Onboarding** | Completed | Risk-profiling and allocation summary finished. |
| 🧩 **Step 2 – Investment Agent** | *In Progress* | Asset-class optimizer done; ETF/fund analysis next. |
| ⏳ **Step 3 – Trading Agent** | Planned | Translate portfolios to trades and send to custodians. |
| 🚀 **Step 4 – Unified Robo-Advisor** | Vision | Fully autonomous, multi-agent wealth-advisory pipeline. |

---

## 🧰 Developer Notes

- State now includes:  
  `intent_to_advise`, `intent_to_investment`, `advice`, `investment`, and standard `messages`.  
- Each node operates **only on user turns**, avoiding LangGraph recursion.  
- The Investment Agent uses modular tools (`mean_variance_optimizer`, future `fund_selector`).  
- Extend easily:
  - Add new endpoints under `/investment` or `/trading`.  
  - Update router conditions in `app.py`.

---