# 🧠 Robo-Advising Agent Framework — Step 1: Advice Onboarding Flow

This repository implements **Step 1** of an intelligent, modular robo-advising platform built on
LLM-powered agents orchestrated with **LangGraph**.  
The current milestone focuses on the **onboarding and risk-profiling experience**, handled by
two conversational agents:

1. **Robo-Advisor Agent** (entry point)  
   — Handles general investor conversations and determines when a user is ready to begin onboarding.  
   — Provides a “heads-up” explaining that a short questionnaire is required to assess the user’s
     risk profile and asks for consent to proceed.  

2. **Advice Agent**  
   — Conducts the 7-question risk-profiling questionnaire.  
   — Handles clarifying prompts (“why?”) using question-specific guidance text.  
   — When all answers are collected, calls the **`general_investing_advice`** tool to
     generate an equity/bond allocation recommendation and summarizes results conversationally.

---

## 🏗 Architecture Overview

```
User
 └──> Robo-Advisor Agent (ChatOpenAI)
       ├─ natural dialogue
       ├─ detects investment-advice intent
       ├─ sends “heads-up + proceed?” confirmation
       │    ├─ if no  → continue chatting
       │    └─ if yes → sets `intent_to_advise=True`
       ↓
 └──> Advice Agent (ChatOpenAI + Tool)
       ├─ sequentially asks 7 risk-profiling questions
       ├─ supports “why?” guidance per question
       ├─ records structured multiple-choice answers
       └─ calls `general_investing_advice_tool`
             → returns {"equity": x, "bond": 1-x}
             → summarizes to the user
```

Each agent runs as a **LangGraph node**, invoked sequentially:

| Node | Role | Exit condition |
|------|------|----------------|
| `robo_entry` | Determine intent and confirmation | Sets `intent_to_advise=True` |
| `advice_agent` | Run questionnaire → compute recommendation | Sets `done=True` |

---

## 🧩 Key Components

| File | Purpose |
|------|----------|
| **`advice/questions.py`** | Exact questionnaire text, multiple-choice options, and “Why do we ask?” guidance. |
| **`advice/general_investing.py`** | Defines the `@tool("general_investing_advice")` function (placeholder allocator). |
| **`advice/advice_agent.py`** | Implements the Advice Agent logic, deterministic parsing, and tool invocation. |
| **`app.py`** | Entry script: defines the Robo-Advisor Agent, LangGraph routing, and command-line REPL. |
| **`.env`** | Stores your `OPENAI_API_KEY` and optional settings (`OPENAI_MODEL`, `OPENAI_TEMPERATURE`). |

---

## ⚙️ Setup & Run

### 1. Environment
```bash
# create environment
conda create -n roboadvisor python=3.11
conda activate roboadvisor

# install deps
pip install langchain langgraph langchain-openai python-dotenv
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

You’ll see:
```
Hi! I’m your robo-advisor. How can I help with your finances today?
> I want investment advice
To give you meaningful investment advice, we’ll complete a brief questionnaire to assess your risk profile (7 quick questions).

Would you like to proceed now? (yes/no)
> yes
Great — we’ll go through a short 7-question risk profile so I can tailor your allocation.
...
```

---

## 🧠 Agent Behaviors

### Robo-Advisor Agent
- Converses freely with the investor using an OpenAI LLM.
- Detects if the user intends to start onboarding.
- Provides a **heads-up confirmation** step before delegating to Advice Agent.
- Sets `state["intent_to_advise"] = True` when user confirms.

### Advice Agent
- Reads question text/options/guidance verbatim from `questions.py`.
- Uses deterministic parsing (`1`, `second`, or option text fragments).
- Handles “why” requests with contextual guidance.
- On completion, calls `general_investing_advice_tool` and formats a summary.

### `general_investing_advice_tool`
- Receives all 7 answers.
- Placeholder logic computes an equity/bond mix based on horizon and withdrawals.
- Replace with your actual allocation engine later.

---

## 🔮 Roadmap

| Phase | Focus | Description |
|--------|--------|-------------|
| ✅ **Step 1 – Advice Onboarding** | Completed | Robo-Advisor Agent + Advice Agent for risk-profiling and allocation summary. |
| 🔄 **Step 2 – Investment Agent** | Next | Connects with internal portfolio-construction engine (multi-goal optimizer, risk-adjusted model portfolios). |
| ⏳ **Step 3 – Trading Agent** | Future | Translates portfolio changes into executable trades, integrates with custodians or recordkeepers. |
| 🚀 **Step 4 – Unified Robo-Advisor** | Vision | End-to-end agentic system coordinating conversation, planning, portfolio construction, and execution. |

---

## 🧰 Developer Notes

- Each agent is **state-driven**. State fields:
  - `messages`: chronological message list  
  - `intent_to_advise`: router flag for handoff  
  - `q_idx`, `answers`, `awaiting_input`, `done`: questionnaire progress tracking
- Agents only modify state on **user turns** (prevents infinite recursion in LangGraph).
- The architecture is easily extendable:
  - Add a new agent node (e.g., `investment_agent_step`, `trading_agent_step`).
  - Extend the router to delegate by `intent_to_invest` or `intent_to_trade`.
- To integrate APIs (portfolio data, transaction services), register new `@tool`s
  and bind them in `_finalize_with_tool_and_llm`.

---

## 🧪 Example Extension Skeleton

```python
# advice/investment_agent.py (future)
def investment_agent_step(state: AgentState, llm: ChatOpenAI):
    # Example: run optimization or retrieve model portfolio
    ...
```

Then in `app.py`:
```python
builder.add_node("investment_agent", lambda s: investment_agent_step(s, llm))
def router_from_entry(state):
    if state.get("intent_to_advise"):
        return "advice_agent"
    if state.get("intent_to_invest"):
        return "investment_agent"
    return END
```
