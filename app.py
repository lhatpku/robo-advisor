# app.py
from __future__ import annotations
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from risk.risk_agent import RiskAgent
from portfolio.portfolio_agent import PortfolioAgent
from investment.investment_agent import InvestmentAgent
from trading.trading_agent import TradingAgent
from entry_agent import EntryAgent
from state import AgentState

# ---------------------------
# Main application logic
# ---------------------------

def build_graph(llm: ChatOpenAI):
    builder = StateGraph(AgentState)
    
    # Create agent instances
    entry_agent = EntryAgent(llm)
    risk_agent = RiskAgent(llm)
    portfolio_agent = PortfolioAgent(llm)
    investment_agent = InvestmentAgent(llm)
    trading_agent = TradingAgent(llm)

    builder.add_node("robo_entry", entry_agent.step)
    builder.add_node("risk_agent", risk_agent.step)
    builder.add_node("portfolio_agent", portfolio_agent.step)
    builder.add_node("investment_agent", investment_agent.step)
    builder.add_node("trading_agent", trading_agent.step)

    builder.set_entry_point("robo_entry")

    # Route to risk, portfolio, investment, or trading.
    builder.add_conditional_edges("robo_entry", entry_agent.router, {
        "risk_agent": "risk_agent",
        "portfolio_agent": "portfolio_agent",
        "investment_agent": "investment_agent",
        "trading_agent": "trading_agent",
        "END": END
    })

    # All agents loop back to entry agent when done.
    builder.add_edge("risk_agent", "robo_entry")
    builder.add_edge("portfolio_agent", "robo_entry")
    builder.add_edge("investment_agent", "robo_entry")
    builder.add_edge("trading_agent", "robo_entry")

    # Keep it simple: no checkpointer required.
    return builder.compile()

# ---------------------------
# Run (simple REPL)
# ---------------------------
if __name__ == "__main__":
    load_dotenv()  # expects OPENAI_API_KEY; optional OPENAI_MODEL / OPENAI_TEMPERATURE

    # Force JSON output from entry agent to avoid parsing issues.
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.2")),
    )

    graph = build_graph(llm)

    # Initial state (note: no extra fields beyond what's already defined in AgentState)
    state: AgentState = {
        "messages": [],
        "q_idx": 0,
        "answers": {},
        "done": False,
        "risk": None,
        "awaiting_input": False,   # used by risk_agent
        "intent_to_risk": False,  # set by entry agent
        "intent_to_portfolio": False,
        "intent_to_investment": False,
        "intent_to_trading": False,
        "entry_greeted": False,
        "portfolio": None,
        "investment": None
    }

    # --- INITIAL TICK to produce greeting ---
    state = graph.invoke(state)
    ai_msgs = [m for m in state["messages"] if m.get("role") == "ai"]
    if ai_msgs:
        print(ai_msgs[-1]["content"])

    # --- normal REPL ---
    while True:
        user_in = input("> ")
        state["messages"].append({"role": "user", "content": user_in})
        state = graph.invoke(state)
        ai_msgs = [m for m in state["messages"] if m.get("role") == "ai"]
        if ai_msgs:
            print(ai_msgs[-1]["content"])