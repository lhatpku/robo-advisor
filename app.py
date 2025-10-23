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
from reviewer_agent import ReviewerAgent
from state import AgentState

# ---------------------------
# Main application logic
# ---------------------------

def build_graph(llm: ChatOpenAI):
    builder = StateGraph(AgentState)
    
    # Create agent instances
    entry_agent = EntryAgent(llm)
    reviewer_agent = ReviewerAgent(llm)
    risk_agent = RiskAgent(llm)
    portfolio_agent = PortfolioAgent(llm)
    investment_agent = InvestmentAgent(llm)
    trading_agent = TradingAgent(llm)

    builder.add_node("robo_entry", entry_agent.step)
    builder.add_node("reviewer_agent", reviewer_agent.step)
    builder.add_node("risk_agent", risk_agent.step)
    builder.add_node("portfolio_agent", portfolio_agent.step)
    builder.add_node("investment_agent", investment_agent.step)
    builder.add_node("trading_agent", trading_agent.step)

    builder.set_entry_point("robo_entry")

    # Route from entry agent to specific agents only
    builder.add_conditional_edges("robo_entry", entry_agent.router, {
        "risk_agent": "risk_agent",
        "portfolio_agent": "portfolio_agent",
        "investment_agent": "investment_agent",
        "trading_agent": "trading_agent",
        "__end__": END
    })

    # Route from reviewer agent back to appropriate agents or end
    builder.add_conditional_edges("reviewer_agent", reviewer_agent.router, {
        "risk_agent": "risk_agent",
        "portfolio_agent": "portfolio_agent",
        "investment_agent": "investment_agent",
        "trading_agent": "trading_agent",
        "robo_entry": "robo_entry",
        "__end__": END
    })

    # All agents route to reviewer when done or end when waiting
    builder.add_conditional_edges("risk_agent", risk_agent.router, {
        "reviewer_agent": "reviewer_agent",
        "risk_agent": "risk_agent",
        "__end__": END
    })
    
    builder.add_conditional_edges("portfolio_agent", portfolio_agent.router, {
        "reviewer_agent": "reviewer_agent",
        "__end__": END
    })
    
    builder.add_conditional_edges("investment_agent", investment_agent.router, {
        "reviewer_agent": "reviewer_agent",
        "__end__": END
    })
    
    builder.add_conditional_edges("trading_agent", trading_agent.router, {
        "reviewer_agent": "reviewer_agent",
        "__end__": END
    })

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

    # Initial state - start fresh with entry agent
    state: AgentState = {
        "messages": [
            {"role": "ai", "content": "Welcome to the AI Robo-Advisor! I'll help you create a personalized investment plan through a structured process."}
        ],
        "answers": {},
        "risk": None,
        "intent_to_risk": False,
        "intent_to_portfolio": False,
        "intent_to_investment": False,
        "intent_to_trading": False,
        "portfolio": None,
        "investment": None,
        "trading_requests": None,
        "ready_to_proceed": None,
        "all_phases_complete": False,
        "next_phase": "risk",  # Start with risk assessment
        "summary_shown": {
            "risk": False,
            "portfolio": False,
            "investment": False,
            "trading": False
        },  # Track if summary has been shown for each phase
        "status_tracking": {
            "risk": {"done": False, "awaiting_input": False},
            "portfolio": {"done": False, "awaiting_input": False},
            "investment": {"done": False, "awaiting_input": False},
            "trading": {"done": False, "awaiting_input": False},
            "reviewer": {"done": False, "awaiting_input": False}
        }
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