"""
Test for reviewer agent handling final completion options
"""

from dotenv import load_dotenv
load_dotenv()

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import build_graph
from state import AgentState
from langchain_openai import ChatOpenAI


def test_reviewer_final_completion():
    """
    Test that the reviewer agent handles final completion options correctly
    """
    print("🧪 Testing Reviewer Agent Final Completion Options")
    print("=" * 60)
    
    # Build the graph
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    graph = build_graph(llm)
    
    # Initialize state with all phases completed
    state: AgentState = {
        "messages": [
            {"role": "user", "content": "set as 0.6"},
            {"role": "ai", "content": "Updated: **60% equity / 40% bonds**."},
            {"role": "user", "content": "proceed"},
            {"role": "ai", "content": "Portfolio optimization complete."},
            {"role": "user", "content": "proceed"},
            {"role": "ai", "content": "Investment selection complete."},
            {"role": "user", "content": "proceed"},
            {"role": "ai", "content": "Trading requests complete."},
            {"role": "user", "content": "proceed"}  # This should trigger final completion
        ],
        "answers": {},
        "risk": {"equity": 0.6, "bond": 0.4},
        "intent_to_risk": False,
        "entry_greeted": True,
        "intent_to_portfolio": False,
        "intent_to_investment": False,
        "intent_to_trading": False,
        "portfolio": {"large cap growth": 0.1, "bonds": 0.4, "cash": 0.05},
        "investment": {"fund1": {"ticker": "SPY", "weight": 0.1}},
        "trading_requests": {
            "trading_requests": [
                {"side": "BUY", "ticker": "SPY", "price": 400.0, "shares": 10},
                {"side": "SELL", "ticker": "CASH", "price": 1.0, "shares": 1000}
            ]
        },
        "ready_to_proceed": None,
        "all_phases_complete": True,
        "next_phase": None,
        "summary_shown": {
            "risk": True,
            "portfolio": True,
            "investment": True,
            "trading": True
        },
        "status_tracking": {
            "risk": {"done": True, "awaiting_input": False},
            "portfolio": {"done": True, "awaiting_input": False},
            "investment": {"done": True, "awaiting_input": False},
            "trading": {"done": True, "awaiting_input": False},
            "reviewer": {"done": False, "awaiting_input": False}
        }
    }
    
    print("📊 Initial state:")
    print(f"   All phases complete: {state.get('all_phases_complete')}")
    print(f"   Risk: {state.get('risk')}")
    print(f"   Portfolio: {state.get('portfolio')}")
    print(f"   Investment: {state.get('investment')}")
    print(f"   Trading: {state.get('trading_requests')}")
    
    # Test 1: User says "review" - should show completion message again
    print("\n--- Test 1: User says 'review' ---")
    state["messages"].append({"role": "user", "content": "review"})
    
    try:
        state = graph.invoke(state)
        print(f"✅ After 'review':")
        print(f"   All phases complete: {state.get('all_phases_complete')}")
        
        if state["messages"]:
            last_message = state["messages"][-1]
            if last_message["role"] == "ai":
                print(f"🤖 Last AI message: {last_message['content'][:200]}...")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    # Test 2: User says "start over" - should reset everything and route to entry agent
    print("\n--- Test 2: User says 'start over' ---")
    state["messages"].append({"role": "user", "content": "start over"})
    
    try:
        state = graph.invoke(state)
        print(f"✅ After 'start over':")
        print(f"   All phases complete: {state.get('all_phases_complete')}")
        print(f"   Risk: {state.get('risk')}")
        print(f"   Portfolio: {state.get('portfolio')}")
        print(f"   Investment: {state.get('investment')}")
        print(f"   Trading: {state.get('trading_requests')}")
        print(f"   Entry greeted: {state.get('entry_greeted')}")
        
        if state["messages"]:
            last_message = state["messages"][-1]
            if last_message["role"] == "ai":
                print(f"🤖 Last AI message: {last_message['content']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    # Test 3: User says "proceed" - should show final confirmation and route to entry agent
    print("\n--- Test 3: User says 'proceed' ---")
    # Reset to completion state first
    state["all_phases_complete"] = True
    state["risk"] = {"equity": 0.6, "bond": 0.4}
    state["portfolio"] = {"large cap growth": 0.1, "bonds": 0.4, "cash": 0.05}
    state["investment"] = {"fund1": {"ticker": "SPY", "weight": 0.1}}
    state["trading_requests"] = {"trading_requests": [{"side": "BUY", "ticker": "SPY", "price": 400.0, "shares": 10}]}
    
    state["messages"].append({"role": "user", "content": "proceed"})
    
    try:
        state = graph.invoke(state)
        print(f"✅ After 'proceed':")
        print(f"   All phases complete: {state.get('all_phases_complete')}")
        
        if state["messages"]:
            last_message = state["messages"][-1]
            if last_message["role"] == "ai":
                print(f"🤖 Last AI message: {last_message['content']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ Reviewer agent final completion test completed!")
    return True


if __name__ == "__main__":
    test_reviewer_final_completion()
