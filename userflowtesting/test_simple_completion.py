"""
Simple test for final completion flow
"""

from dotenv import load_dotenv
load_dotenv()

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import build_graph
from state import AgentState
from langchain_openai import ChatOpenAI


def test_simple_completion():
    """
    Test the final completion flow with a simple scenario
    """
    print("🧪 Testing Simple Final Completion Flow")
    print("=" * 50)
    
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
            {"role": "user", "content": "proceed"}  # Add user message to make last_is_user=True
        ],
        "answers": {},
        "risk": {"equity": 0.6, "bond": 0.4},
        "intent_to_risk": False,
        "entry_greeted": True,
        "intent_to_portfolio": False,
        "intent_to_investment": False,
        "intent_to_trading": True,
        "portfolio": {"large cap growth": 0.1, "bonds": 0.4, "cash": 0.05},
        "investment": {"fund1": {"ticker": "SPY", "weight": 0.1}},
        "trading_requests": {
            "trading_requests": [
                {"side": "BUY", "ticker": "SPY", "price": 400.0, "shares": 10},
                {"side": "SELL", "ticker": "CASH", "price": 1.0, "shares": 1000}
            ]
        },
        "ready_to_proceed": None,
        "all_phases_complete": False,
        "next_phase": None,
        "status_tracking": {
            "risk": {"done": True, "awaiting_input": False},
            "portfolio": {"done": True, "awaiting_input": False},
            "investment": {"done": True, "awaiting_input": False},
            "trading": {"done": True, "awaiting_input": False},
            "reviewer": {"done": False, "awaiting_input": False}
        }
    }
    
    print("📊 Initial state:")
    print(f"   Risk: {state.get('risk')}")
    print(f"   Portfolio: {state.get('portfolio')}")
    print(f"   Investment: {state.get('investment')}")
    print(f"   Trading: {state.get('trading_requests')}")
    print(f"   All phases complete: {state.get('all_phases_complete')}")
    
    # Test the flow
    print("\n--- Testing Graph Invoke ---")
    try:
        state = graph.invoke(state)
        print(f"✅ After graph invoke:")
        print(f"   All phases complete: {state.get('all_phases_complete')}")
        print(f"   Status tracking: {state.get('status_tracking')}")
        
        if state["messages"]:
            last_message = state["messages"][-1]
            if last_message["role"] == "ai":
                print(f"🤖 Last AI message: {last_message['content'][:300]}...")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("✅ Simple completion test completed!")
    return True


if __name__ == "__main__":
    test_simple_completion()
