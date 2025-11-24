"""
Test for reviewer agent handling final completion options
"""

from dotenv import load_dotenv
load_dotenv()

import sys
import os
# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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
        },
        "correlation_id": None
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
        print(f"SUCCESS: After 'review':")
        print(f"   All phases complete: {state.get('all_phases_complete')}")
        
        # Check that reviewer responded
        if state["messages"]:
            last_message = state["messages"][-1]
            if last_message["role"] == "ai":
                content = last_message['content']
                print(f"AI Last message length: {len(content)}")
                assert len(content) > 20, "Reviewer should provide a response"
        
    except Exception as e:
        print(f"FAILED: Error: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Test 2: User says "start over" - should reset everything
    print("\n--- Test 2: User says 'start over' ---")
    state["messages"].append({"role": "user", "content": "start over"})
    
    try:
        state = graph.invoke(state)
        print(f"SUCCESS: After 'start over':")
        print(f"   All phases complete: {state.get('all_phases_complete')}")
        
        # Check that we got a response
        if state["messages"]:
            last_message = state["messages"][-1]
            if last_message["role"] == "ai":
                content = last_message['content']
                print(f"AI Last message length: {len(content)}")
                assert len(content) > 20, "Should get a response after start over"
        
        # Check that next_phase was reset
        next_phase = state.get('next_phase')
        assert next_phase is None or next_phase == "risk", f"Next phase should be None or 'risk', got {next_phase}"
        
    except Exception as e:
        print(f"FAILED: Error: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    
    print("\n" + "=" * 60)
    print("SUCCESS: Reviewer agent final completion test completed!")


if __name__ == "__main__":
    test_reviewer_final_completion()
