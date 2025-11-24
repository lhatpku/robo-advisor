"""
Simple test for final completion flow
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


def test_simple_completion():
    """
    Test the final completion flow with a simple scenario
    """
    print("Testing Simple Final Completion Flow")
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
        "intent_to_trading": False,
        "portfolio": {"large cap growth": 0.1, "bonds": 0.4, "cash": 0.05},
        "investment": {"fund1": {"ticker": "SPY", "weight": 0.1}},
        "trading_requests": {
            "trading_requests": [
                {"Side": "BUY", "Ticker": "SPY", "Price": 400.0, "Shares": 10},
                {"Side": "SELL", "Ticker": "CASH", "Price": 1.0, "Shares": 1000}
            ]
        },
        "ready_to_proceed": None,
        "all_phases_complete": False,
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
    
    print("Initial state:")
    print(f"   Risk: {state.get('risk')}")
    print(f"   Portfolio: {state.get('portfolio')}")
    print(f"   Investment: {state.get('investment')}")
    print(f"   Trading: {state.get('trading_requests')}")
    print(f"   All phases complete: {state.get('all_phases_complete')}")
    
    # Test the flow
    print("\n--- Testing Graph Invoke ---")
    try:
        state = graph.invoke(state)
        print(f"After graph invoke:")
        print(f"   All phases complete: {state.get('all_phases_complete')}")
        print(f"   Status tracking: {state.get('status_tracking')}")
        
        # Check that reviewer agent processed the completion
        # All phases should be marked as done
        status_tracking = state.get('status_tracking', {})
        assert status_tracking.get('risk', {}).get('done') == True, "Risk should be done"
        assert status_tracking.get('portfolio', {}).get('done') == True, "Portfolio should be done"
        assert status_tracking.get('investment', {}).get('done') == True, "Investment should be done"
        assert status_tracking.get('trading', {}).get('done') == True, "Trading should be done"
        
        # Check that all phases have data
        assert state.get('risk') is not None, "Risk should exist"
        assert state.get('portfolio') is not None, "Portfolio should exist"
        assert state.get('investment') is not None, "Investment should exist"
        assert state.get('trading_requests') is not None, "Trading requests should exist"
        
        # Check that reviewer responded (should show final summary or options)
        if state["messages"]:
            last_message = state["messages"][-1]
            if last_message["role"] == "ai":
                content = last_message['content']
                print(f"Last AI message length: {len(content)}")
                # Should have substantial content (summary or options)
                assert len(content) > 50, "Reviewer should provide substantial response"
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    print("\n" + "=" * 50)
    print("SUCCESS: Simple completion test completed!")


if __name__ == "__main__":
    test_simple_completion()
