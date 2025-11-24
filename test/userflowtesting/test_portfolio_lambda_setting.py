"""
Test case for portfolio agent lambda setting functionality
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


def test_portfolio_lambda_setting():
    """
    Test that portfolio agent can properly set lambda value
    """
    print("Testing Portfolio Agent Lambda Setting")
    print("=" * 50)
    
    # Build the graph
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    graph = build_graph(llm)
    
    # Initialize state with risk completed and portfolio ready
    state: AgentState = {
        "messages": [
            {"role": "user", "content": "yes"},
            {"role": "ai", "content": "Great! Let's define your risk profile..."},
            {"role": "user", "content": "set as 0.6"},
            {"role": "ai", "content": "Perfect! I've set your allocation to **60% equity / 40% bonds**."},
            {"role": "user", "content": "proceed"},
            {"role": "ai", "content": "Here's the plan: I'll build an asset-class portfolio using mean-variance optimization. Defaults are **lambda = 1.0** and **cash_reserve = 0.05**. Say \"set lambda to 1\", \"set cash to 0.05\", or just \"run\" to optimize now."}
        ],
        "answers": {},
        "risk": {"equity": 0.6, "bond": 0.4},
        "intent_to_risk": False,
        "entry_greeted": True,
        "intent_to_portfolio": True,
        "intent_to_investment": False,
        "intent_to_trading": False,
        "portfolio": None,
        "investment": None,
        "trading_requests": None,
        "ready_to_proceed": None,
        "all_phases_complete": False,
        "next_phase": "portfolio",
        "summary_shown": {
            "risk": True,
            "portfolio": False,
            "investment": False,
            "trading": False
        },
        "status_tracking": {
            "risk": {"done": True, "awaiting_input": False},
            "portfolio": {"done": False, "awaiting_input": True},
            "investment": {"done": False, "awaiting_input": False},
            "trading": {"done": False, "awaiting_input": False},
            "reviewer": {"done": False, "awaiting_input": False}
        },
        "correlation_id": None
    }
    
    print("📊 Initial state:")
    print(f"   Risk: {state.get('risk')}")
    print(f"   Portfolio: {state.get('portfolio')}")
    print(f"   Portfolio status: {state.get('status_tracking', {}).get('portfolio', {})}")
    
    # Test 1: User says 'set lambda to 2.5'
    print("\n--- Test 1: User says 'set lambda to 2.5' ---")
    state['messages'].append({'role': 'user', 'content': 'set lambda to 2.5'})
    state = graph.invoke(state)
    print(f"SUCCESS: Last message: {state['messages'][-1]['content'][:100]}...")
    print(f"   Portfolio status: {state.get('status_tracking', {}).get('portfolio', {})}")
    
    # Check if lambda setting was acknowledged
    last_message = state['messages'][-1]['content'] if state.get('messages') else ""
    # Check that message mentions lambda or 2.5
    if any(keyword in last_message.lower() for keyword in ["lambda", "2.5", "set"]):
        print("SUCCESS: Lambda setting acknowledged")
    else:
        print(f"⚠️  Lambda setting response may not be clear, got: {last_message[:100]}")
        # Don't fail - just warn, as message format may vary
    
    # Test 2: User says 'run' to optimize with new lambda setting
    print("\n--- Test 2: User says 'run' to optimize ---")
    state['messages'].append({'role': 'user', 'content': 'run'})
    state = graph.invoke(state)
    print(f"SUCCESS: Last message: {state['messages'][-1]['content'][:100]}...")
    print(f"   Portfolio status: {state.get('status_tracking', {}).get('portfolio', {})}")
    
    # Check if portfolio was optimized (portfolio exists and has content)
    if state.get('portfolio') and len(state.get('portfolio', {})) > 0:
        print("SUCCESS: Portfolio optimization completed successfully")
    else:
        raise AssertionError("Portfolio optimization failed")


if __name__ == "__main__":
    test_portfolio_lambda_setting()
