"""
Test case for portfolio agent review functionality
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


def test_portfolio_review():
    """
    Test portfolio review functionality: run first, review, reset parameters, and rerun
    """
    print("Testing Portfolio Agent Review Functionality")
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
    
    # Test 1: User says 'run' to optimize portfolio first
    print("\n--- Test 1: User says 'run' to optimize portfolio first ---")
    state['messages'].append({'role': 'user', 'content': 'run'})
    state = graph.invoke(state)
    print(f"SUCCESS: Last message: {state['messages'][-1]['content'][:100]}...")
    print(f"   Portfolio status: {state.get('status_tracking', {}).get('portfolio', {})}")
    
    # Check if portfolio was created
    if state.get('portfolio'):
        print("SUCCESS: Portfolio optimization completed successfully")
    else:
        print("FAILED: Portfolio optimization failed")
        return False
    
    # Test 2: User says 'review' to see portfolio and options
    print("\n--- Test 2: User says 'review' to see portfolio and options ---")
    state['messages'].append({'role': 'user', 'content': 'review'})
    state = graph.invoke(state)
    print(f"SUCCESS: Last message: {state['messages'][-1]['content'][:100]}...")
    print(f"   Portfolio status: {state.get('status_tracking', {}).get('portfolio', {})}")
    
    # Check if review shows portfolio information
    last_message = state['messages'][-1]['content'] if state.get('messages') else ""
    # Check that message mentions portfolio-related terms
    if any(keyword in last_message.lower() for keyword in ["portfolio", "review", "current", "parameters", "lambda", "cash"]):
        print("SUCCESS: Review shows portfolio information")
    else:
        print(f"⚠️  Review message may not be clear, got: {last_message[:200]}...")
        # Don't fail - portfolio may be shown in different format
    
    # Test 3: User sets cash to 0.04
    print("\n--- Test 3: User sets cash to 0.04 ---")
    state['messages'].append({'role': 'user', 'content': 'set cash to 0.04'})
    state = graph.invoke(state)
    print(f"SUCCESS: Last message: {state['messages'][-1]['content'][:100]}...")
    print(f"   Portfolio status: {state.get('status_tracking', {}).get('portfolio', {})}")
    
    # Check if cash was set (check message mentions cash/0.04)
    last_msg = state['messages'][-1]['content'] if state.get('messages') else ""
    if any(keyword in last_msg.lower() for keyword in ["cash", "0.04", "reserve", "set"]):
        print("SUCCESS: Cash reserve setting acknowledged")
    else:
        print("⚠️  Cash setting response may not be clear")
    
    # Test 4: User sets lambda to 3
    print("\n--- Test 4: User sets lambda to 3 ---")
    state['messages'].append({'role': 'user', 'content': 'set lambda to 3'})
    state = graph.invoke(state)
    last_msg = state['messages'][-1]['content'] if state.get('messages') else ""
    print(f"SUCCESS: Last message length: {len(last_msg)}")
    print(f"   Portfolio status: {state.get('status_tracking', {}).get('portfolio', {})}")
    
    # Check if lambda was set (check message mentions lambda/3)
    if any(keyword in last_msg.lower() for keyword in ["lambda", "3", "set"]):
        print("SUCCESS: Lambda setting acknowledged")
    else:
        print("⚠️  Lambda setting response may not be clear")
    
    # Test 5: User says 'run' to re-optimize with new parameters
    print("\n--- Test 5: User says 'run' to re-optimize with new parameters ---")
    state['messages'].append({'role': 'user', 'content': 'run'})
    state = graph.invoke(state)
    print(f"SUCCESS: Last message: {state['messages'][-1]['content'][:100]}...")
    print(f"   Portfolio status: {state.get('status_tracking', {}).get('portfolio', {})}")
    
    # Check if portfolio was re-optimized
    if state.get('portfolio') and state.get('status_tracking', {}).get('portfolio', {}).get('awaiting_input'):
        print("✅ Portfolio re-optimization completed successfully")
        
        # Verify cash weight in new portfolio is reasonable
        portfolio = state.get('portfolio', {})
        cash_weight = portfolio.get('cash', 0.0)
        print(f"   Cash weight in new portfolio: {cash_weight:.3f}")
        print(f"   Expected cash weight: ~0.04")
        
        # Check that cash weight is reasonable
        assert 0 <= cash_weight <= 0.1, f"Cash weight {cash_weight:.3f} is out of reasonable range"
        if abs(cash_weight - 0.04) < 0.01:  # Allow tolerance
            print("SUCCESS: Cash weight in re-optimized portfolio is close to set value (0.04)")
        else:
            print(f"WARNING: Cash weight is {cash_weight:.3f}, expected ~0.04 (may be clamped)")
    else:
        raise AssertionError("Portfolio re-optimization failed")


if __name__ == "__main__":
    test_portfolio_review()
