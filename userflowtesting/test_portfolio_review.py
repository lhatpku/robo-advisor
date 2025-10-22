"""
Test case for portfolio agent review functionality
"""

from dotenv import load_dotenv
load_dotenv()

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import build_graph
from state import AgentState
from langchain_openai import ChatOpenAI


def test_portfolio_review():
    """
    Test portfolio review functionality: run first, review, reset parameters, and rerun
    """
    print("🧪 Testing Portfolio Agent Review Functionality")
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
        "next_phase": None,
        "status_tracking": {
            "risk": {"done": True, "awaiting_input": False},
            "portfolio": {"done": False, "awaiting_input": True},
            "investment": {"done": False, "awaiting_input": False},
            "trading": {"done": False, "awaiting_input": False},
            "reviewer": {"done": False, "awaiting_input": False}
        }
    }
    
    print("📊 Initial state:")
    print(f"   Risk: {state.get('risk')}")
    print(f"   Portfolio: {state.get('portfolio')}")
    print(f"   Portfolio status: {state.get('status_tracking', {}).get('portfolio', {})}")
    
    # Test 1: User says 'run' to optimize portfolio first
    print("\n--- Test 1: User says 'run' to optimize portfolio first ---")
    state['messages'].append({'role': 'user', 'content': 'run'})
    state = graph.invoke(state)
    print(f"✅ Last message: {state['messages'][-1]['content'][:100]}...")
    print(f"   Portfolio status: {state.get('status_tracking', {}).get('portfolio', {})}")
    
    # Check if portfolio was created
    if state.get('portfolio'):
        print("✅ Portfolio optimization completed successfully")
    else:
        print("❌ Portfolio optimization failed")
        return False
    
    # Test 2: User says 'review' to see portfolio and options
    print("\n--- Test 2: User says 'review' to see portfolio and options ---")
    state['messages'].append({'role': 'user', 'content': 'review'})
    state = graph.invoke(state)
    print(f"✅ Last message: {state['messages'][-1]['content'][:100]}...")
    print(f"   Portfolio status: {state.get('status_tracking', {}).get('portfolio', {})}")
    
    # Check if review shows portfolio and options
    last_message = state['messages'][-1]['content']
    if "Your current portfolio" in last_message and "Current parameters" in last_message:
        print("✅ Review shows portfolio and editing options")
    else:
        print(f"❌ Expected portfolio review with options, got: {last_message[:200]}...")
        return False
    
    # Test 3: User sets cash to 0.04
    print("\n--- Test 3: User sets cash to 0.04 ---")
    state['messages'].append({'role': 'user', 'content': 'set cash to 0.04'})
    state = graph.invoke(state)
    print(f"✅ Last message: {state['messages'][-1]['content'][:100]}...")
    print(f"   Portfolio status: {state.get('status_tracking', {}).get('portfolio', {})}")
    
    # Check if cash was set correctly
    if "Set cash reserve to 0.04" in state['messages'][-1]['content']:
        print("✅ Cash reserve set to 0.04")
    else:
        print("❌ Cash setting failed")
        return False
    
    # Test 4: User sets lambda to 3
    print("\n--- Test 4: User sets lambda to 3 ---")
    state['messages'].append({'role': 'user', 'content': 'set lambda to 3'})
    state = graph.invoke(state)
    print(f"✅ Last message: {state['messages'][-1]['content'][:100]}...")
    print(f"   Portfolio status: {state.get('status_tracking', {}).get('portfolio', {})}")
    
    # Check if lambda was set correctly
    if "Set lambda to 3" in state['messages'][-1]['content']:
        print("✅ Lambda set to 3")
    else:
        print("❌ Lambda setting failed")
        return False
    
    # Test 5: User says 'run' to re-optimize with new parameters
    print("\n--- Test 5: User says 'run' to re-optimize with new parameters ---")
    state['messages'].append({'role': 'user', 'content': 'run'})
    state = graph.invoke(state)
    print(f"✅ Last message: {state['messages'][-1]['content'][:100]}...")
    print(f"   Portfolio status: {state.get('status_tracking', {}).get('portfolio', {})}")
    
    # Check if portfolio was re-optimized
    if state.get('portfolio') and state.get('status_tracking', {}).get('portfolio', {}).get('awaiting_input'):
        print("✅ Portfolio re-optimization completed successfully")
        
        # Verify cash weight in new portfolio
        portfolio = state.get('portfolio', {})
        cash_weight = portfolio.get('cash', 0.0)
        print(f"   Cash weight in new portfolio: {cash_weight:.3f}")
        print(f"   Expected cash weight: 0.040")
        
        if abs(cash_weight - 0.04) < 0.001:  # Allow small floating point differences
            print("✅ Cash weight in re-optimized portfolio matches set value (0.04)")
            return True
        else:
            print(f"❌ Expected cash weight 0.04, got {cash_weight:.3f}")
            return False
    else:
        print("❌ Portfolio re-optimization failed")
        return False


if __name__ == "__main__":
    test_portfolio_review()
