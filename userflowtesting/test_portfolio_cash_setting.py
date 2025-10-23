"""
Test case for portfolio agent cash setting functionality
"""

from dotenv import load_dotenv
load_dotenv()

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import build_graph
from state import AgentState
from langchain_openai import ChatOpenAI


def test_portfolio_cash_setting():
    """
    Test that portfolio agent can properly set cash reserve value
    """
    print("🧪 Testing Portfolio Agent Cash Setting")
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
        }
    }
    
    print("📊 Initial state:")
    print(f"   Risk: {state.get('risk')}")
    print(f"   Portfolio: {state.get('portfolio')}")
    print(f"   Portfolio status: {state.get('status_tracking', {}).get('portfolio', {})}")
    
    # Test 1: User says 'set cash as 0.02'
    print("\n--- Test 1: User says 'set cash as 0.02' ---")
    state['messages'].append({'role': 'user', 'content': 'set cash as 0.02'})
    state = graph.invoke(state)
    print(f"✅ Last message: {state['messages'][-1]['content'][:100]}...")
    print(f"   Portfolio status: {state.get('status_tracking', {}).get('portfolio', {})}")
    
    # Check if the cash setting was acknowledged correctly
    last_message = state['messages'][-1]['content']
    if "Set cash reserve to 0.02" in last_message:
        print("✅ Cash reserve setting acknowledged correctly")
    else:
        print(f"❌ Expected cash setting acknowledgment, got: {last_message}")
        return False
    
    # Test 2: User says 'run' to optimize with new cash setting
    print("\n--- Test 2: User says 'run' to optimize ---")
    state['messages'].append({'role': 'user', 'content': 'run'})
    state = graph.invoke(state)
    print(f"✅ Last message: {state['messages'][-1]['content'][:100]}...")
    print(f"   Portfolio status: {state.get('status_tracking', {}).get('portfolio', {})}")
    
    # Check if portfolio was optimized (portfolio exists and has content)
    if state.get('portfolio') and len(state.get('portfolio', {})) > 0:
        print("✅ Portfolio optimization completed successfully")
        
        # Test 3: Verify cash weight in portfolio result matches set value (0.02)
        print("\n--- Test 3: Verify cash weight in portfolio result ---")
        portfolio = state.get('portfolio', {})
        cash_weight = portfolio.get('cash', 0.0)
        print(f"   Cash weight in portfolio: {cash_weight:.3f}")
        print(f"   Expected cash weight: 0.020")
        
        if abs(cash_weight - 0.02) < 0.001:  # Allow small floating point differences
            print("✅ Cash weight in portfolio result matches set value (0.02)")
            return True
        else:
            print(f"❌ Expected cash weight 0.02, got {cash_weight:.3f}")
            return False
    else:
        print("❌ Portfolio optimization failed")
        return False


if __name__ == "__main__":
    test_portfolio_cash_setting()
