#!/usr/bin/env python3
"""
Test case for trading agent completion
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import ChatOpenAI
from app import build_graph
from state import AgentState

def test_trading_completion():
    """Test trading agent completion and routing to reviewer"""
    print("🧪 Testing Trading Agent Completion")
    print("=" * 60)
    
    llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)
    graph = build_graph(llm)
    
    # Start with a completed investment portfolio
    state: AgentState = {
        "messages": [],
        "answers": {},
        "risk": {"equity": 0.6, "bond": 0.4},
        "intent_to_risk": False,
        "intent_to_portfolio": False,
        "intent_to_investment": False,
        "intent_to_trading": False,
        "entry_greeted": False,
        "portfolio": {"large_cap_growth": 0.1, "bonds": 0.4, "cash": 0.05},
        "investment": {"fund1": {"ticker": "SPY", "weight": 0.1}},
        "trading_requests": None,
        "ready_to_proceed": None,
        "all_phases_complete": False,
        "next_phase": "trading",
        "summary_shown": {
            "risk": True,
            "portfolio": True,
            "investment": True,
            "trading": False
        },
        "status_tracking": {
            "risk": {"done": True, "awaiting_input": False},
            "portfolio": {"done": True, "awaiting_input": False},
            "investment": {"done": True, "awaiting_input": False},
            "trading": {"done": False, "awaiting_input": True},
            "reviewer": {"done": False, "awaiting_input": False}
        }
    }
    
    print(f"📊 Initial state:")
    print(f"   Trading requests: {state.get('trading_requests')}")
    print(f"   Trading status: {state.get('status_tracking', {}).get('trading', {})}")
    print(f"   All phases complete: {state.get('all_phases_complete')}")
    
    # Step 1: User says 'proceed' to go to trading
    print("\n--- Step 1: User says 'proceed' to go to trading ---")
    state['messages'].append({'role': 'user', 'content': 'proceed'})
    state = graph.invoke(state)
    print(f"✅ AI: {state['messages'][-1]['content'][:100]}...")
    print(f"   Trading status: {state.get('status_tracking', {}).get('trading', {})}")
    
    # Step 2: User selects scenario 2
    print("\n--- Step 2: User selects scenario 2 ---")
    state['messages'].append({'role': 'user', 'content': '2'})
    state = graph.invoke(state)
    print(f"✅ AI: {state['messages'][-1]['content'][:100]}...")
    print(f"   Trading requests: {bool(state.get('trading_requests'))}")
    print(f"   Trading status: {state.get('status_tracking', {}).get('trading', {})}")
    print(f"   All phases complete: {state.get('all_phases_complete')}")
    
    # Step 3: User says 'yes' to confirm
    print("\n--- Step 3: User says 'yes' to confirm ---")
    state['messages'].append({'role': 'user', 'content': 'yes'})
    state = graph.invoke(state)
    print(f"✅ AI: {state['messages'][-1]['content'][:100]}...")
    print(f"   Trading requests: {bool(state.get('trading_requests'))}")
    print(f"   Trading status: {state.get('status_tracking', {}).get('trading', {})}")
    print(f"   All phases complete: {state.get('all_phases_complete')}")
    
    print("\n=== Verification ===")
    if state.get('trading_requests') and state.get('all_phases_complete'):
        print("✅ Trading completion test passed!")
        return True
    else:
        print("❌ Trading completion test failed")
        print(f"   Trading requests: {bool(state.get('trading_requests'))}")
        print(f"   All phases complete: {state.get('all_phases_complete')}")
        return False

if __name__ == "__main__":
    test_trading_completion()
