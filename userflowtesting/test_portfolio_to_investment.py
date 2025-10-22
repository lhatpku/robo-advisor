#!/usr/bin/env python3
"""
Test case: Portfolio agent to Investment agent transition
"""
import os
import sys
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import build_graph
from state import AgentState
from langchain_openai import ChatOpenAI

def test_portfolio_to_investment():
    """Test the flow from portfolio agent to investment agent."""
    load_dotenv()
    
    llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.0)
    graph = build_graph(llm)
    
    print("🧪 Testing Portfolio to Investment Agent Transition")
    print("=" * 60)
    
    # Initial state with portfolio completed
    state: AgentState = {
        'messages': [
            {'role': 'ai', 'content': 'Here is the plan: I will build an asset-class portfolio using mean-variance optimization. Defaults are lambda = 1.0 and cash_reserve = 0.05. Say set lambda to 1, set cash to 0.05, or just run to optimize now.'},
            {'role': 'user', 'content': 'run'}
        ],
        'answers': {},
        'risk': {'equity': 0.6, 'bond': 0.4},
        'intent_to_risk': False,
        'intent_to_portfolio': False,
        'intent_to_investment': False,
        'intent_to_trading': False,
        'entry_greeted': False,
        'portfolio': {'large_cap_growth': 0.1, 'bonds': 0.4, 'cash': 0.05},
        'investment': None,
        'trading_requests': None,
        'ready_to_proceed': None,
        'all_phases_complete': False,
        'next_phase': None,
        'status_tracking': {
            'risk': {'done': False, 'awaiting_input': False},
            'portfolio': {'done': False, 'awaiting_input': True},
            'investment': {'done': False, 'awaiting_input': False},
            'trading': {'done': False, 'awaiting_input': False},
            'reviewer': {'done': False, 'awaiting_input': False}
        }
    }
    
    print("📊 Initial state:")
    print(f"   Portfolio: {state.get('portfolio')}")
    print(f"   Portfolio status: {state.get('status_tracking', {}).get('portfolio', {})}")
    
    print("\n--- Step 1: User says 'run' to optimize portfolio ---")
    state = graph.invoke(state)
    print(f"✅ Last message: {state['messages'][-1]['content'][:100]}...")
    print(f"   Portfolio status: {state.get('status_tracking', {}).get('portfolio', {})}")
    
    print("\n--- Step 2: User says 'proceed' to move to investment ---")
    state['messages'].append({'role': 'user', 'content': 'proceed'})
    state = graph.invoke(state)
    print(f"✅ Last message: {state['messages'][-1]['content'][:100]}...")
    print(f"   Portfolio status: {state.get('status_tracking', {}).get('portfolio', {})}")
    print(f"   Ready to proceed: {state.get('ready_to_proceed')}")
    print(f"   Intent to investment: {state.get('intent_to_investment')}")
    
    print("\n--- Step 3: User says 'yes' to proceed with investment ---")
    state['messages'].append({'role': 'user', 'content': 'yes'})
    state = graph.invoke(state)
    print(f"✅ Last message: {state['messages'][-1]['content'][:100]}...")
    print(f"   Investment created: {bool(state.get('investment'))}")
    print(f"   Investment status: {state.get('status_tracking', {}).get('investment', {})}")
    
    print("\n--- Step 4: User selects fund strategy ---")
    state['messages'].append({'role': 'user', 'content': '1'})  # Select balanced strategy
    state = graph.invoke(state)
    print(f"✅ Last message: {state['messages'][-1]['content'][:100]}...")
    print(f"   Investment created: {bool(state.get('investment'))}")
    print(f"   Investment status: {state.get('status_tracking', {}).get('investment', {})}")
    
    print("\n=== Verification ===")
    if state.get('investment'):
        print("✅ Successfully transitioned to investment agent!")
    else:
        print("❌ Failed to transition to investment agent")
        print(f"   Current state: {state.get('status_tracking', {})}")
    
    return state

if __name__ == "__main__":
    test_portfolio_to_investment()
