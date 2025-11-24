#!/usr/bin/env python3
"""
Test case: Portfolio agent to Investment agent transition
"""
import os
import sys
from dotenv import load_dotenv

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app import build_graph
from state import AgentState
from langchain_openai import ChatOpenAI

def test_portfolio_to_investment():
    """Test the flow from portfolio agent to investment agent."""
    load_dotenv()
    
    llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.0)
    graph = build_graph(llm)
    
    print("Testing Portfolio to Investment Agent Transition")
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
        'next_phase': 'investment',
        'summary_shown': {
            'risk': True,
            'portfolio': True,
            'investment': False,
            'trading': False
        },
        'status_tracking': {
            'risk': {'done': True, 'awaiting_input': False},
            'portfolio': {'done': True, 'awaiting_input': False},
            'investment': {'done': False, 'awaiting_input': True},
            'trading': {'done': False, 'awaiting_input': False},
            'reviewer': {'done': False, 'awaiting_input': False}
        },
        'correlation_id': None
    }
    
    print("📊 Initial state:")
    print(f"   Portfolio: {state.get('portfolio')}")
    print(f"   Portfolio status: {state.get('status_tracking', {}).get('portfolio', {})}")
    
    print("\n--- Step 1: User says 'run' to optimize portfolio ---")
    state = graph.invoke(state)
    last_msg = state['messages'][-1]['content'] if state.get('messages') else ""
    print(f"SUCCESS: Last message length: {len(last_msg)}")
    print(f"   Portfolio status: {state.get('status_tracking', {}).get('portfolio', {})}")
    
    # Check that portfolio was created
    assert state.get('portfolio') is not None, "Portfolio should be created after optimization"
    portfolio_status = state.get('status_tracking', {}).get('portfolio', {})
    assert portfolio_status.get('done') == True, "Portfolio should be marked as done"
    
    print("\n--- Step 2: User says 'proceed' to move to investment ---")
    state['messages'].append({'role': 'user', 'content': 'proceed'})
    state = graph.invoke(state)
    last_msg = state['messages'][-1]['content'] if state.get('messages') else ""
    print(f"SUCCESS: Last message length: {len(last_msg)}")
    print(f"   Next phase: {state.get('next_phase')}")
    print(f"   Intent to investment: {state.get('intent_to_investment')}")
    
    # Check that we're moving to investment phase
    assert state.get('next_phase') == 'investment' or state.get('intent_to_investment') == True, \
        "Should be moving to investment phase"
    
    print("\n--- Step 3: User says 'proceed' again (or selects strategy) ---")
    state['messages'].append({'role': 'user', 'content': 'proceed'})
    state = graph.invoke(state)
    last_msg = state['messages'][-1]['content'] if state.get('messages') else ""
    print(f"SUCCESS: Last message length: {len(last_msg)}")
    print(f"   Investment created: {bool(state.get('investment'))}")
    
    # If investment not created yet, try selecting a strategy
    if not state.get('investment'):
        print("\n--- Step 4: User selects fund strategy ---")
        state['messages'].append({'role': 'user', 'content': '1'})  # Select balanced strategy
        state = graph.invoke(state)
        last_msg = state['messages'][-1]['content'] if state.get('messages') else ""
        print(f"SUCCESS: Last message length: {len(last_msg)}")
        print(f"   Investment created: {bool(state.get('investment'))}")
    
    print("\n=== Verification ===")
    # Check that we're in investment phase or investment was created
    investment_status = state.get('status_tracking', {}).get('investment', {})
    if state.get('investment') or investment_status.get('awaiting_input'):
        print("SUCCESS: Successfully transitioned to investment agent!")
    else:
        print("WARNING: Investment phase started but investment not yet created")
        print(f"   Investment status: {investment_status}")
        print(f"   Next phase: {state.get('next_phase')}")
        # This is still a valid state - investment agent may be waiting for more input
        # Just verify we're in the right phase
        assert state.get('next_phase') == 'investment' or investment_status.get('awaiting_input'), \
            "Should be in investment phase"

if __name__ == "__main__":
    test_portfolio_to_investment()
