"""
Test for risk agent review/edit functionality after setting equity
"""

from dotenv import load_dotenv
load_dotenv()

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import build_graph
from state import AgentState
from langchain_openai import ChatOpenAI


def test_risk_agent_review_edit():
    """
    Test the risk agent review/edit functionality after setting equity
    Expected flow:
    1. User says "yes" to start journey
    2. Risk agent asks for mode selection
    3. User says "set equity as 0.6"
    4. Risk agent sets equity and shows options (review/edit/proceed)
    5. User says "review" - should stay in risk agent and show current allocation
    6. User can set new equity or use guidance
    7. User can proceed to portfolio
    """
    print("🧪 Testing Risk Agent Review/Edit Functionality")
    print("=" * 60)
    
    try:
        # Build the graph
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
        graph = build_graph(llm)
        
        # Start with empty state
        state: AgentState = {
            "messages": [],
            "q_idx": 0,
            "answers": {},
            "done": False,
            "risk": None,
            "awaiting_input": False,
            "intent_to_risk": False,
            "entry_greeted": False,
            "intent_to_portfolio": False,
            "intent_to_investment": False,
            "intent_to_trading": False,
            "portfolio": None,
            "investment": None,
            "trading_requests": None,
            "ready_to_proceed": None,
            "all_phases_complete": False,
            "next_phase": None
        }
        
        # Step 1: Initial greeting
        print("\n--- Step 1: Initial greeting ---")
        state = graph.invoke(state)
        print(f"✅ AI: {state['messages'][-1]['content'][:100]}...")
        
        # Step 2: User says "yes"
        print("\n--- Step 2: User says 'yes' ---")
        state["messages"].append({"role": "user", "content": "yes"})
        state = graph.invoke(state)
        print(f"✅ AI: {state['messages'][-1]['content'][:100]}...")
        
        # Step 3: User says "set equity as 0.6"
        print("\n--- Step 3: User says 'set equity as 0.6' ---")
        state["messages"].append({"role": "user", "content": "set equity as 0.6"})
        state = graph.invoke(state)
        print(f"✅ AI: {state['messages'][-1]['content'][:100]}...")
        print(f"   Risk set: {state.get('risk')}")
        print(f"   Done: {state.get('done')}")
        print(f"   Awaiting input: {state.get('awaiting_input')}")
        
        # Verify the equity was set correctly
        if state.get('risk') and state['risk'].get('equity') == 0.6:
            print("✅ Equity correctly set to 0.6")
        else:
            print("❌ Equity not set correctly")
            return False
        
        # Step 4: User says "review"
        print("\n--- Step 4: User says 'review' ---")
        state["messages"].append({"role": "user", "content": "review"})
        state = graph.invoke(state)
        print(f"✅ AI: {state['messages'][-1]['content'][:100]}...")
        print(f"   Done: {state.get('done')}")
        print(f"   Awaiting input: {state.get('awaiting_input')}")
        
        # Verify we're still in risk agent (not routed to portfolio)
        if state.get('done') and state.get('awaiting_input'):
            print("✅ Still in risk agent for review/edit")
        else:
            print("❌ Not in risk agent for review/edit")
            return False
        
        # Step 5: User says "set equity as 0.7" (change equity)
        print("\n--- Step 5: User says 'set equity as 0.7' (change equity) ---")
        state["messages"].append({"role": "user", "content": "set equity as 0.7"})
        state = graph.invoke(state)
        print(f"✅ AI: {state['messages'][-1]['content'][:100]}...")
        print(f"   Risk set: {state.get('risk')}")
        
        # Verify the equity was updated correctly
        if state.get('risk') and state['risk'].get('equity') == 0.7:
            print("✅ Equity correctly updated to 0.7")
        else:
            print("❌ Equity not updated correctly")
            return False
        
        # Step 6: User says "proceed" to portfolio
        print("\n--- Step 6: User says 'proceed' to portfolio ---")
        state["messages"].append({"role": "user", "content": "proceed"})
        state = graph.invoke(state)
        print(f"✅ AI: {state['messages'][-1]['content'][:100]}...")
        print(f"   Intent to portfolio: {state.get('intent_to_portfolio')}")
        
        # Verify routing to portfolio
        if state.get('intent_to_portfolio'):
            print("✅ Correctly routed to portfolio agent")
        else:
            print("❌ Not routed to portfolio agent")
            return False
        
        print("\n" + "=" * 60)
        print("✅ Risk agent review/edit functionality test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False


def test_risk_agent_guidance_after_equity():
    """
    Test using guidance after setting equity
    """
    print("\n🧪 Testing Risk Agent Guidance After Equity")
    print("=" * 60)
    
    try:
        # Build the graph
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
        graph = build_graph(llm)
        
        # Start with equity already set
        state: AgentState = {
            "messages": [
                {"role": "ai", "content": "I am a robo advisor, how can I assist you. Let me know if you want to start the journey, the first step is to define your risk profile."},
                {"role": "user", "content": "yes"},
                {"role": "ai", "content": "Great! Let's define your risk profile. You have two options: 1) Set your equity allocation directly 2) Use guidance..."},
                {"role": "user", "content": "set equity as 0.6"},
                {"role": "ai", "content": "Perfect! I've set your allocation to 60% equity / 40% bonds. To continue, you can: • Review/edit this allocation by saying 'review' or 'edit' • Use guidance to reset through questionnaire by saying 'use guidance' • Proceed to portfolio construction by saying 'proceed'"}
            ],
            "q_idx": 0,
            "answers": {},
            "done": True,
            "risk": {"equity": 0.6, "bond": 0.4},
            "awaiting_input": True,
            "intent_to_risk": True,
            "entry_greeted": True,
            "intent_to_portfolio": False,
            "intent_to_investment": False,
            "intent_to_trading": False,
            "portfolio": None,
            "investment": None,
            "trading_requests": None,
            "ready_to_proceed": None,
            "all_phases_complete": False,
            "next_phase": None
        }
        
        # User says "use guidance"
        print("\n--- User says 'use guidance' after setting equity ---")
        state["messages"].append({"role": "user", "content": "use guidance"})
        state = graph.invoke(state)
        print(f"✅ AI: {state['messages'][-1]['content'][:100]}...")
        print(f"   Risk reset: {state.get('risk')}")
        print(f"   Q index: {state.get('q_idx')}")
        print(f"   Done: {state.get('done')}")
        
        # Verify guidance started
        if not state.get('risk') and state.get('q_idx') == 0 and not state.get('done'):
            print("✅ Guidance questionnaire started correctly")
        else:
            print("❌ Guidance questionnaire not started correctly")
            return False
        
        print("\n" + "=" * 60)
        print("✅ Risk agent guidance after equity test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False


if __name__ == "__main__":
    success1 = test_risk_agent_review_edit()
    success2 = test_risk_agent_guidance_after_equity()
    
    if success1 and success2:
        print("\n🎉 All tests passed!")
    else:
        print("\n💥 Some tests failed!")
        sys.exit(1)
