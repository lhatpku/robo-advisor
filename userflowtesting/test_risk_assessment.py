#!/usr/bin/env python3
"""
Test case: Risk assessment questionnaire flow
"""

import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import build_graph
from state import AgentState
from langchain_openai import ChatOpenAI

def test_risk_assessment_flow():
    """Test risk assessment questionnaire flow."""
    print("Test: Risk Assessment Questionnaire Flow")
    print("-" * 50)
    
    try:
        # Build the graph
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        graph = build_graph(llm)
        
        # Initialize state
        state = AgentState(
            messages=[],
            q_idx=0,
            answers={},
            done=False,
            risk=None,
            awaiting_input=False,
            intent_to_risk=False,
            entry_greeted=False,
            intent_to_portfolio=False,
            intent_to_investment=False,
            intent_to_trading=False,
            portfolio=None,
            investment=None,
            trading_requests=None,
            ready_to_proceed=None,
            all_phases_complete=False,
            next_phase=None
        )
        
        # Test risk assessment steps
        risk_steps = [
            ("set as 0.6", "Set initial equity"),
            ("review", "Review equity allocation"),
            ("guidance", "Use guidance instead"),
            ("3", "Emergency savings: more than 6 months"),
            ("3", "Account representation: more than 50%"),
            ("3", "Investment horizon: 10-15 years"),
            ("3", "Early withdrawals: Likely"),
            ("3", "Investment knowledge: Expert"),
            ("3", "Growth vs income: Value income guarantee more"),
            ("3", "Market crash response: More conservative portfolio")
        ]
        
        for user_input, step_description in risk_steps:
            state["messages"].append({"role": "user", "content": user_input})
            state = graph.invoke(state)
            print(f"  ✓ {step_description}")
        
        # Verify risk assessment result
        if state.get("risk") and state["risk"].get("equity") == 0.15:
            print("  ✓ Test PASSED: Risk assessment completed - 15% equity / 85% bonds")
            return True
        else:
            print("  ✗ Test FAILED: Risk assessment did not complete correctly")
            return False
            
    except Exception as e:
        print(f"  ✗ Test ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    test_risk_assessment_flow()
