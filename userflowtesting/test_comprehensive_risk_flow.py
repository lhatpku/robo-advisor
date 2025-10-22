#!/usr/bin/env python3
"""
Test the comprehensive risk assessment flow from start to finish.
This covers the exact user flow provided by the user.
"""
from dotenv import load_dotenv
load_dotenv()

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import ChatOpenAI
from app import build_graph
from state import AgentState

def test_comprehensive_risk_flow():
    """
    Test the complete risk assessment flow:
    1. Start with greeting
    2. User says "yes" -> shows mode selection
    3. User says "set as 0.6" -> sets equity directly
    4. User says "guidance" -> starts questionnaire
    5. Complete questionnaire with answers
    6. User asks "why" -> gets explanation
    7. User says "proceed" -> moves to portfolio agent
    """
    print("=== Test: Comprehensive Risk Assessment Flow ===")
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    graph = build_graph(llm)
    
    state: AgentState = {
        "messages": [],
        "q_idx": 0,
        "answers": {},
        "done": False,
        "risk": None,
        "awaiting_input": False,
        "intent_to_risk": False,
        "intent_to_portfolio": False,
        "intent_to_investment": False,
        "intent_to_trading": False,
        "entry_greeted": False,
        "portfolio": None,
        "investment": None,
        "trading_requests": None,
        "ready_to_proceed": None,
        "all_phases_complete": False,
        "next_phase": None
    }
    
    try:
        # Step 1: Initial greeting
        state = graph.invoke(state)
        greeting = state["messages"][-1]["content"]
        print(f"1. Greeting: {greeting[:50]}...")
        assert "robo advisor" in greeting.lower()
        
        # Step 2: User says "yes" -> should show mode selection
        state["messages"].append({"role": "user", "content": "yes"})
        state = graph.invoke(state)
        mode_selection = state["messages"][-1]["content"]
        print(f"2. Mode selection: {mode_selection[:50]}...")
        assert "Great! Let's define your risk profile" in mode_selection
        assert "two options" in mode_selection
        
        # Step 3: User says "set as 0.6" -> should set equity directly
        state["messages"].append({"role": "user", "content": "set as 0.6"})
        state = graph.invoke(state)
        equity_set = state["messages"][-1]["content"]
        print(f"3. Equity set: {equity_set[:50]}...")
        assert "60% equity / 40% bonds" in equity_set
        assert state.get("risk") is not None
        assert state.get("risk", {}).get("equity") == 0.6
        
        # Step 4: User says "guidance" -> should start questionnaire
        state["messages"].append({"role": "user", "content": "guidance"})
        state = graph.invoke(state)
        first_question = state["messages"][-1]["content"]
        print(f"4. First question: {first_question[:50]}...")
        assert "emergency savings" in first_question.lower()
        assert "1)" in first_question and "2)" in first_question and "3)" in first_question
        
        # Step 5: Answer first question (3)
        state["messages"].append({"role": "user", "content": "3"})
        state = graph.invoke(state)
        second_question = state["messages"][-1]["content"]
        print(f"5. Second question: {second_question[:50]}...")
        assert "investable assets" in second_question.lower()
        
        # Step 6: Answer second question (3)
        state["messages"].append({"role": "user", "content": "3"})
        state = graph.invoke(state)
        third_question = state["messages"][-1]["content"]
        print(f"6. Third question: {third_question[:100]}...")
        assert "investment" in third_question.lower() and "horizon" in third_question.lower()
        
        # Step 7: Answer third question (3)
        state["messages"].append({"role": "user", "content": "3"})
        state = graph.invoke(state)
        fourth_question = state["messages"][-1]["content"]
        print(f"7. Fourth question: {fourth_question[:100]}...")
        assert "early withdrawals" in fourth_question.lower()
        
        # Step 8: Answer fourth question (3)
        state["messages"].append({"role": "user", "content": "3"})
        state = graph.invoke(state)
        fifth_question = state["messages"][-1]["content"]
        print(f"8. Fifth question: {fifth_question[:50]}...")
        assert "investment knowledge" in fifth_question.lower()
        
        # Step 9: Answer fifth question (3)
        state["messages"].append({"role": "user", "content": "3"})
        state = graph.invoke(state)
        sixth_question = state["messages"][-1]["content"]
        print(f"9. Sixth question: {sixth_question[:50]}...")
        assert "growth versus income" in sixth_question.lower()
        
        # Step 10: Answer sixth question (3)
        state["messages"].append({"role": "user", "content": "3"})
        state = graph.invoke(state)
        seventh_question = state["messages"][-1]["content"]
        print(f"10. Seventh question: {seventh_question[:50]}...")
        assert "market crashes" in seventh_question.lower()
        
        # Step 11: User asks "why" -> should get explanation
        state["messages"].append({"role": "user", "content": "why"})
        state = graph.invoke(state)
        why_explanation = state["messages"][-1]["content"]
        print(f"11. Why explanation: {why_explanation[:50]}...")
        assert "risk preference" in why_explanation.lower()
        assert "market crashes" in why_explanation.lower()
        
        # Step 12: Answer seventh question (2)
        state["messages"].append({"role": "user", "content": "2"})
        state = graph.invoke(state)
        final_result = state["messages"][-1]["content"]
        print(f"12. Final result: {final_result[:50]}...")
        assert "preliminary portfolio guidance" in final_result.lower()
        assert "Equity 25.0%" in final_result
        assert "Bonds 75.0%" in final_result
        
        # Verify the risk state was updated with questionnaire results
        assert state.get("risk") is not None
        assert state.get("risk", {}).get("equity") == 0.25  # Should be updated from questionnaire
        
        # Step 13: User says "proceed" -> should move to portfolio agent
        state["messages"].append({"role": "user", "content": "proceed"})
        state = graph.invoke(state)
        portfolio_start = state["messages"][-1]["content"]
        print(f"13. Portfolio start: {portfolio_start[:50]}...")
        assert "asset-class portfolio" in portfolio_start.lower()
        assert "mean-variance optimization" in portfolio_start.lower()
        
        print("✅ Comprehensive risk assessment flow completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_comprehensive_risk_flow()
