#!/usr/bin/env python3
"""
Test the comprehensive risk assessment flow from start to finish.
This test follows the actual flow through the graph and checks state changes
and output formats rather than exact message matching.
"""
from dotenv import load_dotenv
load_dotenv()

import sys
import os
# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from langchain_openai import ChatOpenAI
from app import build_graph
from state import AgentState

def test_comprehensive_risk_flow():
    """
    Test the complete risk assessment flow using questionnaire:
    1. Entry agent shows risk phase summary
    2. User says "proceed" -> routes to risk agent
    3. Risk agent shows mode selection
    4. User says "guidance" -> starts questionnaire
    5. Complete questionnaire with answers
    6. User asks "why" -> gets explanation
    7. Complete questionnaire -> risk set, routes to reviewer
    8. Reviewer validates -> routes to entry agent
    9. Entry agent shows portfolio summary
    10. User says "proceed" -> routes to portfolio agent
    """
    print("=== Test: Comprehensive Risk Assessment Flow (Questionnaire) ===")
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    graph = build_graph(llm)
    
    state: AgentState = {
        "messages": [],
        "answers": {},
        "risk": None,
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
        "next_phase": "risk",
        "summary_shown": {
            "risk": False,
            "portfolio": False,
            "investment": False,
            "trading": False
        },
        "status_tracking": {
            "risk": {"done": False, "awaiting_input": False},
            "portfolio": {"done": False, "awaiting_input": False},
            "investment": {"done": False, "awaiting_input": False},
            "trading": {"done": False, "awaiting_input": False},
            "reviewer": {"done": False, "awaiting_input": False}
        },
        "correlation_id": None
    }
    
    try:
        # Step 1: Initial invoke - Entry agent shows risk phase summary
        state = graph.invoke(state)
        assert len(state["messages"]) > 0, "Should have at least one message"
        summary = state["messages"][-1]["content"]
        print(f"1. Phase summary received (length: {len(summary)})")
        # Check that it's a summary message (contains risk-related keywords)
        assert any(keyword in summary.lower() for keyword in ["risk", "assessment", "profile", "start"]), \
            f"Summary should mention risk/assessment/profile, got: {summary[:100]}"
        
        # Step 2: User says "proceed" -> should route to risk agent
        state["messages"].append({"role": "user", "content": "proceed"})
        state = graph.invoke(state)
        assert state.get("intent_to_risk") == True, "Should set intent_to_risk flag"
        mode_selection = state["messages"][-1]["content"]
        print(f"2. Mode selection received (length: {len(mode_selection)})")
        # Check that risk agent responded (should mention options or guidance)
        assert len(mode_selection) > 50, "Mode selection should be substantial"
        
        # Step 3: User says "guidance" -> should start questionnaire
        state["messages"].append({"role": "user", "content": "guidance"})
        state = graph.invoke(state)
        first_question = state["messages"][-1]["content"]
        print(f"3. First question received (length: {len(first_question)})")
        # Check that it's a question (contains question marks or numbered options)
        assert "?" in first_question or any(str(i) in first_question for i in range(1, 4)), \
            "Should be a question with options"
        
        # Step 4-6: Answer first 3 questions
        for i, answer in enumerate(["3", "3", "3"], start=4):
            state["messages"].append({"role": "user", "content": answer})
            state = graph.invoke(state)
            response = state["messages"][-1]["content"]
            print(f"{i}. Question {i-3} answered, response length: {len(response)}")
            # Check that we're still in questionnaire or got result
            assert len(response) > 20, "Should have a response"
        
        # Step 7: User asks "why" -> should get explanation (doesn't count as answer)
        state["messages"].append({"role": "user", "content": "why"})
        state = graph.invoke(state)
        why_explanation = state["messages"][-1]["content"]
        print(f"7. Why explanation received (length: {len(why_explanation)})")
        # Check that it's an explanation (substantial text)
        assert len(why_explanation) > 50, "Explanation should be substantial"
        
        # Step 8: Answer the question that was explained (need to answer it after "why")
        state["messages"].append({"role": "user", "content": "2"})
        state = graph.invoke(state)
        response = state["messages"][-1]["content"]
        print(f"8. Question answered after why, response length: {len(response)}")
        
        # Step 9-11: Answer remaining questions (total 7 questions, so need 3 more)
        remaining_answers = ["3", "3", "3"]  # Answer remaining 3 questions
        for i, answer in enumerate(remaining_answers, start=9):
            state["messages"].append({"role": "user", "content": answer})
            state = graph.invoke(state)
            response = state["messages"][-1]["content"]
            print(f"{i}. Question answered, response length: {len(response)}")
            # After all 7 questions, risk should be set
            if state.get("risk") is not None:
                print(f"   Risk was set after question {i-8}")
                break
        
        # Step 12: Verify risk was set (should be set after all 7 questions)
        if state.get("risk") is None:
            # Maybe need one more invoke or proceed
            print("   Risk not set yet, checking if questionnaire is complete...")
            # Check answers count
            answers_count = len(state.get("answers", {}))
            print(f"   Answers collected: {answers_count}")
        
        assert state.get("risk") is not None, f"Risk should be set after questionnaire. Answers: {len(state.get('answers', {}))}"
        assert "equity" in state.get("risk", {}), "Risk should have equity field"
        assert "bond" in state.get("risk", {}) or "bonds" in state.get("risk", {}), "Risk should have bond field"
        equity = state.get("risk", {}).get("equity")
        assert equity is not None and 0 <= equity <= 1, f"Equity should be between 0 and 1, got {equity}"
        print(f"11. Risk set: equity={equity:.2f}")
        
        # Step 13: User says "proceed" -> should route to reviewer, then entry shows portfolio summary
        state["messages"].append({"role": "user", "content": "proceed"})
        state = graph.invoke(state)
        
        # After proceed, risk should be done and route to reviewer
        # Reviewer will then route to entry, which shows portfolio summary
        # May need multiple invokes to get through the flow
        max_iterations = 3
        iteration = 0
        while iteration < max_iterations:
            risk_status = state.get("status_tracking", {}).get("risk", {})
            if risk_status.get("done") == True and state.get("next_phase") == "portfolio":
                break
            # If not done, invoke again (might be routing through reviewer)
            if state.get("messages") and state["messages"][-1].get("role") == "ai":
                state = graph.invoke(state)
            iteration += 1
        
        # Check that risk phase is done
        risk_status = state.get("status_tracking", {}).get("risk", {})
        assert risk_status.get("done") == True, f"Risk phase should be marked as done. Status: {risk_status}"
        
        # Check that next_phase was updated to portfolio
        assert state.get("next_phase") == "portfolio", f"Next phase should be portfolio, got: {state.get('next_phase')}"
        
        # Check that portfolio summary was shown or portfolio agent was reached
        portfolio_summary = state["messages"][-1]["content"] if state.get("messages") else ""
        print(f"13. Portfolio phase started, message length: {len(portfolio_summary)}")
        # Should mention portfolio or construction or optimization
        assert any(keyword in portfolio_summary.lower() for keyword in ["portfolio", "construction", "optimization", "asset", "start"]), \
            f"Should mention portfolio-related terms, got: {portfolio_summary[:100]}"
        
        print("SUCCESS: Comprehensive risk assessment flow completed successfully!")
        
    except Exception as e:
        print(f"FAILED: Test failed: {e}")
        import traceback
        traceback.print_exc()
        print(f"\nCurrent state:")
        print(f"  Risk: {state.get('risk')}")
        print(f"  Answers: {len(state.get('answers', {}))} questions answered")
        print(f"  Next phase: {state.get('next_phase')}")
        print(f"  Status tracking: {state.get('status_tracking')}")
        if state.get('messages'):
            print(f"  Last message: {state['messages'][-1]['content'][:200]}")
        raise

if __name__ == "__main__":
    test_comprehensive_risk_flow()
