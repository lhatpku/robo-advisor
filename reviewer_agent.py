"""
Reviewer Agent - Validates completion and manages flow between phases
"""

from typing import Dict, Any, List, Tuple
from langchain_openai import ChatOpenAI
from state import AgentState
from prompts.reviewer_prompts import (
    REVIEWER_SYSTEM_PROMPT,
    REVIEWER_VALIDATION_PROMPTS,
    REVIEWER_PROCEED_PROMPTS,
    REVIEWER_COMPLETION_MESSAGE
)
from utils.summary_utils import generate_final_completion_message


class ReviewerAgent:
    """
    Reviewer agent that validates completion of each phase and manages flow.
    Only agent that sets ready_to_proceed and all_phases_complete flags.
    """
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        
        # Local state management
        self._awaiting_input = False
        self._done = False
    
    def _get_status(self, state: AgentState, agent: str) -> Dict[str, bool]:
        """Get status tracking for a specific agent."""
        return state.get("status_tracking", {}).get(agent, {"done": False, "awaiting_input": False})
    
    def _set_status(self, state: AgentState, agent: str, done: bool = None, awaiting_input: bool = None) -> None:
        """Set status tracking for a specific agent."""
        if "status_tracking" not in state:
            state["status_tracking"] = {}
        if agent not in state["status_tracking"]:
            state["status_tracking"][agent] = {"done": False, "awaiting_input": False}
        
        if done is not None:
            state["status_tracking"][agent]["done"] = done
        if awaiting_input is not None:
            state["status_tracking"][agent]["awaiting_input"] = awaiting_input
    
    def _validate_phase_completion(self, state: AgentState, phase: str) -> Tuple[bool, str]:
        """
        Validate if a specific phase is complete.
        
        Returns:
            Tuple of (is_complete, feedback_message)
        """
        if phase == "risk":
            risk = state.get("risk")
            if risk and "equity" in risk and "bond" in risk:
                return True, REVIEWER_VALIDATION_PROMPTS["risk"]["complete"]
            return False, REVIEWER_VALIDATION_PROMPTS["risk"]["incomplete"]
        
        elif phase == "portfolio":
            portfolio = state.get("portfolio")
            if portfolio and isinstance(portfolio, dict) and len(portfolio) > 0:
                return True, REVIEWER_VALIDATION_PROMPTS["portfolio"]["complete"]
            return False, REVIEWER_VALIDATION_PROMPTS["portfolio"]["incomplete"]
        
        elif phase == "investment":
            investment = state.get("investment")
            if investment and isinstance(investment, dict) and len(investment) > 0:
                return True, REVIEWER_VALIDATION_PROMPTS["investment"]["complete"]
            return False, REVIEWER_VALIDATION_PROMPTS["investment"]["incomplete"]
        
        elif phase == "trading":
            trading = state.get("trading_requests")
            if trading and isinstance(trading, dict) and "trading_requests" in trading and len(trading.get("trading_requests", [])) > 0:
                return True, REVIEWER_VALIDATION_PROMPTS["trading"]["complete"]
            return False, REVIEWER_VALIDATION_PROMPTS["trading"]["incomplete"]
        
        return False, f"Unknown phase: {phase}"
    
    def _get_next_phase(self, state: AgentState) -> str:
        """
        Determine the next phase to proceed to.
        
        Returns:
            Next phase name or None if all complete
        """
        phases = ["risk", "portfolio", "investment", "trading"]
        
        for phase in phases:
            is_complete, _ = self._validate_phase_completion(state, phase)
            if not is_complete:
                return phase
        
        return None  # All phases complete
    
    def _classify_user_response(self, user_input: str) -> str:
        """
        Classify user response to proceed/edit question.
        
        Returns:
            "proceed", "edit", or "unclear"
        """
        user_lower = user_input.lower()
        
        proceed_words = ["proceed", "continue", "next", "yes", "ok", "go ahead", "ready"]
        edit_words = ["edit", "change", "modify", "redo", "back", "no"]
        
        if any(word in user_lower for word in proceed_words):
            return "proceed"
        elif any(word in user_lower for word in edit_words):
            return "edit"
        else:
            return "unclear"
    
    def step(self, state: AgentState) -> AgentState:
        """
        Main step function for the reviewer agent.
        Validates completion and sets ready_to_proceed or all_phases_complete flags.
        """
        # Initialize global state if first time
        status = self._get_status(state, "reviewer")
        if not status["awaiting_input"] and not status["done"]:
            self._set_status(state, "reviewer", awaiting_input=False, done=False)
        
        # Get validation status for all phases
        phases = ["risk", "portfolio", "investment", "trading"]
        validation_results = {}
        
        for phase in phases:
            is_complete, feedback = self._validate_phase_completion(state, phase)
            validation_results[phase] = (is_complete, feedback)
        
        # Check if any phase is incomplete and needs to continue
        incomplete_phases = [phase for phase, (is_complete, _) in validation_results.items() if not is_complete]
        
        if incomplete_phases:
            # Find the first incomplete phase that has been started
            started_phases = []
            if state.get("risk"):
                started_phases.append("risk")
            if state.get("portfolio"):
                started_phases.append("portfolio")
            if state.get("investment"):
                started_phases.append("investment")
            if state.get("trading_requests"):
                started_phases.append("trading")
            
            # Find incomplete phases that have been started
            incomplete_started_phases = [phase for phase in incomplete_phases if phase in started_phases]
            
            if incomplete_started_phases:
                # Route back to the first incomplete phase that has been started
                phase_name = incomplete_started_phases[0]
                _, feedback = validation_results[phase_name]
                
                # Reset done flag for the phase being continued
                self._set_status(state, "reviewer", done=False)
                
                # Provide feedback to user
                state["messages"].append({
                    "role": "ai",
                    "content": f"**Reviewer Feedback:** {feedback}\n\nPlease complete this step before proceeding to the next phase."
                })
                
                return state
        
        # All phases are complete or no incomplete phases found
        # Determine next ready phase
        next_phase = self._get_next_phase(state)
        
        if next_phase:
            # There's a next phase ready - update ready_to_proceed to indicate this phase is ready
            if "ready_to_proceed" not in state or state["ready_to_proceed"] is None:
                state["ready_to_proceed"] = {}
            
            # Mark the next agent as ready to proceed
            state["ready_to_proceed"][next_phase] = True
            
            # Set next_phase for entry agent
            state["next_phase"] = next_phase
            
            # Clear all intent flags so entry agent can process the ready_to_proceed state
            state["intent_to_risk"] = False
            state["intent_to_portfolio"] = False
            state["intent_to_investment"] = False
            state["intent_to_trading"] = False
        else:
            # All phases are complete
            state["all_phases_complete"] = True
            state["next_phase"] = None
            
            # Check if user has provided input for final completion options
            msgs = state.get("messages", [])
            last_is_user = bool(msgs) and msgs[-1].get("role") == "user"
            last_user = msgs[-1]["content"] if last_is_user else ""
            
            if last_user:
                if any(word in last_user.lower() for word in ["start over", "new portfolio", "reset"]):
                    # User wants to start over - reset all state and route to entry agent
                    state["risk"] = None
                    state["portfolio"] = None
                    state["investment"] = None
                    state["trading_requests"] = None
                    state["all_phases_complete"] = False
                    state["ready_to_proceed"] = None
                    state["intent_to_risk"] = False
                    state["intent_to_portfolio"] = False
                    state["intent_to_investment"] = False
                    state["intent_to_trading"] = False
                    state["entry_greeted"] = False
                    state["summary_shown"] = {
                        "risk": False,
                        "portfolio": False,
                        "investment": False,
                        "trading": False
                    }
                    
                    # Reset status tracking for all agents
                    state["status_tracking"] = {
                        "risk": {"done": False, "awaiting_input": False},
                        "portfolio": {"done": False, "awaiting_input": False},
                        "investment": {"done": False, "awaiting_input": False},
                        "trading": {"done": False, "awaiting_input": False},
                        "reviewer": {"done": False, "awaiting_input": False}
                    }
                    self._set_status(state, "reviewer", done=False, awaiting_input=False)
                    
                    # Clear messages and start fresh
                    state["messages"] = []
                    
                    state["messages"].append({
                        "role": "ai",
                        "content": "Great! Let's start fresh with a new portfolio. How can I assist you today?"
                    })
                    return state
                elif any(word in last_user.lower() for word in ["proceed", "continue", "yes", "ok", "ready"]):
                    # User confirmed completion - show final message and route to entry agent
                    state["all_phases_complete"] = False  # Reset to allow entry agent to handle
                    state["messages"].append({
                        "role": "ai",
                        "content": "🎉 **Portfolio Planning Complete!**\n\nThank you for using our robo-advisor! Your personalized investment plan is ready for execution.\n\nType **'start over'** to create a new portfolio, or **'exit'** to end the session."
                    })
                    return state
                elif "review" in last_user.lower():
                    # User wants to review - show the completion message again
                    pass  # Will show completion message below
                else:
                    # Unknown command - show help
                    state["messages"].append({
                        "role": "ai",
                        "content": "I'm not sure what you'd like to do. You can:\n• **Start over** with a new portfolio by saying 'start over'\n• **Proceed** to confirm completion\n• **Review** to see the summary again"
                    })
                    return state
            
            # Generate comprehensive completion summary (for first time or review)
            completion_message = generate_final_completion_message(state)
            state["messages"].append({
                "role": "ai",
                "content": completion_message
            })
        
        return state
    
    def router(self, state: AgentState) -> str:
        """
        Router function that determines the next step based on validation results.
        
        Returns:
            Next node name
        """
        
        # Check if all phases are complete
        if state.get("all_phases_complete"):
            # Check if user provided input for final completion options
            msgs = state.get("messages", [])
            last_is_user = bool(msgs) and msgs[-1].get("role") == "user"
            last_user = msgs[-1]["content"] if last_is_user else ""
            
            if last_user:
                if any(word in last_user.lower() for word in ["start over", "new portfolio", "reset"]):
                    # User wants to start over - route to entry agent
                    return "robo_entry"
                elif any(word in last_user.lower() for word in ["proceed", "continue", "yes", "ok", "ready"]):
                    # User confirmed completion - route to entry agent
                    return "robo_entry"
                else:
                    # User said review or other command - stay in reviewer agent
                    return "__end__"
            else:
                # No user input yet - stay in reviewer agent to show completion message
                return "__end__"
        
        # All phases are complete or no incomplete phases found - route to entry agent
        return "robo_entry"
