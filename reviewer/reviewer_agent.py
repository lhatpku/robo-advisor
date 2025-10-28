"""
Reviewer Agent - Validates completion and manages flow between phases
"""

from typing import Dict, Any
from langchain_openai import ChatOpenAI
from state import AgentState
from pydantic import BaseModel
from prompts.reviewer_prompts import (
    ReviewerIntent,
    INTENT_CLASSIFICATION_PROMPT,
    ReviewerMessages
)
from .reviewer_utils import ReviewerUtils


class ReviewerAgent:
    """
    Reviewer agent that validates completion of each phase and manages flow.
    Simplified version: validates phase completion, shows final summary when all complete.
    """
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.utils = ReviewerUtils()
        self._structured_llm = llm.with_structured_output(ReviewerIntent).bind(temperature=0.0)
    
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
    
    def _classify_intent(self, user_input: str) -> ReviewerIntent:
        """Classify user intent using LLM with structured output."""
        prompt = INTENT_CLASSIFICATION_PROMPT.format(user_input=user_input)
        return self._structured_llm.invoke(prompt)
    
    def step(self, state: AgentState) -> AgentState:
        """
        Main step function for the reviewer agent.
        Simplified logic:
        1. If all phases complete: show summary and handle user input
        2. Otherwise: validate output and update state
        """
        # Validate all phases
        phases = ["risk", "portfolio", "investment", "trading"]
        validation_results = {}
        
        for phase in phases:
            is_complete, feedback = self.utils.validate_phase_completion(state, phase)
            validation_results[phase] = (is_complete, feedback)
        
        # Check if all phases are complete
        all_complete = all(is_complete for is_complete, _ in validation_results.values())
        
        if all_complete or state.get("all_phases_complete", False):
            # All phases complete - show final summary and handle user input
            if not state.get("all_phases_complete"):
                # First time showing summary
                state["all_phases_complete"] = True
                
                # Generate and show summary with options
                completion_message = self.utils.generate_final_completion_message(state)
                state["messages"].append({
                    "role": "ai",
                    "content": f"{completion_message}\n\n{ReviewerMessages.final_summary_with_options()}"
                })
                self._set_status(state, "reviewer", awaiting_input=True)
                return state
            
            # Already shown summary - check for user input
            msgs = state.get("messages", [])
            if msgs and msgs[-1].get("role") == "user":
                last_user = msgs[-1]["content"]
                intent = self._classify_intent(last_user)
                
                if intent.action == "start_over":
                    self.utils.reset_state(state)
                    state["messages"] = []  # Clear all messages first
                    state["messages"].append({
                        "role": "ai",
                        "content": ReviewerMessages.start_over_message()
                    })
                    self._set_status(state, "reviewer", done=False, awaiting_input=True)
                    return state
                
                elif intent.action == "finish":
                    state["messages"].append({
                        "role": "ai",
                        "content": ReviewerMessages.thank_you_message()
                    })
                    self._set_status(state, "reviewer", awaiting_input=True)
                    return state
            
        else:
            # Not all complete - just validate and update state
            # Check if user said "proceed" (coming from entry agent)
            next_phase = self.utils.get_next_phase(state)
            
            if next_phase:
                # Update state to proceed to next phase
                if "ready_to_proceed" not in state or state["ready_to_proceed"] is None:
                    state["ready_to_proceed"] = {}
                state["ready_to_proceed"][next_phase] = True
                state["next_phase"] = next_phase
        
            self._set_status(state, "reviewer", awaiting_input=False)

        # Clear all intent flags
        state["intent_to_risk"] = False
        state["intent_to_portfolio"] = False
        state["intent_to_investment"] = False
        state["intent_to_trading"] = False

        return state
    
    def router(self, state: AgentState) -> str:
        """
        Router function that determines the next step.
        """
        status = self._get_status(state, "reviewer")
        
        # If awaiting input, stay here
        if status["awaiting_input"]:
            return "__end__"
        
        # Otherwise route to entry agent
        return "robo_entry"
