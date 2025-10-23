"""
Entry agent for the robo-advisor application.

This agent handles initial user interaction and routes to appropriate phases
based on user intent and reviewer agent's next_phase field.
"""

from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from state import AgentState
from prompts.entry_prompts import INTENT_CLASSIFICATION_PROMPT, EntryMessages, EntryIntent


class EntryAgent:
    """Entry agent that handles user interaction and routing."""
    
    def __init__(self, llm: ChatOpenAI):
        """Initialize the entry agent."""
        self.llm = llm
        self._structured_llm = llm.with_structured_output(EntryIntent).bind(temperature=0.0)
    
    def step(self, state: AgentState) -> AgentState:
        """
        Main step function for entry agent.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated agent state
        """

        if (state.get("intent_to_risk") or 
        state.get("intent_to_portfolio") or 
        state.get("intent_to_investment") or 
        state.get("intent_to_trading")):
            return state
        
        # Get next phase from reviewer agent (default to risk)
        next_phase = state.get("next_phase", "risk") or "risk"
        
        # Show summary only if not shown yet
        if not state["summary_shown"].get(next_phase, False):
            state["summary_shown"][next_phase] = True
            return self._show_phase_summary(state, next_phase)
        
        # Only act on USER turns
        if not state.get("messages") or state["messages"][-1].get("role") != "user":
            return state
        
        last_user = state["messages"][-1].get("content", "")
        
        # Classify user intent
        intent = self._classify_intent(last_user)
        
        # Handle different intents
        if intent.action == "proceed":
            return self._handle_proceed_intent(state, next_phase)
        elif intent.action == "learn_more":
            # Use question from intent, or create a default question based on next_phase
            question = intent.question or f"What is {next_phase} and how does it work?"
            return self._handle_learn_more_intent(state, question)
        else:
            # Unknown intent - show help
            state["messages"].append({
                "role": "ai",
                "content": EntryMessages.unclear_intent()
            })
            return state
    
    def _classify_intent(self, user_input: str) -> EntryIntent:
        """Classify user intent using LLM with structured output."""
        prompt = INTENT_CLASSIFICATION_PROMPT.format(user_input=user_input)
        return self._structured_llm.invoke(prompt)
    
    def _handle_proceed_intent(self, state: AgentState, next_phase: str) -> AgentState:
        """Handle when user wants to proceed to next phase."""
        # Set the appropriate intent flag
        if next_phase == "risk":
            state["intent_to_risk"] = True
        elif next_phase == "portfolio":
            state["intent_to_portfolio"] = True
        elif next_phase == "investment":
            state["intent_to_investment"] = True
        elif next_phase == "trading":
            state["intent_to_trading"] = True
        
        return state
    
    def _handle_learn_more_intent(self, state: AgentState, question: str) -> AgentState:
        """Handle when user wants to learn more about something."""
        # For now, provide a general explanation based on the question
        # In the future, this could use RAG to provide more detailed answers
        state["messages"].append({
            "role": "ai",
            "content": f"Great question: {question}\n\nLet me explain this part of the investment planning process..."
        })
        
        # Ask if they want to proceed
        state["messages"].append({
            "role": "ai",
            "content": "Would you like to proceed to the next phase, or do you have other questions?"
        })
        
        return state
    
    def _show_phase_summary(self, state: AgentState, completed_phase: str) -> AgentState:
        """Show summary for a completed phase and ask what to do next."""
        summary_message = EntryMessages.get_stage_summary(completed_phase)
        
        state["messages"].append({
            "role": "ai",
            "content": summary_message
        })
        
        return state
    
    def router(self, state: AgentState) -> str:
        """
        Router function that determines the next step based on intent flags.
        """
        # Check intent flags in order of phases
        if state.get("intent_to_risk"):
            return "risk_agent"
        elif state.get("intent_to_portfolio"):
            return "portfolio_agent"
        elif state.get("intent_to_investment"):
            return "investment_agent"
        elif state.get("intent_to_trading"):
            return "trading_agent"
        
        # No intent flags set - stay in entry agent
        return "__end__"
