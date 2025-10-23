# investment/investment_agent.py
from __future__ import annotations
from typing import Dict, Any, List, Optional, Literal
from langchain_openai import ChatOpenAI
from state import AgentState
from investment.investment_utils import InvestmentUtils
from pydantic import BaseModel, Field
from prompts.investment_prompts import (
    INTENT_CLASSIFICATION_PROMPT,
    InvestmentMessages
)

# Intent Classification Model
class InvestmentIntent(BaseModel):
    """Intent classification for investment agent user input."""
    action: Literal[
        "create_investment",      # Start fund selection process
        "select_criteria",        # Choose selection criteria (balanced, low cost, etc.)
        "review_investment",      # Show current investment portfolio
        "edit_asset_class",       # Edit specific asset class
        "analyze_fund",           # Analyze specific fund ticker
        "proceed",                # Move to next phase
        "unknown"                 # Unclear intent
    ] = "unknown"
    
    criteria: Optional[str] = Field(
        default=None,
        description="Selection criteria if action is select_criteria (balanced, low_cost, high_performance, low_risk)"
    )
    
    asset_class: Optional[str] = Field(
        default=None,
        description="Asset class name if action is edit_asset_class"
    )
    
    ticker: Optional[str] = Field(
        default=None,
        description="Fund ticker symbol if action is analyze_fund"
    )
    

class InvestmentAgent:
    """
    Investment agent that handles the conversion of asset-class portfolios
    into tradeable portfolios with specific funds/ETFs.
    """
    
    def __init__(self, llm: ChatOpenAI):
        """
        Initialize the InvestmentAgent.
        
        Args:
            llm: ChatOpenAI instance for generating responses
        """
        self.llm = llm
        self.utils = InvestmentUtils(llm)
        
        # Structured LLM for intent classification
        self._structured_llm = llm.with_structured_output(InvestmentIntent).bind(temperature=0.0)
        
        # Local state for mode tracking
        self._investment_criteria_selection = False
        self._investment_edit_mode = False
        self._investment_edit_asset_class = None
        self._investment_edit_options = None
        
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
    
    def _classify_intent(self, user_input: str) -> InvestmentIntent:
        """Classify user intent using LLM with structured output."""
        prompt = INTENT_CLASSIFICATION_PROMPT.format(user_input=user_input)
        
        try:
            intent = self._structured_llm.invoke(prompt)
            return intent
        except Exception as e:
            print(f"Error classifying intent: {e}")
            return InvestmentIntent(action="unknown")
    
    def step(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main step function for the investment agent.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated agent state
        """
        # Initialize global state if first time
        status = self._get_status(state, "investment")
        if not status["awaiting_input"] and not status["done"]:
            self._set_status(state, "investment", awaiting_input=True, done=False)
        
        # Check if portfolio exists
        portfolio = state.get("portfolio", {})
        if not portfolio:
            state["messages"].append({
                "role": "ai", 
                "content": InvestmentMessages.need_portfolio_data()
            })
            return state
        
        # Only act on USER turns
        if not state.get("messages") or state["messages"][-1].get("role") != "user":
            return state

        last_user = state["messages"][-1].get("content", "")
        
        # Special handling for edit mode - if user is selecting a fund option by number
        if self._investment_edit_mode and last_user.isdigit():
            edit_data = {
                "asset_class": self._investment_edit_asset_class,
                "options": self._investment_edit_options
            }
            result = self.utils.handle_edit_mode(state, edit_data)
            
            # Check if edit was successful (user selected a valid option)
            last_ai_message = result.get("messages", [])[-1] if result.get("messages") else None
            if last_ai_message and "Updated" in last_ai_message.get("content", ""):
                # Clear edit mode after successful selection
                self._investment_edit_mode = False
                self._investment_edit_asset_class = None
                self._investment_edit_options = None
            
            return result
        
        # Classify user intent
        intent = self._classify_intent(last_user)
        
        # Handle different intents
        if intent.action == "create_investment":
            self._investment_criteria_selection = True
            return self.utils.create_initial_investment(state)
        
        elif intent.action == "select_criteria":
            result = self.utils.handle_criteria_selection(state, intent.criteria)
            # Clear criteria selection mode after handling
            self._investment_criteria_selection = False
            return result
        
        elif intent.action == "review_investment":
            investment = state.get("investment")
            if investment and isinstance(investment, dict) and investment:
                self.utils.display_investment_portfolio(state, investment)
            else:
                state["messages"].append({
                    "role": "ai",
                    "content": InvestmentMessages.intro_message()
                })
            return state
        
        elif intent.action == "edit_asset_class":
            if state.get("investment"):
                if intent.asset_class:
                    # User specified an asset class, show options
                    edit_data = self.utils.show_asset_class_options(state, intent.asset_class)
                    if edit_data:
                        self._investment_edit_mode = True
                        self._investment_edit_asset_class = edit_data["asset_class"]
                        self._investment_edit_options = edit_data["options"]
                else:
                    state["messages"].append({
                        "role": "ai",
                        "content": InvestmentMessages.edit_asset_class_prompt()
                    })
            else:
                state["messages"].append({
                    "role": "ai",
                    "content": InvestmentMessages.need_investment_first()
                })
            return state
        
        elif intent.action == "analyze_fund":
            if intent.ticker:
                return self.utils.handle_fund_analysis_request(state, intent.ticker)
            else:
                state["messages"].append({
                    "role": "ai",
                    "content": InvestmentMessages.fund_analysis_prompt()
                })
                return state
        
        elif intent.action == "proceed":
            if state.get("portfolio") and state.get("investment"):
                self._set_status(state, "investment", done=True, awaiting_input=False)
                # state["messages"].append({
                #     "role": "ai",
                #     "content": InvestmentMessages.investment_ready()
                # })
                return state
            else:
                state["messages"].append({
                    "role": "ai",
                    "content": InvestmentMessages.need_investment_first()
                })
                return state
        
        else:  # unknown intent
            # Check if we're in criteria selection mode
            if self._investment_criteria_selection:
                return self.utils.create_initial_investment(state)
            
            # Check if we're in edit mode
            if self._investment_edit_mode:
                edit_data = {
                    "asset_class": self._investment_edit_asset_class,
                    "options": self._investment_edit_options
                }
                result = self.utils.handle_edit_mode(state, edit_data)
                # Check if edit was successful (user selected a valid option)
                last_ai_message = result.get("messages", [])[-1] if result.get("messages") else None
                if last_ai_message and "Updated" in last_ai_message.get("content", ""):
                    # Clear edit mode after successful selection
                    self._investment_edit_mode = False
                    self._investment_edit_asset_class = None
                    self._investment_edit_options = None
                return result
            
            # Check if investment already exists
            if state.get("investment"):
                # Check if user mentioned a specific asset class
                investment = state.get("investment")
                self.utils.display_investment_portfolio(state, investment)
            
            # Show help message
            state["messages"].append({
                "role": "ai",
                "content": InvestmentMessages.unclear_intent()
            })
            return state
    
    
    def router(self, state: Dict[str, Any]) -> str:
        """
        Route based on investment agent state.
        """
        # If awaiting input, go to end to wait for user input
        status = self._get_status(state, "investment")
        if status["awaiting_input"]:
            return "__end__"
        
        # If investment exists and user wants to proceed, go to reviewer
        if state.get("investment") and status["done"]:
            return "reviewer_agent"
        
        # If investment doesn't exist yet, go to end to wait for user input
        return "__end__"
