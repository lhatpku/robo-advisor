# portfolio/portfolio_agent.py
from __future__ import annotations
from typing import Dict, Any, Optional, Literal
import os
from portfolio.config import get_expected_returns, get_covariance_matrix, DEFAULT_LAMBDA, DEFAULT_CASH_RESERVE, get_cash_reserve_constraints, validate_cash_reserve
from portfolio.portfolio_manager import PortfolioManager
from state import AgentState
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class PortfolioIntent(BaseModel):
    """Intent classification for portfolio agent user input."""
    action: Literal["set_lambda", "set_cash", "run_optimization", "review", "proceed", "unknown"] = Field(
        description="The action the user wants to perform"
    )
    lambda_value: Optional[float] = Field(
        default=None, 
        description="Lambda value if user wants to set lambda"
    )
    cash_value: Optional[float] = Field(
        default=None, 
        description="Cash reserve value if user wants to set cash"
    )
    confidence: float = Field(
        default=0.0,
        description="Confidence score for the intent classification (0-1)"
    )


class PortfolioAgent:
    """
    Portfolio management agent that handles portfolio optimization
    and parameter setting based on user input.
    """
    
    def __init__(self, llm: ChatOpenAI):
        """
        Initialize the PortfolioAgent.
        
        Args:
            llm: ChatOpenAI instance for generating responses
        """
        self.llm = llm
        self.portfolio_manager = PortfolioManager(llm)
        self._structured_llm = llm.with_structured_output(PortfolioIntent).bind(temperature=0.0)
        
        # Local parameters that persist across method calls
        self._lambda = DEFAULT_LAMBDA
        self._cash_reserve = DEFAULT_CASH_RESERVE
        
    
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
    
    def _classify_intent(self, user_input: str) -> PortfolioIntent:
        """Classify user intent using LLM with structured output."""
        prompt = f"""
You are a portfolio optimization assistant. Classify the user's intent from their input.

User input: "{user_input}"

Available actions:
- set_lambda: User wants to set the lambda parameter (e.g., "set lambda to 1.5", "lambda 2")
- set_cash: User wants to set the cash reserve parameter (e.g., "set cash to 0.03", "cash 0.02")
- run_optimization: User wants to run portfolio optimization (e.g., "run", "optimize", "go")
- review: User wants to review current portfolio (e.g., "review", "show", "display")
- proceed: User wants to proceed to next step (e.g., "proceed", "next", "continue", "looks good", "satisfied", "go ahead", "move on", "fine as is", "I'm done")
- unknown: Intent is unclear or not related to portfolio optimization

Extract the specific values if setting parameters. For cash reserve, use values between 0.02 and 0.05.
For lambda, use positive values typically between 0.1 and 10.

Examples:
- "set cash as 0.02" -> action: set_cash, cash_value: 0.02
- "lambda 1.5" -> action: set_lambda, lambda_value: 1.5
- "run" -> action: run_optimization
- "review" -> action: review
- "proceed" -> action: proceed
- "looks good" -> action: proceed
- "I'm satisfied" -> action: proceed
- "hello" -> action: unknown
"""
        
        try:
            intent = self._structured_llm.invoke(prompt)
            return intent
        except Exception as e:
            print(f"Error classifying intent: {e}")
            return PortfolioIntent(action="unknown", confidence=0.0)
    
    def _format_portfolio(self, portfolio: Dict[str, float]) -> str:
        """Return a compact markdown table of weights sorted by weight desc."""
        if not portfolio:
            return "_(no positions)_"
        items = sorted(portfolio.items(), key=lambda kv: kv[1], reverse=True)
        lines = ["| Asset Class | Weight |", "|---|---:|"]
        for k, v in items:
            lines.append(f"| {k.replace('_',' ')} | {v*100:.2f}% |")
        total = sum(portfolio.values()) * 100
        lines.append(f"| **Total** | **{total:.2f}%** |")
        return "\n".join(lines)


    def step(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main step function for the portfolio agent.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated agent state
        """
        # Initialize global state if first time
        status = self._get_status(state, "portfolio")
        if not status["awaiting_input"] and not status["done"]:
            self._set_status(state, "portfolio", awaiting_input=True, done=False)
        
        # Use global state for persistence across graph invocations
        status = self._get_status(state, "portfolio")
        if not status["awaiting_input"]:
            self._set_status(state, "portfolio", awaiting_input=True)
        if not status["done"]:
            self._set_status(state, "portfolio", done=False)
        
        risk = state.get("risk") or {}
        if not risk:
            state["messages"].append({"role":"ai","content":"I need the equity/bond recommendation from the risk Agent before I can build the portfolio."})
            return state

        # Use instance variables for current parameters
        lam = self._lambda
        cash_reserve = self._cash_reserve
        
        # Get dynamic constraints from config
        min_cash, max_cash = get_cash_reserve_constraints()
        
        # Only act on USER turns
        if not state.get("messages") or state["messages"][-1].get("role") != "user":
            return state

        last_user = state["messages"][-1].get("content", "")
        
        # Classify user intent
        intent = self._classify_intent(last_user)
        
        # Handle different intents
        if intent.action == "set_lambda":
            if intent.lambda_value is not None:
                self._lambda = intent.lambda_value
                state["messages"].append({
                    "role": "ai", 
                    "content": f"✅ Set lambda to {intent.lambda_value}. Current parameters: • Lambda: {intent.lambda_value} • Cash Reserve: {cash_reserve:.2f}\n\nSay 'run' to optimize or 'set cash to X' to adjust further."
                })
            else:
                state["messages"].append({
                    "role": "ai", 
                    "content": "Please specify a lambda value. For example: 'set lambda to 1.5' or 'lambda 2'"
                })
            self._set_status(state, "portfolio", awaiting_input=True)
            return state
            
        elif intent.action == "set_cash":
            if intent.cash_value is not None:
                # Validate cash value against constraints
                if min_cash <= intent.cash_value <= max_cash:
                    self._cash_reserve = intent.cash_value
                    state["messages"].append({
                        "role": "ai", 
                        "content": f"✅ Set cash reserve to {intent.cash_value:.2f}. Current parameters: • Lambda: {lam} • Cash Reserve: {intent.cash_value:.2f}\n\nSay 'run' to optimize or 'set lambda to X' to adjust further."
                    })
                else:
                    state["messages"].append({
                        "role": "ai", 
                        "content": f"❌ Cash reserve must be between {min_cash:.2f} and {max_cash:.2f}. You entered {intent.cash_value:.2f}."
                    })
            else:
                state["messages"].append({
                    "role": "ai", 
                    "content": f"Please specify a cash reserve value between {min_cash:.2f} and {max_cash:.2f}. For example: 'set cash to 0.03' or 'cash 0.02'"
                })
            self._set_status(state, "portfolio", awaiting_input=True)
            return state
            
        elif intent.action == "run_optimization":
            # Run portfolio optimization
            min_cash, max_cash = get_cash_reserve_constraints()
            clamped_cash = min(max_cash, max(min_cash, cash_reserve))
            
            call_args = {
                "risk_equity": float(risk.get("equity", 0.0)),
                "risk_bonds": float(risk.get("bond", 0.0)),
                "lam": lam,
                "cash_reserve": clamped_cash,
            }
            res = self.portfolio_manager.execute_tool_call({"tool":"mean_variance_optimizer","args":call_args})
            if isinstance(res, dict) and res:
                state["portfolio"] = res
                note = "" if clamped_cash == cash_reserve else f" (cash_reserve {cash_reserve:.2f} was clamped to {clamped_cash:.2f})"
                table = self._format_portfolio(res)
                state["messages"].append({
                    "role":"ai",
                    "content": f"Optimization complete{note}. I've built your asset-class portfolio.\n\n{table}\n\n Review weights or proceed to ETF selection?"
                })
                executed_optimizer = True
                # Set awaiting_input to True to allow review/editing, but not done yet
                self._set_status(state, "portfolio", awaiting_input=True, done=False)
            else:
                state["messages"].append({"role":"ai","content": "Portfolio optimization failed. Please try again."})
                self._set_status(state, "portfolio", awaiting_input=True)
            return state
            
        elif intent.action == "review":
            if state.get("portfolio"):
                # Show current portfolio with editing options
                table = self._format_portfolio(state["portfolio"])
                state["messages"].append({
                    "role":"ai",
                    "content": f"**Your current portfolio:**\n\n{table}\n\n**Current parameters:** • Lambda: {lam} • Cash Reserve: {cash_reserve:.2f}\n\n**What would you like to do next?**\n• **Edit** parameters: say 'set lambda to X' or 'set cash to Y'\n• **Re-optimize**: say 'run' to optimize with current parameters\n• **Proceed** to ETF selection: say 'proceed'"
                })
            else:
                # Show intro message if no portfolio exists
                state["messages"].append({
                    "role":"ai",
                    "content": (
                        "Here's the plan: I'll build an asset-class portfolio using mean-variance optimization.\n"
                        f"Defaults are **lambda = {lam}** and **cash_reserve = {cash_reserve:.2f}**.\n"
                        f"Say \"set lambda to 1\", \"set cash to {max_cash:.2f}\", or just \"run\" to optimize now."
                    )
                })
            self._set_status(state, "portfolio", awaiting_input=True)
            return state
            
        elif intent.action == "proceed":
            if state.get("portfolio"):
                state["messages"].append({"role": "ai", "content": "Great — proceeding to the next step."})
                self._set_status(state, "portfolio", done=True, awaiting_input=False)
            else:
                # Show intro message if no portfolio exists
                state["messages"].append({
                    "role": "ai",
                    "content": (
                        "Here's the plan: I'll build an asset-class portfolio using mean-variance optimization.\n"
                        f"Defaults are **lambda = {lam}** and **cash_reserve = {cash_reserve:.2f}**.\n"
                        f"Say \"set lambda to 1\", \"set cash to {max_cash:.2f}\", or just \"run\" to optimize now."
                    )
                })
                self._set_status(state, "portfolio", awaiting_input=True)
            return state
            
        else:  # unknown or unclear intent
            # Show intro message for unclear inputs
            state["messages"].append({
                "role":"ai",
                "content": (
                    "Here's the plan: I'll build an asset-class portfolio using mean-variance optimization.\n"
                    f"Defaults are **lambda = {lam}** and **cash_reserve = {cash_reserve:.2f}**.\n"
                    f"Say \"set lambda to 1\", \"set cash to {max_cash:.2f}\", or just \"run\" to optimize now."
                )
            })
            self._set_status(state, "portfolio", awaiting_input=True)
            return state

    def router(self, state: AgentState) -> str:
        """
        Route based on portfolio agent state.
        """
        # If awaiting input, go to end to wait for user input
        status = self._get_status(state, "portfolio")
        if status["awaiting_input"]:
            return "__end__"
        
        # If portfolio exists and user wants to proceed, go to reviewer
        if state.get("portfolio") and status["done"]:
            return "reviewer_agent"
        
        # If portfolio doesn't exist yet, go to end to wait for user input
        return "__end__"

