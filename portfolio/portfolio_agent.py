# portfolio/portfolio_agent.py
from __future__ import annotations
from typing import Dict, Any
import os
from portfolio.config import get_expected_returns, get_covariance_matrix, DEFAULT_LAMBDA, DEFAULT_CASH_RESERVE, get_cash_reserve_constraints, validate_cash_reserve
from portfolio.portfolio_manager import PortfolioManager
from state import AgentState
from langchain_openai import ChatOpenAI


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

    def infer_proceed_intent(self, user_text: str) -> bool:
        """
        Use the LLM to infer whether the user wants to PROCEED to the next step
        (e.g., ETF selection) rather than review or change anything.

        Returns:
            True  -> proceed
            False -> do not proceed (review/ask/change/unclear)
        """
        system = (
            "Classify if user wants to PROCEED to next step (ETF selection) or continue reviewing.\n\n"
            "PROCEED signals: 'looks good', 'satisfied', 'proceed', 'go ahead', 'continue', 'next step', 'move on', 'fine as is', 'I'm done'\n"
            "NOT proceed: questions, changes, review requests, details\n\n"
            "Return JSON: {\"proceed\": true|false, \"reason\": \"brief explanation\"}"
        )
        user = f"Latest user message:\n{user_text}\n\nReturn only JSON."
        resp = self.llm.invoke([{"role": "system", "content": system},
                               {"role": "user", "content": user}])
        content = getattr(resp, "content", "") if resp else ""
        try:
            import json
            data = json.loads(content)
            return bool(data.get("proceed", False))
        except Exception:
            return False

    def step(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main step function for the portfolio agent.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated agent state
        """
        risk = state.get("risk") or {}
        if not risk:
            state["messages"].append({"role":"ai","content":"I need the equity/bond recommendation from the risk Agent before I can build the portfolio."})
            return state

        # Local parameters only
        lam = DEFAULT_LAMBDA
        cash_reserve = DEFAULT_CASH_RESERVE
        
        # Only act on USER turns
        if not state.get("messages") or state["messages"][-1].get("role") != "user":
            return state

        last_user = state["messages"][-1].get("content", "").lower()

        # Handle review command - show current portfolio or intro message
        if any(word in last_user for word in ["review", "show", "display", "see", "current"]):
            if state.get("portfolio"):
                # Show current portfolio
                table = self._format_portfolio(state["portfolio"])
                state["messages"].append({
                    "role":"ai",
                    "content": f"**Your current portfolio:**\n\n{table}\n\n**What would you like to do next?**\n• **Edit** parameters (lambda, cash) and re-optimize\n• **Proceed** to ETF selection\n• **Go back** to risk assessment"
                })
            else:
                # Show intro message if no portfolio exists
                state["messages"].append({
                    "role":"ai",
                    "content": (
                        "Here's the plan: I'll build an asset-class portfolio using mean-variance optimization.\n"
                        f"Defaults are **lambda = {lam}** and **cash_reserve = {cash_reserve:.2f}**.\n"
                        "Say \"set lambda to 1\", \"set cash to 0.05\", or just \"run\" to optimize now."
                    )
                })
            return state

        # Handle edit commands - show intro message again
        if any(word in last_user for word in ["edit", "change", "modify", "adjust"]):
            state["messages"].append({
                "role":"ai",
                "content": (
                    "Here's the plan: I'll build an asset-class portfolio using mean-variance optimization.\n"
                    f"Defaults are **lambda = {lam}** and **cash_reserve = {cash_reserve:.2f}**.\n"
                    "Say \"set lambda to 1\", \"set cash to 0.05\", or just \"run\" to optimize now."
                )
            })
            return state

        # Show intro message if no portfolio exists yet, but still process user input
        if not state.get("portfolio"):
            # Check if user wants to run optimization
            if "run" in last_user:
                # User wants to run optimization, skip intro message and proceed
                pass
            else:
                # Show intro message for other inputs
                state["messages"].append({
                    "role":"ai",
                    "content": (
                        "Here's the plan: I'll build an asset-class portfolio using mean-variance optimization.\n"
                        f"Defaults are **lambda = {lam}** and **cash_reserve = {cash_reserve:.2f}**.\n"
                        "Say \"set lambda to 1\", \"set cash to 0.05\", or just \"run\" to optimize now."
                    )
                })
                return state

        # Only route back to entry agent for explicit "proceed" commands
        if state.get("portfolio") and state.get("messages") and state["messages"][-1].get("role") == "user":
            last_user = state["messages"][-1].get("content", "").lower()
            if last_user.strip() in ["proceed", "next", "continue", "go ahead", "move on"]:
                state["messages"].append({"role": "ai", "content": "Great — proceeding to the next step."})
                state["awaiting_input"] = False
                state["intent_to_portfolio"] = False
                state["intent_to_investment"] = True  # Route to investment agent
                state["done"] = True   
                return state

        # Ask LLM to propose tool calls (single source of truth)
        # Pass current parameters to the portfolio manager
        state_with_params = state.copy()
        state_with_params["current_lambda"] = lam
        state_with_params["current_cash_reserve"] = cash_reserve
        tool_calls = self.portfolio_manager._plan_tools_with_llm(state_with_params)
        if not tool_calls:
            state["messages"].append({"role":"ai","content":"Tell me \"run\" to execute the optimizer with current settings, or \"set lambda to X / set cash to Y\"."})
            return state

        # Execute tool calls
        executed_optimizer = False

        for call in tool_calls:
            name = call.get("tool")
            args = call.get("args", {})

            if name == "set_portfolio_param":
                res = self.portfolio_manager.execute_tool_call(call)
                if res and res.get("ok"):
                    param = res["param"]
                    if param == "lambda":
                        lam = res["new_value"]
                    elif param == "cash_reserve":
                        cash_reserve = res["new_value"]
                    state["messages"].append({"role":"ai","content": f"{res['note']} (now {param} = {res['new_value']})"})
                else:
                    state["messages"].append({"role":"ai","content": f"Could not update parameter: {res.get('note','invalid input')}"})
                continue

            if name == "mean_variance_optimizer":
                # Clamp cash for optimizer constraints
                min_cash, max_cash = get_cash_reserve_constraints()
                clamped = min(max_cash, max(min_cash, cash_reserve))

                call_args = {
                    "risk_equity": float(risk.get("equity", 0.0)),
                    "risk_bonds": float(risk.get("bond", 0.0)),
                    "lam": lam,
                    "cash_reserve": clamped,
                }
                res = self.portfolio_manager.execute_tool_call({"tool":"mean_variance_optimizer","args":call_args})
                if isinstance(res, dict) and res:
                    state["portfolio"] = res
                    note = "" if clamped == cash_reserve else f" (cash_reserve {cash_reserve:.2f} was clamped to {clamped:.2f})"
                    table = self._format_portfolio(res)
                    state["messages"].append({"role":"ai","content": f"Optimization complete{note}. I've built your asset-class portfolio.\n\n{table}\n\n Review weights or proceed to ETF selection?"})
                    executed_optimizer = True
                else:
                    state["messages"].append({"role":"ai","content":"Optimization didn't return a portfolio. Try lambda=1, cash=0.05 and say \"run\"."})
                continue

            # Unknown tool name
            state["messages"].append({"role":"ai","content": f"(Skipping unknown tool: {name})"})

        return state

