# portfolio/portfolio_agent.py
from __future__ import annotations
from typing import Dict, Any
import os
from portfolio.portfolio_manager import PortfolioManager
from state import AgentState
from langchain_openai import ChatOpenAI

# Resolve the config path relative to THIS file
_THIS_DIR = os.path.dirname(__file__)
DEFAULT_MU_COV_PATH = os.path.join(_THIS_DIR, "config", "asset_stats.xlsx")


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

        # ensure defaults
        inv = state.setdefault("portfolio", {}) or {}
        state["portfolio"] = inv
        inv.setdefault("lambda", 1.0)
        inv.setdefault("cash_reserve", 0.05)
        inv.setdefault("mu_cov_xlsx_path", DEFAULT_MU_COV_PATH)

        # One-time intro on entry
        if not inv.get("__inv_intro_done__"):
            state["messages"].append({
                "role":"ai",
                "content": (
                    "Here's the plan: I'll build an asset-class portfolio using mean-variance optimization.\n"
                    f"Defaults are **lambda = {inv['lambda']}** and **cash_reserve = {inv['cash_reserve']:.2f}**.\n"
                    "Say \"set lambda to 1\", \"set cash to 0.05\", or just \"run\" to optimize now."
                )
            })
            inv["__inv_intro_done__"] = True
            return state

        # Only act on USER turns
        if not state.get("messages") or state["messages"][-1].get("role") != "user":
            return state

        # If a portfolio already exists, ask LLM if the user intends to PROCEED (no keyword list)
        if inv.get("portfolio") and state.get("messages") and state["messages"][-1].get("role") == "user":
            last_user = state["messages"][-1].get("content", "")
            if self.infer_proceed_intent(last_user):
                state["messages"].append({"role": "ai", "content": "Great — proceeding to the next step."})
                state["awaiting_input"] = False
                state["intent_to_portfolio"] = False
                state["intent_to_investment"] = True  # Route to investment agent
                state["done"] = True   
                return state

        # Ask LLM to propose tool calls (single source of truth)
        tool_calls = self.portfolio_manager._plan_tools_with_llm(state)
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
                    inv[param] = res["new_value"]
                    state["messages"].append({"role":"ai","content": f"{res['note']} (now {param} = {inv[param]})"})
                else:
                    state["messages"].append({"role":"ai","content": f"Could not update parameter: {res.get('note','invalid input')}"})
                continue

            if name == "mean_variance_optimizer":
                # Clamp cash for optimizer constraints
                cr = float(inv.get("cash_reserve", 0.05))
                clamped = min(0.05, max(0.00, cr))
                lam = float(inv.get("lambda", 1.0))

                call_args = {
                    "mu_cov_xlsx_path": inv.get("mu_cov_xlsx_path", DEFAULT_MU_COV_PATH),
                    "risk_equity": float(risk.get("equity", 0.0)),
                    "risk_bonds": float(risk.get("bond", 0.0)),
                    "lam": lam,
                    "cash_reserve": clamped,
                }
                res = self.portfolio_manager.execute_tool_call({"tool":"mean_variance_optimizer","args":call_args})
                if isinstance(res, dict) and res:
                    inv["portfolio"] = res
                    note = "" if clamped == cr else f" (cash_reserve {cr:.2f} was clamped to {clamped:.2f})"
                    table = self._format_portfolio(inv["portfolio"])
                    state["messages"].append({"role":"ai","content": f"Optimization complete{note}. I've built your asset-class portfolio.\n\n{table}\n\n Review weights or proceed to ETF selection?"})
                    executed_optimizer = True
                else:
                    state["messages"].append({"role":"ai","content":"Optimization didn't return a portfolio. Try lambda=1, cash=0.05 and say \"run\"."})
                continue

            # Unknown tool name
            state["messages"].append({"role":"ai","content": f"(Skipping unknown tool: {name})"})

        return state

