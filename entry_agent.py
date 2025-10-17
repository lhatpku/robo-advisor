# entry_agent.py
from __future__ import annotations
import json
from typing import Optional, Literal
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from state import AgentState


class EntryIntent(BaseModel):
    action: Literal[
        "greet",          # NEW
        "set_equity",
        "ask_risk",
        "proceed_portfolio",
        "proceed_investment",
        "proceed_trading",
        "review",
        "reset_equity",
        "smalltalk",
        "unknown",
    ] = "unknown"
    equity: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    reply: str = ""


class EntryAgent:
    """
    LLM-powered entry agent that decides if the user wants to start risk assessment
    and replies once per user turn.
    """
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self._structured_llm = llm.with_structured_output(EntryIntent).bind(temperature=0.0)
    
    def _classify_entry_intent(
        self,
        have_risk: bool,
        last_user: str,
        last_ai: str | None = None,
        have_portfolio: bool = False,
        have_investment: bool = False,
    ) -> dict:
        system = (
            "You are a robo-advisor routing agent. Classify user intent and extract equity values.\n\n"
            "ACTIONS:\n"
            "• greet: New session or unclear intent\n"
            "• set_equity: User provided equity weight (extract number 0.05-0.95)\n"
            "• ask_risk: User wants risk questionnaire\n"
            "• proceed_portfolio: User ready to build portfolio (requires existing risk assessment)\n"
            "• proceed_investment: User ready for fund selection (requires existing portfolio)\n"
            "• proceed_trading: User ready for trading execution (requires existing investment)\n"
            "• review: User wants to set equity but provided no number\n"
            "• reset_equity: User wants to clear current equity choice\n"
            "• smalltalk/unknown: General chat or unclear\n\n"
            "EQUITY EXTRACTION:\n"
            "• Convert percentages to decimals: '60%' → 0.60, '60' → 0.60\n"
            "• Use first valid number found (0.05-0.95 range)\n"
            "• Return null if no valid number\n\n"
            "DECISION RULES:\n"
            "• 'set equity' without number → review (not ask_risk)\n"
            "• Only ask_risk for explicit guidance requests\n"
            "• Need risk assessment before proceed_portfolio\n"
            "• Need portfolio before proceed_investment\n"
            "• Need investment before proceed_trading\n"
            "• When portfolio exists and user says 'proceed' → proceed_investment\n"
            "• When investment exists and user says 'trade' → proceed_trading\n"
            "• Keep responses concise and helpful"
        )

        examples = [
            # Session start
            {
                "case": "session start greeting",
                "have_risk": False,
                "last_ai": "",
                "last_user": "",
                "expect": {"action": "greet", "equity": None}
            },
            # User wants to set equity but no number -> REVIEW (not ask_risk)
            {
                "case": "user says 'set equity level' (no number)",
                "have_risk": False,
                "last_ai": "How can I assist you today?",
                "last_user": "set equity level",
                "expect": {"action": "review", "equity": None}
            },
            {
                "case": "user says 'I can provide an equity weight' (no number)",
                "have_risk": False,
                "last_ai": "How can I assist you today?",
                "last_user": "I can provide an equity weight",
                "expect": {"action": "review", "equity": None}
            },
            {
                "case": "user says 'I know my preference' (no number)",
                "have_risk": False,
                "last_ai": "You can provide an equity weight or ask for guidance.",
                "last_user": "I know my preference",
                "expect": {"action": "review", "equity": None}
            },
            # Numeric set equity
            {
                "case": "explicit decimal",
                "have_risk": False,
                "last_ai": "Provide an equity weight or ask for guidance.",
                "last_user": "0.6",
                "expect": {"action": "set_equity", "equity": 0.60}
            },
            {
                "case": "explicit percent",
                "have_risk": False,
                "last_ai": "Provide an equity weight or ask for guidance.",
                "last_user": "60%",
                "expect": {"action": "set_equity", "equity": 0.60}
            },
            {
                "case": "integer meaning percent",
                "have_risk": False,
                "last_ai": "Provide an equity weight.",
                "last_user": "60",
                "expect": {"action": "set_equity", "equity": 0.60}
            },
            # Guidance only when explicitly requested
            {
                "case": "guidance confirmation",
                "have_risk": False,
                "last_ai": "Would you like guidance to determine your equity/bond split?",
                "last_user": "sure",
                "expect": {"action": "ask_risk", "equity": None}
            },
            {
                "case": "bare guidance", 
                "have_risk": True,
                "last_ai": "Current mix shown with options.", 
                "last_user": "guidance",
                "expect": {"action": "ask_risk", "equity": None}
            },
            {
                "case": "use guidance phrase", 
                "have_risk": True,
                "last_ai": "To continue, you can set equity, use guidance, reset equity, or proceed.",
                "last_user": "use guidance",
                "expect": {"action": "ask_risk", "equity": None}
            },
            # Proceed to portfolio after risk exists
            {
                "case": "proceed after risk exists",
                "have_risk": True,
                "last_ai": "Shall I proceed to build your portfolio now?",
                "last_user": "proceed",
                "expect": {"action": "proceed_portfolio", "equity": None}
            },
            # Proceed to investment after portfolio exists
            {
                "case": "proceed to investment after portfolio",
                "have_risk": True,
                "have_portfolio": True,
                "last_ai": "Do you want to proceed to get a tradeable portfolio with selected funds?",
                "last_user": "yes",
                "expect": {"action": "proceed_investment", "equity": None}
            },
            {
                "case": "user wants fund selection",
                "have_risk": True,
                "have_portfolio": True,
                "last_ai": "Portfolio complete. What would you like to do next?",
                "last_user": "select funds",
                "expect": {"action": "proceed_investment", "equity": None}
            },
            {
                "case": "proceed after portfolio exists",
                "have_risk": True,
                "have_portfolio": True,
                "last_ai": "Review weights or proceed to ETF selection?",
                "last_user": "proceed",
                "expect": {"action": "proceed_investment", "equity": None}
            },
            # Proceed to trading after investment exists
            {
                "case": "proceed to trading after investment",
                "have_risk": True,
                "have_portfolio": True, 
                "have_investment": True,  
                "last_ai": "Ready to generate trading requests?",
                "last_user": "yes",
                "expect": {"action": "proceed_trading", "equity": None}
            },
            {
                "case": "user wants trading",
                "have_risk": True,
                "have_portfolio": True, 
                "have_investment": True,  
                "last_ai": "Investment complete. What's next?",
                "last_user": "generate trades",
                "expect": {"action": "proceed_trading", "equity": None}
            },
            {
                "case": "trade command",
                "have_risk": True,
                "have_portfolio": True, 
                "have_investment": True,  
                "last_ai": "Investment Portfolio ready. Next steps?",
                "last_user": "trade",
                "expect": {"action": "proceed_trading", "equity": None}
            },
            {
                "case": "trade command",
                "have_risk": True,
                "have_portfolio": True, 
                "have_investment": True,  
                "last_ai": "Investment Portfolio ready. Next steps?",
                "last_user": "proceed",
                "expect": {"action": "proceed_trading", "equity": None}
            },
            # Reset equity
            {
                "case": "reset equity",
                "have_risk": True,
                "last_ai": "Would you like to edit your equity or proceed?",
                "last_user": "reset equity",
                "expect": {"action": "reset_equity", "equity": None}
            },
        ]

        user = (
            f"have_risk={str(have_risk).lower()}\n"
            f"have_portfolio={str(have_portfolio).lower()}\n"
            f"have_investment={str(have_investment).lower()}\n"
            f"previous_ai_message:\n{(last_ai or '').strip()}\n\n"
            f"latest_user_message:\n{last_user}\n\n"
            f"examples:\n{examples}\n\n"
            "Output ONLY the structured object."
        )

        if not (last_user or "").strip():
            return {"action": "greet", "equity": None, "reply": ""}

        try:
            intent = self._structured_llm.invoke(
                [{"role": "system", "content": system},
                 {"role": "user",   "content": user}]
            )
            
            # Normalize return into a dict across LC versions:
            if isinstance(intent, dict):
                data = intent
            elif hasattr(intent, "model_dump"):           # Pydantic v2
                data = intent.model_dump()
            elif hasattr(intent, "dict"):                 # Pydantic v1
                data = intent.dict()
            else:
                # Fallback: try to read JSON from a message-like object
                text = getattr(intent, "content", "") or str(intent)
                try:
                    import json
                    data = json.loads(text)
                except Exception:
                    data = {}

            # Defaults / normalization
            data = dict(data)  # ensure real dict
            data.setdefault("action", "unknown")
            data.setdefault("equity", None)
            data.setdefault("reply", "")
            
            return data

        except Exception:
            return {"action": "unknown", "equity": None, "reply": ""}

    def step(self, state: AgentState) -> AgentState:
        """
        Main entry point for the entry agent step.
        """
        # Determine context
        msgs = state.get("messages") or []
        last_is_user = bool(msgs) and msgs[-1].get("role") == "user"
        last_user = msgs[-1]["content"] if last_is_user else ""
        prev_ai = next((m.get("content","") for m in reversed(msgs[:-1]) if m.get("role")=="ai"), "")

        have_risk = bool(state.get("risk"))

        if state.get("intent_to_risk"):
            # risk flow owns the turn; do not speak or classify.
            return state
        
        if state.get("intent_to_investment") and not state.get("intent_to_trading"):
            # investment flow owns the turn; do not speak or classify.
            return state

        if state.get("intent_to_trading"):
            # trading flow owns the turn; do not speak or classify.
            return state

        # --- INIT / GREETING: allow LLM to greet when starting or after returning here ---
        if not msgs or (not last_is_user and not state.get("entry_greeted")):
            portfolio = state.get("portfolio") or {}
            have_portfolio = bool(portfolio.get("portfolio"))
            intent = self._classify_entry_intent(have_risk, last_user="", last_ai=prev_ai, have_portfolio=have_portfolio)
            if intent.get("action") == "greet" or intent.get("reply"):
                reply = (intent.get("reply") or
                         ("How can I assist you today?\n"
                          "• **Set equity** with a number (e.g., `0.60` or `60%`).\n"
                          "• **Use guidance** to set equity by typing `use guidance`."))
                state["messages"].append({"role":"ai","content": reply})
                state["entry_greeted"] = True
                return state
            # If the LLM didn't produce a greeting, set a concise default once
            state["messages"].append({
                "role":"ai",
                "content":"How can I assist you today? You can set an equity weight (0.05–0.95), or proceed to portfolioing if you already have provided already."
            })
            state["entry_greeted"] = True
            return state

        # --- Normal path: only act on USER turns (prevents loops) ---
        if not last_is_user:
            return state

        portfolio = state.get("portfolio") or {}
        have_portfolio = bool(portfolio.get("portfolio"))
        have_investment = bool(portfolio.get("investment"))
        intent = self._classify_entry_intent(have_risk, last_user, prev_ai, have_portfolio, have_investment)
        action = (intent.get("action") or "unknown").strip().lower()
        reply  = (intent.get("reply")  or "").strip()
        intent_equity = intent.get("equity", None)
        equity = intent_equity if action == "set_equity" else None

        def say(msg: str):
            if msg:
                state["messages"].append({"role": "ai", "content": msg})

        # set_equity
        if action == "set_equity" and (equity is not None):
            
            try:
                e = float(equity)
            except Exception:
                e = None
            if e is None or not (0.05 <= e <= 0.95):
                say("Please provide an equity allocation between **0.05 and 0.95** (e.g., 0.70 for 70%).")
                return state

            state["risk"] = {"equity": float(e), "bond": round(1.0 - float(e), 6)}
            e = float(state["risk"]["equity"])
            b = 1.0 - e

            say(
                f"Updated: **{e*100:.0f}% equity / {b*100:.0f}% bonds**.\n"
                "Would you like to **review/edit** this split or **proceed** to portfolio construction?"
            )
            return state

        # ask_risk
        if action == "ask_risk":
            state["intent_to_risk"] = True
            return state

        # proceed_portfolio
        if action == "proceed_portfolio" and have_risk:
            state["intent_to_portfolio"] = True
            return state

        # proceed_investment
        if action == "proceed_investment":
            portfolio = state.get("portfolio", {})
            if portfolio:
                state["intent_to_investment"] = True
                return state
            else:
                say("I need to build your portfolio first before selecting specific funds. Let me help you with that.")
                return state

        # proceed_trading
        if action == "proceed_trading":
            if have_investment:
                state["intent_to_trading"] = True
                return state
            else:
                say("I need to complete your investment selection first before generating trading requests. Let me help you with that.")
                return state

        # Handle completion after investment (user says "done", "ok", etc.)
        if have_investment and any(word in last_user for word in ["done", "ok", "okay", "good", "fine", "next", "proceed", "continue", "ready", "complete", "finished", "trade", "trading"]):
            say("Great! Your investment portfolio is complete. Would you like to proceed to generate trading requests to see how to execute this portfolio?")
            return state

        # reset_equity
        if action == "reset_equity":
            state["risk"] = None
            say(
                "Equity choice cleared. You can:\n"
                "• **Set equity** now with a number (e.g., `0.70` or `70%`).\n"
                "• Or **use guidance** to set equity by typing `use guidance`."
            )
            return state

        # review
        if action == "review" and have_risk:
            e = float(state["risk"]["equity"])
            say(
                f"Current mix: **{e*100:.0f}% equity / {(1-e)*100:.0f}% bonds**.\n"
                "To continue, you can:\n"
                "• **Use guidance** to reset the equity by answering the questionnaire by typing `use guidance`.\n"
                "• **Reset equity** by typing `reset equity as`.\n"
                "• Or **proceed** to portfolio construction."
            )
            return state

        # smalltalk/unknown nudges
        if not have_risk:
            say(
                reply or
                "If you already know your equity preference, reply with a number between **0.05 and 0.95** "
                "(e.g., 0.70). Otherwise, say you'd like guidance and I'll run a quick questionnaire."
            )
        else:
            e = float(state["risk"]["equity"])
            say(
                reply or
                f"You're at **{e*100:.0f}% equity / {(1-e)*100:.0f}% bonds**. "
                "Would you like to **review/edit** this, or **proceed** to portfolio construction?"
            )

        return state

    def router(self, state: AgentState) -> str:
        """
        Router function that determines the next step based on state.
        """
        msgs = state.get("messages") or []
        last_is_user = bool(msgs) and msgs[-1].get("role") == "user"

        # Proceed to portfolio only when risk exists, user chose to proceed, and it's a user turn
        if state.get("risk") and state.get("intent_to_portfolio") and last_is_user:
            # Clear the intent flags when routing to portfolio
            state["intent_to_risk"] = False
            return "portfolio_agent"

        # Proceed to investment when portfolio exists and user chose to proceed
        if state.get("portfolio") and state.get("intent_to_investment") and last_is_user:
            # Clear the intent flags when routing to investment
            state["intent_to_portfolio"] = False
            return "investment_agent"

        # Proceed to trading when investment exists and user chose to proceed
        if state.get("investment") and state.get("intent_to_trading") and last_is_user:
            # Clear the intent flags when routing to trading
            state["intent_to_investment"] = False
            state["intent_to_trading"] = False
            return "trading_agent"

        # Start risk only when the user asked for guidance and we don't already have risk
        if state.get("intent_to_risk") and last_is_user:
            # Clear the intent flag when routing to risk
            state["intent_to_risk"] = False
            return "risk_agent"

        return "END"


# Backward compatibility functions
def robo_entry_agent_step(state: AgentState, llm: ChatOpenAI) -> AgentState:
    """Backward compatibility wrapper for the old function-based interface."""
    agent = EntryAgent(llm)
    return agent.step(state)


def router_from_entry(state: AgentState) -> str:
    """Backward compatibility wrapper for the old function-based interface."""
    # This is a bit tricky since we need an LLM instance, but the router doesn't use it
    # We'll create a minimal agent just for the router
    from langchain_openai import ChatOpenAI
    agent = EntryAgent(ChatOpenAI())  # This won't be used in router
    return agent.router(state)