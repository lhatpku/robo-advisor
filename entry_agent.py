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
        "start_journey",  # User wants to begin the journey
        "proceed_portfolio",
        "proceed_investment",
        "proceed_trading",
        "smalltalk",
        "unknown",
    ] = "unknown"
    equity: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    reply: str = ""


class EntryAgent:
    """
    LLM-powered entry agent that manages the conversation flow and sets intent flags.
    Only agent that sets intent_to_* flags based on user responses to proceed prompts.
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
            "You are a robo-advisor routing agent. Classify user intent for starting the journey.\n\n"
            "ACTIONS:\n"
            "• greet: New session or unclear intent\n"
            "• start_journey: User wants to begin the robo-advisor process\n"
            "• proceed_portfolio: User ready to build portfolio (requires existing risk assessment)\n"
            "• proceed_investment: User ready for fund selection (requires existing portfolio)\n"
            "• proceed_trading: User ready for trading execution (requires existing investment)\n"
            "• smalltalk/unknown: General chat or unclear\n\n"
            "DECISION RULES:\n"
            "• 'proceed', 'next', 'start', 'begin', 'go' → start_journey ONLY if no risk exists yet\n"
            "• If risk exists and user says 'proceed' → proceed_portfolio\n"
            "• If portfolio exists and user says 'proceed' → proceed_investment\n"
            "• If investment exists and user says 'proceed' → proceed_trading\n"
            "• Need risk assessment before proceed_portfolio\n"
            "• Need portfolio before proceed_investment\n"
            "• Need investment before proceed_trading\n"
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
            # User wants to start journey
            {
                "case": "user says 'proceed'",
                "have_risk": False,
                "last_ai": "I am a robo advisor, how can I assist you. Let me know if you want to start the journey, the first step is to define your risk profile.",
                "last_user": "proceed",
                "expect": {"action": "start_journey", "equity": None}
            },
            {
                "case": "user says 'next'",
                "have_risk": False,
                "last_ai": "I am a robo advisor, how can I assist you. Let me know if you want to start the journey, the first step is to define your risk profile.",
                "last_user": "next",
                "expect": {"action": "start_journey", "equity": None}
            },
            {
                "case": "user says 'start'",
                "have_risk": False,
                "last_ai": "I am a robo advisor, how can I assist you. Let me know if you want to start the journey, the first step is to define your risk profile.",
                "last_user": "start",
                "expect": {"action": "start_journey", "equity": None}
            },
            {
                "case": "user says 'begin'",
                "have_risk": False,
                "last_ai": "I am a robo advisor, how can I assist you. Let me know if you want to start the journey, the first step is to define your risk profile.",
                "last_user": "begin",
                "expect": {"action": "start_journey", "equity": None}
            },
            # Proceed to portfolio when risk exists
            {
                "case": "proceed to portfolio",
                "have_risk": True,
                "last_ai": "Perfect! I've set your allocation to **60% equity / 40% bonds**.",
                "last_user": "proceed",
                "expect": {"action": "proceed_portfolio", "equity": None}
            },
            {
                "case": "proceed to portfolio after risk",
                "have_risk": True,
                "last_ai": "Risk assessment complete. Next steps?",
                "last_user": "proceed",
                "expect": {"action": "proceed_portfolio", "equity": None}
            },
            # Proceed to investment when portfolio exists
            {
                "case": "proceed to investment",
                "have_risk": True,
                "have_portfolio": True,
                "last_ai": "Portfolio ready. Next steps?",
                "last_user": "proceed",
                "expect": {"action": "proceed_investment", "equity": None}
            },
            # Proceed to trading when investment exists
            {
                "case": "proceed to trading",
                "have_risk": True,
                "have_portfolio": True,
                "have_investment": True,
                "last_ai": "Investment ready. Next steps?",
                "last_user": "proceed",
                "expect": {"action": "proceed_trading", "equity": None}
            },
            # Unknown/smalltalk
            {
                "case": "smalltalk",
                "have_risk": False,
                "last_ai": "I am a robo advisor, how can I assist you. Let me know if you want to start the journey, the first step is to define your risk profile.",
                "last_user": "hello",
                "expect": {"action": "smalltalk", "equity": None}
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
                         ("I am a robo advisor, how can I assist you. Let me know if you want to start the journey, the first step is to define your risk profile."))
                state["messages"].append({"role":"ai","content": reply})
                state["entry_greeted"] = True
                return state
            # If the LLM didn't produce a greeting, set a concise default once
            state["messages"].append({
                "role":"ai",
                "content":"I am a robo advisor, how can I assist you. Let me know if you want to start the journey, the first step is to define your risk profile."
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

        # start_journey
        if action == "start_journey":
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


        # Handle phase ready prompt from reviewer (automatic or with user input)
        if state.get("ready_to_proceed"):
            # Process automatically if no user input, or if user says proceed
            if not last_user or any(word in last_user.lower() for word in ["yes", "proceed", "continue", "next", "go ahead", "ready"]):
                
                # Find the next phase that's ready to proceed
                ready_to_proceed = state.get("ready_to_proceed", {})
                next_phase = None
                for phase, ready in ready_to_proceed.items():
                    if ready:
                        # Check if this phase hasn't been completed yet
                        if phase == "risk" and not have_risk:
                            next_phase = phase
                            break
                        elif phase == "portfolio" and not have_portfolio:
                            next_phase = phase
                            break
                        elif phase == "investment" and not have_investment:
                            next_phase = phase
                            break
                        elif phase == "trading" and not state.get("trading_requests"):
                            next_phase = phase
                            break
                
                if next_phase:
                    # Set the appropriate intent flag
                    if next_phase == "risk":
                        state["intent_to_risk"] = True
                    elif next_phase == "portfolio":
                        state["intent_to_portfolio"] = True
                    elif next_phase == "investment":
                        state["intent_to_investment"] = True
                    elif next_phase == "trading":
                        state["intent_to_trading"] = True
                    
                    # Clear the ready_to_proceed flag
                    state["ready_to_proceed"] = None
                    return state
                else:
                    # All phases are complete
                    state["ready_to_proceed"] = None
                    return state
            else:
                # User provided input but not a proceed command
                # Clear the ready_to_proceed flag and process normally
                state["ready_to_proceed"] = None

        # Handle unknown actions
        if action in ["smalltalk", "unknown"]:
            say("I'm here to help you with your investment planning. Say 'proceed' to start the journey, or let me know how I can assist you.")
            return state

        # Default fallback
        say("I'm not sure what you'd like to do. You can say 'proceed' to start the investment planning journey.")
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

        return "__end__"


# Backward compatibility functions
def robo_entry_agent_step(state: AgentState, llm: ChatOpenAI) -> AgentState:
    """Backward compatibility wrapper for the old function-based interface."""
    agent = EntryAgent(llm)
    return agent.step(state)


def robo_entry_agent_router(state: AgentState, llm: ChatOpenAI) -> str:
    """Backward compatibility wrapper for the old function-based interface."""
    agent = EntryAgent(llm)
    return agent.router(state)
