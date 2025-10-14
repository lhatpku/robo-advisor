# app.py
from __future__ import annotations
import os
import json
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from advice.advice_agent import advice_agent_step
from investment.investment_agent import investment_agent_step
from state import AgentState

from typing import Optional, Literal
from pydantic import BaseModel, Field

class EntryIntent(BaseModel):
    action: Literal[
        "greet",          # NEW
        "set_equity",
        "ask_advice",
        "proceed_invest",
        "review",
        "reset_equity",
        "smalltalk",
        "unknown",
    ] = "unknown"
    equity: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    reply: str = ""

# ---------------------------
# LLM-powered entry agent (no keyword list, no helper)
# Decides if the user wants to start advice and replies once per user turn.
# ---------------------------
def _classify_entry_intent(
    llm: ChatOpenAI,
    have_advice: bool,
    last_user: str,
    last_ai: str | None = None,
) -> dict:

    structured_llm = llm.with_structured_output(EntryIntent).bind(temperature=0.0)

    system = (
        "You are a routing planner for a robo-advisor entry node.\n"
        "Pick ONE action for the next step given the latest USER message and the previous AI message.\n\n"
        "Actions:\n"
        "- greet: session start or unclear intent; brief welcome + choices.\n"
        "- set_equity: USER explicitly provided an equity weight (e.g., '0.6', '60%', 'equity 65').\n"
        "- ask_advice: USER wants questionnaire guidance (or replies yes/sure/ok to an AI prompt offering guidance).\n"
        "- proceed_invest: USER is satisfied and wants to build the portfolio now (or replies yes/sure/ok/next/proceed to such a prompt) AND advice already exists.\n"
        "- review: USER wants to review/tweak or **set equity** but did NOT provide a number.\n"
        "- reset_equity: USER asks to reset/clear the equity choice (e.g., 'reset equity', 'start over').\n"
        "- smalltalk/unknown: chit-chat or unclear.\n\n"
        "Extraction rules for set_equity:\n"
        "• Convert integers/percentages to a 0..1 float (e.g., '60'/'60%' → 0.60). Accept decimals like '0.6' or '.6'.\n"
        "• If multiple numbers appear, use the first plausible equity weight; if none, equity=null (do not guess).\n\n"
        "Decision rules (important):\n"
        "• If the USER says they want to **set/update/change equity** but gives **no number** (e.g., 'set equity', 'set equity level', 'I can provide an equity weight', 'I know my preference'), choose **review** (NOT ask_advice).\n"
        "• Only choose ask_advice when the USER explicitly asks for guidance (or confirms an AI offer of guidance).\n"
        "• When advice is missing, prefer set_equity or review (if no number) over proceed_invest.\n"
        "• Keep the reply short and helpful."
    )

    examples = [
        # Session start
        {
            "case": "session start greeting",
            "have_advice": False,
            "last_ai": "",
            "last_user": "",
            "expect": {"action": "greet", "equity": None}
        },
        # User wants to set equity but no number -> REVIEW (not ask_advice)
        {
            "case": "user says 'set equity level' (no number)",
            "have_advice": False,
            "last_ai": "How can I assist you today?",
            "last_user": "set equity level",
            "expect": {"action": "review", "equity": None}
        },
        {
            "case": "user says 'I can provide an equity weight' (no number)",
            "have_advice": False,
            "last_ai": "How can I assist you today?",
            "last_user": "I can provide an equity weight",
            "expect": {"action": "review", "equity": None}
        },
        {
            "case": "user says 'I know my preference' (no number)",
            "have_advice": False,
            "last_ai": "You can provide an equity weight or ask for guidance.",
            "last_user": "I know my preference",
            "expect": {"action": "review", "equity": None}
        },
        # Numeric set equity
        {
            "case": "explicit decimal",
            "have_advice": False,
            "last_ai": "Provide an equity weight or ask for guidance.",
            "last_user": "0.6",
            "expect": {"action": "set_equity", "equity": 0.60}
        },
        {
            "case": "explicit percent",
            "have_advice": False,
            "last_ai": "Provide an equity weight or ask for guidance.",
            "last_user": "60%",
            "expect": {"action": "set_equity", "equity": 0.60}
        },
        {
            "case": "integer meaning percent",
            "have_advice": False,
            "last_ai": "Provide an equity weight.",
            "last_user": "60",
            "expect": {"action": "set_equity", "equity": 0.60}
        },
        # Guidance only when explicitly requested
        {
            "case": "guidance confirmation",
            "have_advice": False,
            "last_ai": "Would you like guidance to determine your equity/bond split?",
            "last_user": "sure",
            "expect": {"action": "ask_advice", "equity": None}
        },

        {"case":"bare guidance", "have_advice":True,
        "last_ai":"Current mix shown with options.", "last_user":"guidance",
        "expect":{"action":"ask_advice","equity":None}},

        {"case":"use guidance phrase", "have_advice":True,
        "last_ai":"To continue, you can set equity, use guidance, reset equity, or proceed.",
        "last_user":"use guidance",
        "expect":{"action":"ask_advice","equity":None}},

        # Proceed to invest after advice exists
        {
            "case": "proceed after advice exists",
            "have_advice": True,
            "last_ai": "Shall I proceed to build your portfolio now?",
            "last_user": "proceed",
            "expect": {"action": "proceed_invest", "equity": None}
        },
        # Reset equity
        {
            "case": "reset equity",
            "have_advice": True,
            "last_ai": "Would you like to edit your equity or proceed?",
            "last_user": "reset equity",
            "expect": {"action": "reset_equity", "equity": None}
        },
    ]

    user = (
        f"have_advice={str(have_advice).lower()}\n"
        f"previous_ai_message:\n{(last_ai or '').strip()}\n\n"
        f"latest_user_message:\n{last_user}\n\n"
        f"examples:\n{examples}\n\n"
        "Output ONLY the structured object."
    )

    if not (last_user or "").strip():
        return {"action": "greet", "equity": None, "reply": ""}

    try:
        intent = structured_llm.invoke(
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


def robo_entry_agent_step(state: AgentState, llm: ChatOpenAI) -> AgentState:
    # Determine context
    msgs = state.get("messages") or []
    last_is_user = bool(msgs) and msgs[-1].get("role") == "user"
    last_user = msgs[-1]["content"] if last_is_user else ""
    prev_ai = next((m.get("content","") for m in reversed(msgs[:-1]) if m.get("role")=="ai"), "")

    have_advice = bool(state.get("advice"))

    if state.get("intent_to_advise"):
        # Advice flow owns the turn; do not speak or classify.
        return state

    # --- INIT / GREETING: allow LLM to greet when starting or after returning here ---
    if not msgs or (not last_is_user and not state.get("entry_greeted")):
        intent = _classify_entry_intent(llm, have_advice, last_user="", last_ai=prev_ai)
        if intent.get("action") == "greet" or intent.get("reply"):
            reply = (intent.get("reply") or
                     ("How can I assist you today?\n"
                      "• Set the equity level or\n"
                      "• If you already have equity level set already, say you’re ready to proceed to portfolio construction"))
            state["messages"].append({"role":"ai","content": reply})
            state["entry_greeted"] = True
            return state
        # If the LLM didn’t produce a greeting, set a concise default once
        state["messages"].append({
            "role":"ai",
            "content":"How can I assist you today? You can set an equity weight (0.05–0.95), or proceed to investing if you already have provided already."
        })
        state["entry_greeted"] = True
        return state

    # --- Normal path: only act on USER turns (prevents loops) ---
    if not last_is_user:
        return state

    intent = _classify_entry_intent(llm, have_advice, last_user, prev_ai)
    action = (intent.get("action") or "unknown").strip().lower()
    reply  = (intent.get("reply")  or "").strip()
    intent_equity = intent.get("equity", None)
    equity = intent_equity if action == "set_equity" else None

    def say(msg: str):
        if msg:
            state["messages"].append({"role": "ai", "content": msg})

    # (rest of your existing logic stays the same)
    # 1) set_equity
    if action == "set_equity" and (equity is not None):
        
        try:
            e = float(equity)
        except Exception:
            e = None
        if e is None or not (0.05 <= e <= 0.95):
            say("Please provide an equity allocation between **0.05 and 0.95** (e.g., 0.70 for 70%).")
            return state

        state["advice"] = {"equity": float(e), "bond": round(1.0 - float(e), 6)}
        e = float(state["advice"]["equity"])
        b = 1.0 - e

        say(
            f"Updated: **{e*100:.0f}% equity / {b*100:.0f}% bonds**.\n"
            "Would you like to **review/edit** this split or **proceed** to investment construction?"
        )
        return state

    # 2) ask_advice
    if action == "ask_advice":
        state["intent_to_advise"] = True
        return state

    # 3) proceed_invest
    if action == "proceed_invest" and have_advice:
        state["intent_to_investment"] = True
        return state

    # 3.5) reset_equity
    if action == "reset_equity":
        state["advice"] = None
        state["messages"].append({"role": "ai", "content": (
            "Equity choice cleared. You can:\n"
            "• **Set equity** now with a number (e.g., `0.70` or `70%`).\n"
            "• Or **use guidance** by typing `use guidance`."
        )})
        return state

    # 4) review
    if action == "review" and have_advice:
        e = float(state["advice"]["equity"])
        say(
            f"Current mix: **{e*100:.0f}% equity / {(1-e)*100:.0f}% bonds**.\n"
            "To continue, you can:\n"
            "• **Set equity** with a number (e.g., `0.60` or `60%`).\n"
            "• **Use guidance** by typing `use guidance`.\n"
            "• **Reset equity** by typing `reset equity`.\n"
            "• Or **proceed** to portfolio construction."
        )
        return state

    # 5) smalltalk/unknown nudges
    if not have_advice:
        say(
            reply or
            "If you already know your equity preference, reply with a number between **0.05 and 0.95** "
            "(e.g., 0.70). Otherwise, say you’d like guidance and I’ll run a quick questionnaire."
        )
    else:
        e = float(state["advice"]["equity"])
        say(
            reply or
            f"You’re at **{e*100:.0f}% equity / {(1-e)*100:.0f}% bonds**. "
            "Would you like to **review/edit** this, or **proceed** to portfolio construction?"
        )

    return state

# ---------------------------
# Router (only transition on USER turns)
# ---------------------------
def router_from_entry(state: AgentState) -> str:
    
    msgs = state.get("messages") or []
    last_is_user = bool(msgs) and msgs[-1].get("role") == "user"

    # Proceed to investment only when advice exists, user chose to proceed, and it's a user turn
    if state.get("advice") and state.get("intent_to_investment") and last_is_user:
        return "investment_agent"

    # Start advice only when the user asked for guidance and we don't already have advice
    if state.get("intent_to_advise") and last_is_user:
        return "advice_agent"

    return END


def build_graph(llm: ChatOpenAI):
    builder = StateGraph(AgentState)

    builder.add_node("robo_entry", lambda s: robo_entry_agent_step(s, llm))
    builder.add_node("advice_agent", lambda s: advice_agent_step(s, llm))
    builder.add_node("investment_agent", lambda s: investment_agent_step(s, llm))

    builder.set_entry_point("robo_entry")

    # Route to advice or investment.
    builder.add_conditional_edges("robo_entry", router_from_entry, {
        "advice_agent": "advice_agent",
        "investment_agent": "investment_agent",
        END: END
    })

    # Advice agent loops until done; then END.
    builder.add_edge("advice_agent", "robo_entry")
    builder.add_edge("investment_agent", "robo_entry")

    # Keep it simple: no checkpointer required.
    return builder.compile()

# ---------------------------
# Run (simple REPL)
# ---------------------------
if __name__ == "__main__":
    load_dotenv()  # expects OPENAI_API_KEY; optional OPENAI_MODEL / OPENAI_TEMPERATURE

    # Force JSON output from entry agent to avoid parsing issues.
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.2")),
    )

    graph = build_graph(llm)

    # Initial state (note: no extra fields beyond what's already defined in AgentState)
    state: AgentState = {
        "messages": [],
        "q_idx": 0,
        "answers": {},
        "done": False,
        "advice": None,
        "awaiting_input": False,   # used by advice_agent
        "intent_to_advise": False,  # set by entry agent
        "intent_to_investment": False,
        "entry_greeted": False,
        "investment": None
    }

    # --- INITIAL TICK to produce greeting ---
    state = graph.invoke(state)
    ai_msgs = [m for m in state["messages"] if m.get("role") == "ai"]
    if ai_msgs:
        print(ai_msgs[-1]["content"])

    # --- normal REPL ---
    while True:
        user_in = input("> ")
        state["messages"].append({"role": "user", "content": user_in})
        state = graph.invoke(state)
        ai_msgs = [m for m in state["messages"] if m.get("role") == "ai"]
        if ai_msgs:
            print(ai_msgs[-1]["content"])