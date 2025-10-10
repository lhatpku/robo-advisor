# app.py
from __future__ import annotations
import os
import json
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from advice.advice_agent import AgentState, advice_agent_step

# ---------------------------
# LLM-powered entry agent (no keyword list, no helper)
# Decides if the user wants to start advice and replies once per user turn.
# ---------------------------
ENTRY_SYSTEM_PROMPT = """\
You are the Robo-Advisor Entry Agent.
Goals:
- Converse naturally with the investor about their goals and questions.
- Determine if the investor wants to START the investment advice onboarding now.
- If they want to start now, set wants_advice=true.
- Otherwise, respond helpfully and keep the conversation going, but do NOT start the onboarding.

STRICT OUTPUT: Return ONLY a compact JSON object with two fields:
{"wants_advice": true|false, "assistant_reply": "<your short reply>"}

Rules:
- Be brief and professional.
- Do NOT ask onboarding questions yourself; that is handled by the Advice Agent.
- Set wants_advice=true only if the user clearly indicates they want investment strategy advice/onboarding now.
"""

def robo_entry_agent_step(state: AgentState, llm: ChatOpenAI) -> AgentState:
    """
    Entry agent with confirmation:
    - If we already set intent_to_advise=True, do nothing (router will hand off).
    - Otherwise:
        a) If last was AI -> no-op (prevents loops).
        b) If last was USER and the previous AI message contained our confirmation marker,
           interpret yes/no. On yes -> set intent_to_advise=True; on no -> reply and keep chatting.
        c) Else (normal turn), ask LLM if the user wants advice now. If yes, send heads-up + ask to proceed
           (but DO NOT set intent_to_advise yet). If no, send a brief helpful reply.
    """
    if state.get("intent_to_advise"):
        return state

    if not state["messages"]:
        return state

    # If last turn was AI, we've already responded this tick -> do nothing.
    if state["messages"][-1].get("role") != "user":
        return state

    last_user = state["messages"][-1]["content"]

    # ---- (B) Handle confirmation replies (no new state; detect via marker in prior AI msg) ----
    CONFIRM_MARKER = "[ADVICE_CONFIRM]"
    prev_ai_msgs = [m for m in state["messages"][:-1] if m.get("role") == "ai"]
    if prev_ai_msgs and CONFIRM_MARKER in prev_ai_msgs[-1].get("content", ""):
        t = last_user.strip().lower()
        affirmative = any(x in t for x in ["yes", "y", "sure", "ok", "okay", "proceed", "start", "let's do it", "go ahead"])
        negative   = any(x in t for x in ["no", "n", "not now", "later", "stop", "cancel"])

        if affirmative:
            # Acknowledge and hand off next tick via router
            state["messages"].append({
                "role": "ai",
                "content": "Great — we’ll go through a short 7-question risk profile so I can tailor your allocation."
            })
            state["intent_to_advise"] = True
            return state

        if negative:
            # Stay in entry agent; keep chatting
            state["messages"].append({
                "role": "ai",
                "content": "No problem. We can continue discussing your goals or any questions you have."
            })
            return state

        # If unclear, re-prompt succinctly (keep the marker so we know we’re still confirming)
        state["messages"].append({
            "role": "ai",
            "content": f"{CONFIRM_MARKER} Would you like to proceed with the 7-question risk profile now? (yes/no)"
        })
        return state

    # ---- (C) Normal turn: classify intent with LLM (no JSON mode required) ----
    ENTRY_SYSTEM_PROMPT = (
        "You are the Robo-Advisor Entry Agent.\n"
        "Decide if the user wants to START the investment advice onboarding now.\n"
        'Return ONLY JSON: {"wants_advice": true|false, "assistant_reply": "<short helpful reply>"}\n'
        "Be brief and professional. Do NOT ask the onboarding questions yourself."
    )

    import json
    msgs = [
        {"role": "system", "content": ENTRY_SYSTEM_PROMPT},
        {"role": "user", "content": f"Latest user message:\n{last_user}\n\nReturn the strict JSON only."}
    ]
    resp = llm.invoke(msgs)
    content = getattr(resp, "content", "") if resp else ""

    try:
        data = json.loads(content)
    except Exception:
        data = {"wants_advice": False, "assistant_reply": content.strip()[:500]}

    wants = bool(data.get("wants_advice"))
    reply = (data.get("assistant_reply") or "").strip()

    if wants:
        # ---- (A) Heads-up + proceed?  (do NOT set intent yet)
        heads_up = (
            "To give you meaningful investment advice, we’ll complete a brief questionnaire "
            "to assess your risk profile (7 quick questions)."
        )
        state["messages"].append({
            "role": "ai",
            "content": f"{heads_up}\n\n{CONFIRM_MARKER} Would you like to proceed now? (yes/no)"
        })
        return state

    # Not starting: reply once (since last was user) and wait for the next user input.
    if reply:
        state["messages"].append({"role": "ai", "content": reply})
    return state


# ---------------------------
# Build graph
# ---------------------------
def build_graph(llm: ChatOpenAI):
    builder = StateGraph(AgentState)

    builder.add_node("robo_entry", lambda s: robo_entry_agent_step(s, llm))
    builder.add_node("advice_agent", lambda s: advice_agent_step(s, llm))
    builder.set_entry_point("robo_entry")

    # Route to advice when intent_to_advise is True; otherwise stop this tick.
    def router_from_entry(state: AgentState) -> str:
        return "advice_agent" if state.get("intent_to_advise") else END

    builder.add_conditional_edges("robo_entry", router_from_entry, {
        "advice_agent": "advice_agent",
        END: END
    })

    # Advice agent loops until done; then END.
    builder.add_edge("advice_agent", END)

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
        "recommendation": None,
        "awaiting_input": False,   # used by advice_agent
        "intent_to_advise": False  # set by entry agent
    }

    # Console loop: you type, the graph runs one tick, prints last AI reply if any.
    while True:
        user_in = input("> ")
        state["messages"].append({"role": "user", "content": user_in})
        state = graph.invoke(state)
        ai_msgs = [m for m in state["messages"] if m["role"] == "ai"]
        if ai_msgs:
            print(ai_msgs[-1]["content"])
