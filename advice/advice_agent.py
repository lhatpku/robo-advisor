# advice/advice_agent.py
from __future__ import annotations
from typing import TypedDict, List, Dict, Any, Optional, Tuple
import re
from dataclasses import asdict

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from .questions import MCQuestion, MCAnswer, QUESTIONS
from .general_investing import general_investing_advice_tool

AGENT_SYSTEM_PROMPT = """\
You are the Advice Agent.

Your responsibilities:
1) Ask exactly one question at a time from the provided 7-question set (text and options are given to you verbatim).
2) Present options as numbered choices (1..N) exactly as provided.
3) Accept natural replies like “2”, “the second one”, or a phrase matching an option.
4) If the user asks “why”, briefly show the provided guidance for that question, then re-show the same question.
5) After ALL 7 answers are collected, you MUST call the `general_investing_advice` tool with the structured answers
   (qid -> {selected_index, selected_label, raw_user_text}). Then summarize the returned allocation clearly.

Style: concise, warm, and professional.
Never reword question text, options, or guidance.
"""

class AgentState(TypedDict):
    messages: List[Dict[str, Any]]
    q_idx: int
    answers: Dict[str, Dict[str, Any]]   # qid -> MCAnswer as dict
    done: bool
    recommendation: Optional[Dict[str, float]]
    awaiting_input: bool                 # prevents recursion while waiting
    intent_to_advise: bool

_ORDINALS = {
    "first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3,
    "fourth": 4, "4th": 4, "fifth": 5, "5th": 5, "sixth": 6, "6th": 6,
    "seventh": 7, "7th": 7, "eighth": 8, "8th": 8, "ninth": 9, "9th": 9,
    "tenth": 10, "10th": 10,
}

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip())

def _wants_guidance(text: str) -> bool:
    t = _norm(text)
    return any(k in t for k in ["why", "explain", "not sure", "help", "what do you mean", "guidance"])

def _render_question(q: MCQuestion) -> str:
    lines = [q.text, ""]
    for i, opt in enumerate(q.options, start=1):
        lines.append(f"{i}) {opt}")
    lines += ["", "Reply with the option number (e.g., '2'), or say 'I pick the second one'. If unsure, say 'why?'."]
    return "\n".join(lines)

def _parse_choice(user_text: str, q: MCQuestion) -> Optional[Tuple[int, str]]:
    text = _norm(user_text)
    # numeric
    m = re.search(r"\b(\d{1,2})\b", text)
    if m:
        k = int(m.group(1))
        if 1 <= k <= len(q.options):
            return k - 1, q.options[k - 1]
    # ordinal
    for w, n in _ORDINALS.items():
        if re.search(rf"\b{re.escape(w)}\b", text) and 1 <= n <= len(q.options):
            return n - 1, q.options[n - 1]
    # simple fuzzy token overlap
    matches = []
    for i, opt in enumerate(q.options):
        key = _norm(opt)
        toks = [t for t in key.split() if len(t) > 2]
        hits = sum(1 for t in toks if t in text)
        if hits >= max(1, len(toks) // 2):
            matches.append((i, opt))
    if len(matches) == 1:
        return matches[0]
    return None

def _ask_with_llm(llm: ChatOpenAI, q: MCQuestion) -> str:
    system = SystemMessage(content=AGENT_SYSTEM_PROMPT)
    dev = HumanMessage(content=f"Ask the following question EXACTLY as written, with numbered options:\n\n{_render_question(q)}")
    resp = llm.invoke([system, dev])
    return resp.content if isinstance(resp, AIMessage) else str(resp)

def _explain_and_ask_with_llm(llm: ChatOpenAI, q: MCQuestion) -> str:
    system = SystemMessage(content=AGENT_SYSTEM_PROMPT)
    dev = HumanMessage(content=(
        "The user asked why. Briefly explain using this guidance (verbatim as needed), "
        "then re-show the question with numbered options:\n\n"
        f"Guidance:\n{q.guidance}\n\nQuestion and options:\n{_render_question(q)}"
    ))
    resp = llm.invoke([system, dev])
    return resp.content if isinstance(resp, AIMessage) else str(resp)

def _retry_with_llm(llm: ChatOpenAI, q: MCQuestion) -> str:
    system = SystemMessage(content=AGENT_SYSTEM_PROMPT)
    dev = HumanMessage(content=f"User input did not clearly map to an option. Apologize briefly and re-show:\n\n{_render_question(q)}")
    resp = llm.invoke([system, dev])
    return resp.content if isinstance(resp, AIMessage) else str(resp)

def _finalize_with_tool_and_llm(state: AgentState, llm: ChatOpenAI) -> AgentState:
    llm_with_tools = llm.bind_tools([general_investing_advice_tool])

    system = SystemMessage(content=AGENT_SYSTEM_PROMPT)
    dev = HumanMessage(content=(
        "We have collected all 7 answers. Now you MUST call the tool `general_investing_advice`.\n"
        f"Answers payload:\n{state['answers']}\n\nCall the tool with the full answers payload."
    ))

    ai = llm_with_tools.invoke([system, dev])  # expect tool call
    tool_results: List[ToolMessage] = []

    # If the model emits a tool call, ensure payload has `answers`; else call directly as fallback.
    called_tool = False
    if hasattr(ai, "tool_calls") and ai.tool_calls:
        for tc in ai.tool_calls:
            if tc["name"] == "general_investing_advice":
                args = tc.get("args") or {}
                if not isinstance(args, dict):
                    args = {}
                if "answers" not in args or not args["answers"]:
                    args["answers"] = state["answers"]

                result = general_investing_advice_tool.invoke(args)
                state["recommendation"] = result
                tool_results.append(ToolMessage(name=tc["name"], content=str(result), tool_call_id=tc["id"]))
                called_tool = True

    # Fallback: model didn’t emit a tool call — call the tool directly
    if not called_tool:
        result = general_investing_advice_tool.invoke({"answers": state["answers"]})
        state["recommendation"] = result
        # No tool_call_id in this path
        tool_results.append(ToolMessage(name="general_investing_advice", content=str(result)))

    final_msgs = [system, dev, ai] + tool_results + [
        HumanMessage(content=(
            "Using the tool result above, produce a concise, friendly summary that:\n"
            "1) Recaps the 7 selections (qid order), and\n"
            "2) Presents the equity/bond allocation clearly as percentages with one decimal place.\n"
            "Do not ask new questions."
        ))
    ]
    final = llm_with_tools.invoke(final_msgs)
    final_text = final.content if isinstance(final, AIMessage) else str(final)
    state["messages"].append({"role": "ai", "content": final_text})
    return state

def advice_agent_step(state: AgentState, llm: ChatOpenAI) -> AgentState:
    # finished already?
    if state.get("done"):
        return state

    # all questions answered -> finalize once
    if state["q_idx"] >= len(QUESTIONS):
        # 💡 Sanity guard: if answers are incomplete, resume asking instead of finalizing
        if len(state["answers"]) < len(QUESTIONS):
            state["done"] = False
            state["awaiting_input"] = False
            # jump to the first unanswered question
            state["q_idx"] = len(state["answers"])
            return state

        if not state.get("awaiting_input", False):
            state["done"] = True
            state["awaiting_input"] = False
            return _finalize_with_tool_and_llm(state, llm)
        return state  # no-op

    q = QUESTIONS[state["q_idx"]]
    user_msgs = [m for m in state["messages"] if m["role"] == "user"]

    # If we haven't asked this question yet (awaiting_input is False), ask it now.
    if not state.get("awaiting_input", False):
        msg = _ask_with_llm(llm, q)
        state["messages"].append({"role": "ai", "content": msg})
        state["awaiting_input"] = True
        return state

    # We asked already and are waiting; if no new user message, do nothing (prevents recursion)
    if not user_msgs:
        return state

    last_user = user_msgs[-1]["content"]

    if _wants_guidance(last_user):
        msg = _explain_and_ask_with_llm(llm, q)
        state["messages"].append({"role": "ai", "content": msg})
        state["awaiting_input"] = True
        return state

    parsed = _parse_choice(last_user, q)
    if not parsed:
        msg = _retry_with_llm(llm, q)
        state["messages"].append({"role": "ai", "content": msg})
        state["awaiting_input"] = True
        return state

    # record and advance
    idx, label = parsed
    mc = MCAnswer(selected_index=idx, selected_label=label, raw_user_text=last_user)
    state["answers"][q.id] = asdict(mc)
    state["q_idx"] += 1
    state["awaiting_input"] = False

    if state["q_idx"] >= len(QUESTIONS):
        return state

    nxt = QUESTIONS[state["q_idx"]]
    msg = _ask_with_llm(llm, nxt)
    state["messages"].append({"role": "ai", "content": msg})
    state["awaiting_input"] = True
    return state
