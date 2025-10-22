# risk/risk_agent.py
from __future__ import annotations
from typing import TypedDict, List, Dict, Any, Optional, Tuple
import re
from dataclasses import asdict

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from .risk_manager import RiskManager, MCQuestion, MCAnswer
from state import AgentState


class RiskAgent:
    """
    Risk assessment agent that conducts questionnaire-based risk profiling
    and calculates appropriate asset allocation.
    """
    
    AGENT_SYSTEM_PROMPT = """\
        You are a risk assessment agent conducting a 7-question financial questionnaire.

        ROLE: Guide users through risk profiling to determine optimal asset allocation.

        PROCESS:
        1. Ask ONE question at a time (exactly as written)
        2. Present numbered options (1-N) verbatim
        3. Parse natural responses: "2", "second option", or phrase matches
        4. If user asks "why", explain guidance briefly, then re-ask
        5. After all 7 answers, calculate allocation and summarize results

        STYLE: Professional, warm, concise. Never modify question text or options.
        """

    _ORDINALS = {
        "first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3,
        "fourth": 4, "4th": 4, "fifth": 5, "5th": 5, "sixth": 6, "6th": 6,
        "seventh": 7, "7th": 7, "eighth": 8, "8th": 8, "ninth": 9, "9th": 9,
        "tenth": 10, "10th": 10,
    }
    
    def __init__(self, llm: ChatOpenAI):
        """
        Initialize the RiskAgent.
        
        Args:
            llm: ChatOpenAI instance for generating responses
        """
        self.llm = llm
        self.risk_manager = RiskManager()
    
    def _norm(self, s: str) -> str:
        """Normalize string for comparison."""
        return re.sub(r"\s+", " ", s.lower().strip())
    
    def _wants_guidance(self, text: str) -> bool:
        """Check if user is asking for guidance/explanation."""
        t = self._norm(text)
        return any(k in t for k in ["why", "explain", "not sure", "help", "what do you mean", "guidance"])
    
    def _render_question(self, q: MCQuestion) -> str:
        """Render question with numbered options."""
        lines = [q.text, ""]
        for i, opt in enumerate(q.options, start=1):
            lines.append(f"{i}) {opt}")
        lines += ["", "Reply with the option number (e.g., '2'), or say 'I pick the second one'. If unsure, say 'why?'."]
        return "\n".join(lines)
    
    def _parse_choice(self, user_text: str, q: MCQuestion) -> Optional[Tuple[int, str]]:
        """Parse user input to extract selected option."""
        text = self._norm(user_text)
        # numeric
        m = re.search(r"\b(\d{1,2})\b", text)
        if m:
            k = int(m.group(1))
            if 1 <= k <= len(q.options):
                return k - 1, q.options[k - 1]
        # ordinal
        for w, n in self._ORDINALS.items():
            if re.search(rf"\b{re.escape(w)}\b", text) and 1 <= n <= len(q.options):
                return n - 1, q.options[n - 1]
        # simple fuzzy token overlap
        matches = []
        for i, opt in enumerate(q.options):
            key = self._norm(opt)
            toks = [t for t in key.split() if len(t) > 2]
            hits = sum(1 for t in toks if t in text)
            if hits >= max(1, len(toks) // 2):
                matches.append((i, opt))
        if len(matches) == 1:
            return matches[0]
        return None
    
    def _ask_with_llm(self, q: MCQuestion) -> str:
        """Ask question using LLM."""
        system = SystemMessage(content=self.AGENT_SYSTEM_PROMPT)
        dev = HumanMessage(content=f"Ask the following question EXACTLY as written, with numbered options:\n\n{self._render_question(q)}")
        resp = self.llm.invoke([system, dev])
        return resp.content if isinstance(resp, AIMessage) else str(resp)
    
    def _explain_and_ask_with_llm(self, q: MCQuestion) -> str:
        """Explain guidance and re-ask question using LLM."""
        system = SystemMessage(content=self.AGENT_SYSTEM_PROMPT)
        dev = HumanMessage(content=(
            "The user asked why. Briefly explain using this guidance (verbatim as needed), "
            "then re-show the question with numbered options:\n\n"
            f"Guidance:\n{q.guidance}\n\nQuestion and options:\n{self._render_question(q)}"
        ))
        resp = self.llm.invoke([system, dev])
        return resp.content if isinstance(resp, AIMessage) else str(resp)
    
    def _retry_with_llm(self, q: MCQuestion) -> str:
        """Retry asking question after unclear input using LLM."""
        system = SystemMessage(content=self.AGENT_SYSTEM_PROMPT)
        dev = HumanMessage(content=f"User input did not clearly map to an option. Apologize briefly and re-show:\n\n{self._render_question(q)}")
        resp = self.llm.invoke([system, dev])
        return resp.content if isinstance(resp, AIMessage) else str(resp)
    
    def _finalize_with_tool_and_llm(self, state: AgentState) -> AgentState:
        """
        Finalize risk assessment by calculating allocation and providing summary.
        """
        # 1) Calculate risk allocation using the risk manager
        result = self.risk_manager.calculate_risk_allocation(state["answers"])
        state["risk"] = result or {}

        # 2) Build a deterministic summary (no LLM)
        eq = float(state["risk"].get("equity", 0.0))
        bd = float(state["risk"].get("bond", 0.0))
        eq_pct = round(eq * 100.0, 1)
        bd_pct = round(bd * 100.0, 1)

        # Map qid -> label from questions to ensure canonical labels
        qlabel_by_id = {q.id: q.label for q in self.risk_manager.questions}

        # Order the overview in the same order as questions
        lines = []
        lines.append("Thanks! Based on your responses, here's your preliminary portfolio guidance:")
        lines.append("")
        lines.append(f"**Allocation:** Equity {eq_pct:.1f}%  •  Bonds {bd_pct:.1f}%")
        lines.append("")
        lines.append("**Your answers** (for your records):")
        for q in self.risk_manager.questions:
            ans = state["answers"].get(q.id, {})
            sel = ans.get("selected_label", "")
            # Use the canonical label from MCQuestion
            label = qlabel_by_id.get(q.id, q.id)
            lines.append(f"- {label}: {sel}")

        lines.append("")
        lines.append(
            "Note: This allocation is a starting point derived from your emergency savings, account concentration, "
            "time horizon (adjusted for potential withdrawals), and your stated risk preferences. "
            "If anything changes, we can revisit the questionnaire to update your allocation."
        )

        msg = "\n".join(lines)
        state["messages"].append({"role": "ai", "content": msg})

        state["awaiting_input"] = False
        state["done"] = True                   
        return state
    
    def step(self, state: AgentState) -> AgentState:
        """
        Main step function for the risk agent.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated agent state
        """
        # finished already?
        if state.get("done"):
            return state

        # all questions answered -> finalize once
        if state["q_idx"] >= len(self.risk_manager.questions):
            # Sanity guard: if answers are incomplete, resume asking instead of finalizing
            if len(state["answers"]) < len(self.risk_manager.questions):
                state["done"] = False
                state["awaiting_input"] = False
                # jump to the first unanswered question
                state["q_idx"] = len(state["answers"])
                return state

            if not state.get("awaiting_input", False):
                state["done"] = True
                state["awaiting_input"] = False
                return self._finalize_with_tool_and_llm(state)
            return state  # no-op

        q = self.risk_manager.questions[state["q_idx"]]

        # If we haven't asked this question yet (awaiting_input is False), ask it now.
        if not state.get("awaiting_input", False):
            msg = self._ask_with_llm(q)
            state["messages"].append({"role": "ai", "content": msg})
            state["awaiting_input"] = True
            return state

        # We're awaiting input. Only proceed if the latest message is from the user.
        if not state["messages"] or state["messages"][-1].get("role") != "user":
            # No new user input since we asked -> do nothing (prevents double-ask)
            return state

        last_user = state["messages"][-1]["content"]

        if self._wants_guidance(last_user):
            msg = self._explain_and_ask_with_llm(q)
            state["messages"].append({"role": "ai", "content": msg})
            state["awaiting_input"] = True
            return state

        parsed = self._parse_choice(last_user, q)
        if not parsed:
            msg = self._retry_with_llm(q)
            state["messages"].append({"role": "ai", "content": msg})
            state["awaiting_input"] = True
            return state

        # record and advance
        idx, label = parsed
        mc = MCAnswer(selected_index=idx, selected_label=label, raw_user_text=last_user)
        state["answers"][q.id] = asdict(mc)
        state["q_idx"] += 1
        state["awaiting_input"] = False

        # If that was the last question, finalize on the next tick only once
        if state["q_idx"] >= len(self.risk_manager.questions):
            # Mark awaiting_input=False so it won't re-ask the last question on the next tick
            state["awaiting_input"] = False
            state["done"] = True
            return self._finalize_with_tool_and_llm(state)

        # Otherwise, ask the next question exactly once and await the next user turn
        nxt = self.risk_manager.questions[state["q_idx"]]
        msg = self._ask_with_llm(nxt)
        state["messages"].append({"role": "ai", "content": msg})
        state["awaiting_input"] = True
        return state

    def router(self, state: AgentState) -> str:
        """
        Router function that determines the next step based on risk agent completion.
        Only routes to reviewer when completely done with all questions.
        """
        # Only route to reviewer when completely done
        if state.get("done", False):
            return "reviewer_agent"
        
        # When waiting for input, don't route anywhere - just wait
        if state.get("awaiting_input", False):
            return "__end__"
        
        # If not done and not awaiting input, go to end to wait for user input
        return "__end__"

