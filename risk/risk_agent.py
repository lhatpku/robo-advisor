# risk/risk_agent.py
from __future__ import annotations
from typing import TypedDict, List, Dict, Any, Optional, Literal
import re
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from .risk_manager import RiskManager, MCQuestion, MCAnswer
from state import AgentState
from prompts.risk_prompts import (
    RISK_INTENT_SYSTEM_PROMPT,
    MODE_SELECTION_MESSAGE,
    get_direct_equity_message,
    get_review_edit_message,
    get_questionnaire_question_template,
    get_questionnaire_finalization_message,
    UNKNOWN_INTENT_MESSAGE,
    PROCEED_WITHOUT_RISK_MESSAGE,
    INVALID_EQUITY_MESSAGE,
    NO_RISK_ALLOCATION_MESSAGE,
    UNKNOWN_QUESTIONNAIRE_RESPONSE_TEMPLATE
)


class RiskIntent(BaseModel):
    """Structured output for risk agent intent classification."""
    action: Literal[
        "set_equity",           # User wants to set equity directly
        "use_guidance",         # User wants to use questionnaire
        "review_edit",          # User wants to review/edit current allocation
        "proceed",              # User wants to proceed to next phase
        "start_journey",        # User wants to start the risk assessment journey
        "unknown"               # Unknown or unclear intent
    ] = "unknown"
    equity_value: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    reply: str = ""


class RiskAgent:
    """
    Risk assessment agent that handles both direct equity input and questionnaire-based risk profiling.
    Uses local state management and structured LLM output for intent classification.
    """
    
    def __init__(self, llm: ChatOpenAI):
        """
        Initialize the RiskAgent.
        
        Args:
            llm: ChatOpenAI instance for generating responses
        """
        self.llm = llm
        self.risk_manager = RiskManager()
        
        # Local state management (no global state fields)
        self._risk_intro_done = False
        self._in_questionnaire = False
        self._current_question_idx = 0
        
        # Structured LLM for intent classification
        self._structured_llm = llm.with_structured_output(RiskIntent).bind(temperature=0.0)
    
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
    
    def _classify_risk_intent(self, state: AgentState) -> RiskIntent:
        """Classify user intent using structured LLM output."""
        if not state.get("messages"):
            return RiskIntent(action="unknown", equity_value=None, reply="")
        
        last_user_msg = ""
        for msg in reversed(state["messages"]):
            if msg.get("role") == "user":
                last_user_msg = msg.get("content", "")
                break
        
        if not last_user_msg:
            return RiskIntent(action="unknown", equity_value=None, reply="")
        
        # Context information
        has_risk = bool(state.get("risk"))
        in_questionnaire = self._in_questionnaire
        current_question_idx = self._current_question_idx
        
        system = RISK_INTENT_SYSTEM_PROMPT

        user_prompt = f"""Context:
- Has risk allocation: {has_risk}
- In questionnaire: {in_questionnaire}
- Current question index: {current_question_idx}

User message: "{last_user_msg}"

Classify the intent and extract equity value if applicable."""

        try:
            intent = self._structured_llm.invoke([
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt}
            ])
            
            # Normalize return
            if isinstance(intent, dict):
                return RiskIntent(**intent)
            elif hasattr(intent, "model_dump"):
                return RiskIntent(**intent.model_dump())
            elif hasattr(intent, "dict"):
                return RiskIntent(**intent.dict())
            else:
                return RiskIntent(action="unknown", equity_value=None, reply="")
                
        except Exception:
            return RiskIntent(action="unknown", equity_value=None, reply="")
    
    def _ask_mode_selection(self, state: AgentState) -> AgentState:
        """Ask user to choose between direct equity or guidance."""
        msg = MODE_SELECTION_MESSAGE
        state["messages"].append({"role": "ai", "content": msg})
        self._set_status(state, "risk", awaiting_input=True)
        return state
    
    def _handle_direct_equity(self, state: AgentState, equity_value: float) -> AgentState:
        """Handle direct equity input."""
        # Validate equity range
        if not (0.05 <= equity_value <= 0.95):
            msg = INVALID_EQUITY_MESSAGE
            state["messages"].append({"role": "ai", "content": msg})
            self._set_status(state, "risk", awaiting_input=True)
            return state
        
        # Set risk allocation
        state["risk"] = {"equity": equity_value, "bond": 1.0 - equity_value}
        
        # Show confirmation and ask to proceed or review
        msg = get_direct_equity_message(equity_value)
        state["messages"].append({"role": "ai", "content": msg})
        self._set_status(state, "risk", done=True)
        self._set_status(state, "risk", awaiting_input=True)
        return state
    
    def _handle_review_edit(self, state: AgentState) -> AgentState:
        """Handle review/edit commands."""
        if not state.get("risk"):
            msg = NO_RISK_ALLOCATION_MESSAGE
            state["messages"].append({"role": "ai", "content": msg})
            self._set_status(state, "risk", awaiting_input=True)
            return state
        
        current_equity = state["risk"]["equity"]
        msg = get_review_edit_message(current_equity)
        state["messages"].append({"role": "ai", "content": msg})
        self._set_status(state, "risk", awaiting_input=True)
        return state
    
    def _handle_guidance_mode(self, state: AgentState) -> AgentState:
        """Start questionnaire mode."""
        # Reset questionnaire state
        self._in_questionnaire = True
        self._current_question_idx = 0
        state["answers"] = {}
        self._set_status(state, "risk", done=False)
        self._set_status(state, "risk", awaiting_input=False)
        
        # Clear existing risk allocation to start fresh
        state["risk"] = None
        
        # Start with first question
        return self._ask_current_question(state)
    
    def _ask_current_question(self, state: AgentState) -> AgentState:
        """Ask the current question in the questionnaire."""
        if self._current_question_idx >= len(self.risk_manager.questions):
            # All questions answered - finalize
            return self._finalize_questionnaire(state)
        
        q = self.risk_manager.questions[self._current_question_idx]
        
        # Render question with numbered options
        lines = [q.text, ""]
        for i, opt in enumerate(q.options, start=1):
            lines.append(f"{i}) {opt}")
        lines += ["", get_questionnaire_question_template()]
        
        msg = "\n".join(lines)
        state["messages"].append({"role": "ai", "content": msg})
        self._set_status(state, "risk", awaiting_input=True)
        return state
    
    def _handle_questionnaire_response(self, state: AgentState) -> AgentState:
        """Handle user response to questionnaire question."""
        if not state.get("messages") or state["messages"][-1].get("role") != "user":
            return state
        
        # Check bounds
        if self._current_question_idx >= len(self.risk_manager.questions):
            # All questions answered - finalize
            return self._finalize_questionnaire(state)
        
        last_user = state["messages"][-1]["content"]
        q = self.risk_manager.questions[self._current_question_idx]
        
        # Handle "why" requests
        if any(word in last_user.lower() for word in ["why", "explain", "not sure", "help"]):
            msg = f"{q.guidance}\n\n{q.text}\n\n"
            for i, opt in enumerate(q.options, start=1):
                msg += f"{i}) {opt}\n"
            msg += "\nReply with the option number (e.g., '2')."
            state["messages"].append({"role": "ai", "content": msg})
            self._set_status(state, "risk", awaiting_input=True)
            return state
        
        # Parse user's choice
        choice_result = self._parse_choice(last_user, q)
        if choice_result is None:
            # Unclear input -> retry
            options_text = "\n".join([f"{i}) {opt}" for i, opt in enumerate(q.options, start=1)])
            msg = UNKNOWN_QUESTIONNAIRE_RESPONSE_TEMPLATE.format(
                question_text=q.text,
                options=options_text
            ) + "\n\nReply with the option number (e.g., '2')."
            state["messages"].append({"role": "ai", "content": msg})
            self._set_status(state, "risk", awaiting_input=True)
            return state
        
        choice_idx, choice_text = choice_result
        
        # Store the answer
        qid = q.id
        if "answers" not in state:
            state["answers"] = {}
        state["answers"][qid] = {
            "question_id": qid,
            "selected_index": choice_idx,
            "selected_label": choice_text,
            "question_text": q.text,
            "options": q.options,
        }
        
        # Move to next question
        self._current_question_idx += 1
        
        # Ask the next question immediately
        return self._ask_current_question(state)
    
    def _parse_choice(self, user_text: str, q: MCQuestion) -> Optional[tuple[int, str]]:
        """Parse user input to extract selected option."""
        text = re.sub(r"\s+", " ", user_text.lower().strip())
        
        # Check for numeric input
        m = re.search(r"\b(\d{1,2})\b", text)
        if m:
            k = int(m.group(1))
            if 1 <= k <= len(q.options):
                return k - 1, q.options[k - 1]
        
        # Check for ordinal words
        ordinals = {
            "first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3,
            "fourth": 4, "4th": 4, "fifth": 5, "5th": 5, "sixth": 6, "6th": 6,
            "seventh": 7, "7th": 7, "eighth": 8, "8th": 8, "ninth": 9, "9th": 9,
            "tenth": 10, "10th": 10,
        }
        
        for word, num in ordinals.items():
            if re.search(rf"\b{re.escape(word)}\b", text) and 1 <= num <= len(q.options):
                return num - 1, q.options[num - 1]
        
        # Simple fuzzy token overlap
        matches = []
        for i, opt in enumerate(q.options):
            key = re.sub(r"\s+", " ", opt.lower().strip())
            toks = [t for t in key.split() if len(t) > 2]
            hits = sum(1 for t in toks if t in text)
            if hits >= max(1, len(toks) // 2):
                matches.append((i, opt))
        
        if len(matches) == 1:
            return matches[0]
        
        return None
    
    def _finalize_questionnaire(self, state: AgentState) -> AgentState:
        """Finalize questionnaire and calculate risk allocation."""
        # Calculate risk allocation using the risk manager
        result = self.risk_manager.calculate_risk_allocation(state["answers"])
        state["risk"] = result or {}
        
        # Build summary
        eq = float(state["risk"].get("equity", 0.0))
        bd = float(state["risk"].get("bond", 0.0))
        eq_pct = round(eq * 100.0, 1)
        bd_pct = round(bd * 100.0, 1)
        
        # Map qid -> label from questions
        qlabel_by_id = {q.id: q.label for q in self.risk_manager.questions}
        
        msg = get_questionnaire_finalization_message(eq_pct/100, bd_pct/100, state["answers"])
        state["messages"].append({"role": "ai", "content": msg})
        
        # Reset questionnaire state
        self._in_questionnaire = False
        self._current_question_idx = 0
        self._set_status(state, "risk", done=True)
        self._set_status(state, "risk", awaiting_input=True)
        
        return state
    
    def step(self, state: AgentState) -> AgentState:
        """
        Main step function for the risk agent.
        Uses structured LLM output for intent classification and local state management.
        """
        # Initialize global state if first time
        status = self._get_status(state, "risk")
        if not status["awaiting_input"] and not status["done"]:
            self._set_status(state, "risk", awaiting_input=True, done=False)
        
        # Only act on USER turns
        if not state.get("messages") or state["messages"][-1].get("role") != "user":
            return state
        
        # Classify user intent
        intent = self._classify_risk_intent(state)
        action = intent.action
        equity_value = intent.equity_value
        
        # Handle questionnaire responses
        if self._in_questionnaire:
            return self._handle_questionnaire_response(state)
        
        # Handle different actions
        if action == "set_equity" and equity_value is not None:
            return self._handle_direct_equity(state, equity_value)
        
        elif action == "start_journey":
            return self._ask_mode_selection(state)
        
        elif action == "use_guidance":
            return self._handle_guidance_mode(state)
        
        elif action == "review_edit":
            return self._handle_review_edit(state)
        
        elif action == "proceed":
            if state.get("risk"):
                self._set_status(state, "risk", done=True)
                self._set_status(state, "risk", awaiting_input=False)
                return state
            else:
                msg = PROCEED_WITHOUT_RISK_MESSAGE
                state["messages"].append({"role": "ai", "content": msg})
                self._set_status(state, "risk", awaiting_input=True)
                return state
        
        else:
            # Unknown action or no risk set yet - show intro or ask for clarification
            if not state.get("risk") and not self._risk_intro_done:
                self._risk_intro_done = True
                return self._ask_mode_selection(state)
            else:
                msg = UNKNOWN_INTENT_MESSAGE
                state["messages"].append({"role": "ai", "content": msg})
                self._set_status(state, "risk", awaiting_input=True)
                return state
    
    def router(self, state: AgentState) -> str:
        """
        Router function that determines the next step based on state.
        Simple routing logic like investment agent.
        """
        # If awaiting input, go to end to wait for user input
        status = self._get_status(state, "risk")
        if status["awaiting_input"]:
            return "__end__"
        
        # Check if user wants to use guidance mode (even if risk exists)
        if state.get("messages") and state["messages"][-1].get("role") == "user":
            last_user = state["messages"][-1]["content"].lower()
            if any(word in last_user for word in ["guidance", "questionnaire", "help me decide"]):
                return "__end__"  # Stay in risk agent to handle guidance
        
        # If risk exists and user wants to proceed, go to reviewer
        if state.get("risk") and status["done"]:
            return "reviewer_agent"
        
        # If risk doesn't exist yet, go to end to wait for user input
        return "__end__"