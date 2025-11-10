"""
Entry agent: first layer that greets users and routes them to the right phase.
"""
from typing import Dict, Any, Optional
from langchain_groq import ChatGroq
from state import AgentState
from prompts.entry_prompts import INTENT_CLASSIFICATION_PROMPT, EntryMessages, EntryIntent
from rag.retriever import build_pinecone_retriever, query_pinecone
import logging
import re
import json


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(ch)

class EntryAgent:
    """Manages initial user intent and routing between investment phases."""
    def __init__(self, llm: ChatGroq):
        # Use compatible LLM
        self.llm = llm

        # Bind a structured output parser so LLM always returns EntryIntent format
        self._structured_llm = llm.with_structured_output(EntryIntent).bind(temperature=0.0)

    def step(self, state: AgentState) -> AgentState:
        """
        It analyzes the current conversation state and decides what to do next.
        """
        # If user already indicated a phase, do nothing (prevents duplicate routing)
        if (
                state.get("intent_to_risk") or
                state.get("intent_to_portfolio") or
                state.get("intent_to_investment") or
                state.get("intent_to_trading")
        ):

            return state

        # Determine the next phase — default to 'risk' if reviewer hasn’t decided
        next_phase = state.get("next_phase", "risk") or "risk"

        # Ensure summary_shown exists in the state (avoid KeyError)
        if "summary_shown" not in state:
            state["summary_shown"] = {}

        # Show summary for the phase if it hasn't been shown yet
        if not state["summary_shown"].get(next_phase, False):
            state["summary_shown"][next_phase] = True
            return self._show_phase_summary(state, next_phase)

        # Only act when user has spoken (ignore system or AI turns)
        if not state.get("messages") or state["messages"][-1].get("role") != "user":
            return state

        # Get last user message
        last_user_message = state["messages"][-1].get("content", "")

        # Ask LLM to classify intent
        intent = self._classify_intent(last_user_message)
        logger.info(f"Intent detected: {intent}")

        # ─── Route based on classification ───
        if intent.action == "proceed":
            return self._handle_proceed_intent(state, next_phase)

        elif intent.action == "learn_more":
            question = intent.question or f"What is {next_phase} and how does it work?"
            return self._handle_learn_more_intent(state, question)

        # Fallback when LLM returns unexpected value
        state["messages"].append({
            "role": "ai",
            "content": EntryMessages.unclear_intent()
        })

    # ──────────────────────────────
    # Helpers
    # ──────────────────────────────

    def _classify_intent(self, user_input: str) -> EntryIntent:
        """LLM prompt to classify intent."""
        try:
            prompt = INTENT_CLASSIFICATION_PROMPT.format(user_input=user_input)
            raw = self._structured_llm.invoke(prompt)
            # try to get content attribute (AiMessage) or str
            raw_text = getattr(raw, "content", str(raw))
            # Try pydantic parsing first if structured call worked
            try:
                # If structured_llm returned a mapping-like object, __dict__ might work
                if isinstance(raw, dict):
                    return EntryIntent(**raw)
                # Try to parse as JSON
                parsed = json.loads(raw_text)
                return EntryIntent(**parsed)
            except Exception:
                # Fallback: try to extract intent heuristically
                return self._heuristic_parse_intent(raw_text)
        except Exception as e:
            logger.error(f"Intent classification failed: {e}", exc_info=True)
            # fallback: do heuristic parse from the raw user_input
            return self._heuristic_parse_intent(user_input)

    def _heuristic_parse_intent(self, text: str) -> EntryIntent:
        txt = text.lower()
        # Look for "learn", "explain", "what is", etc.
        if re.search(r"\b(learn|explain|what is|tell me more|how does)\b", txt):
            # try to extract a short question phrase
            qmatch = re.search(r"(what is .*|explain .*|tell me more about .*|how does .*|what does .*|why .*)", text,
                               re.I)
            question = qmatch.group(1).strip() if qmatch else None
            return EntryIntent(action="learn_more", question=question)
        if re.search(r"\b(proceed|next|yes|start|continue|ready|go ahead)\b", txt):
            return EntryIntent(action="proceed")
        # default fallback: ask to learn more
        return EntryIntent(action="learn_more", question=None)

    def _handle_proceed_intent(self, state: AgentState, next_phase: str) -> AgentState:
        """Set phase flags depending on what comes next."""
        if next_phase == "risk":
            state["intent_to_risk"] = True
        elif next_phase == "portfolio":
            state["intent_to_portfolio"] = True
        elif next_phase == "investment":
            state["intent_to_investment"] = True
        elif next_phase == "trading":
            state["intent_to_trading"] = True

        return state

    def _handle_learn_more_intent(self, state: AgentState, question: str) -> AgentState:
        """Provide detailed, context-aware explanations using Pinecone retrieval."""
        logger.info(f"Triggered learn_more for question: {question}")
        try:
            # STEP 1: Retrieve relevant context from Pinecone
            context = query_pinecone(question)
            context_text = context if context else "No relevant information found."
            if len(context_text) > 4000:
                context_text = context_text[:3900] + "\n\n[...context truncated...]"

            # STEP 2: Build augmented prompt
            augmented_prompt = EntryMessages.build_augmented_prompt(question, context_text)

            # STEP 3: Get response from LLM
            logger.info(f" Augmented prompt sent:\n{augmented_prompt[:300]}...")
            response = self.llm.invoke(augmented_prompt)
            raw_answer = getattr(response, "content", str(response)).strip()

            # STEP 4: Clean and user-facing answer (no context mentions)
            if "context does not include" in raw_answer.lower():
                answer = EntryMessages.not_enough_info_for_learn_more()
            else:
                answer = raw_answer  # Just the explanation itself

            # 🧹 Print nicely for console
            print("\n" + "=" * 70)
            print(f"🎓 **Learn More:** {question}\n")
            print(answer)
            print("=" * 70 + "\n")

            # STEP 5: Store in state
            answer_text = f"{answer}"
            state.setdefault("answers", {})
            state["answers"][question] = {"response": answer, "context_used": bool(context)}
            state["messages"].append({"role": "ai", "content": answer_text})

        except Exception as e:
            logger.error(f"Error in learn_more handler: {e}", exc_info=True)
            fallback = EntryMessages.fallback()
            state["messages"].append({"role": "ai", "content": fallback})
            print(fallback)

        # Prompt for next action
        state["messages"].append({
            "role": "ai",
            "content": "Would you like to proceed to the next phase, or learn more about another topic?"
        })
        return state

    def _handle_learn_more_intent_classic(self, state: AgentState, question: str) -> AgentState:

        """Provide explanations when user asks a question."""
        # Right now it's hardcoded; later this could call a RAG pipeline
        state["messages"].append({
            "role": "ai",
            "content": f"Great question: {question}\n\nLet me explain this part of the investment planning process..."
        })

        # Then invite user to continue
        state["messages"].append({
            "role": "ai",
            "content": "Would you like to proceed to the next phase, or do you have other questions?"
        })
        return state

    def _show_phase_summary(self, state: AgentState, completed_phase: str) -> AgentState:
        """Displays short summary message for a completed phase."""
        summary_message = EntryMessages.get_stage_summary(completed_phase)
        state["messages"].append({
            "role": "ai",
            "content": summary_message
        })
        return state

    # Router logic
    def router(self, state: AgentState) -> str:
        """
        Decides which agent should act next based on flags.
        """
        reviewer_status = state.get("status_tracking", {}).get("reviewer", {})
        if reviewer_status.get("awaiting_input"):
            return "reviewer_agent"

        if state.get("intent_to_risk"):
            return "risk_agent"
        elif state.get("intent_to_portfolio"):
            return "portfolio_agent"
        elif state.get("intent_to_investment"):
            return "investment_agent"
        elif state.get("intent_to_trading"):
            return "trading_agent"

        return "__end__"
