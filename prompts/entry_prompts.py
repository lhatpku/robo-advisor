"""
Entry Agent Prompts
"""

from typing import Optional, Literal, Dict
from pydantic import BaseModel

# ──────────────────────────────────────────────────────────────
# Intent schema for structured LLM output
# ──────────────────────────────────────────────────────────────
class EntryIntent(BaseModel):
    """Structured output for entry agent intent classification."""
    # The user's intent can only be to proceed or to learn more
    action: Literal["proceed", "learn_more"]

    # Optional structured question (used only for 'learn_more')
    question: Optional[str] = None


# Prompt template for classifying user intent

INTENT_CLASSIFICATION_PROMPT = """
You are an investment planning assistant. Classify the user's intent and extract relevant information.

User input: "{user_input}"

You must respond with a JSON object matching this exact structure:
{{
    "action": "proceed" or "learn_more",
    "question": "string or null"
}}

Definitions:
- "proceed": the user wants to move forward in the investment planning process.
- "learn_more": the user wants an explanation or clarification.

Examples:
1️⃣ "yes", "next", "start", "continue" → {{"action": "proceed", "question": null}}
2️⃣ "what is portfolio construction?" → {{"action": "learn_more", "question": "What is portfolio construction?"}}
3️⃣ "can you explain the risk phase?" → {{"action": "learn_more", "question": "Explain the risk phase"}}
4️⃣ "i’m ready" → {{"action": "proceed", "question": null}}
5️⃣ "tell me more about fund selection" → {{"action": "learn_more", "question": "Tell me more about fund selection"}}

Return ONLY valid JSON — no commentary, no explanations.
"""


# ──────────────────────────────────────────────────────────────
# Class for all entry-related AI messages
# ──────────────────────────────────────────────────────────────

class EntryMessages:
    """Defines all text templates for entry agent"""

    @staticmethod
    def welcome_message() -> str:
        return "Welcome to the AI Robo-Advisor, I will help you create a personalized investment plan"

    @staticmethod
    def next_phase_intro(phase: str) -> str:
        """
        explain what is coming into the next phase, dynamically based on the name
        """
        phase_descriptions = {
            "risk": "Risk Assessment",
            "portfolio": "Portfolio Construction",
            "investment": "Fund Selection",
            "trading": "Trading Implementation"
        }

        phase_name = phase_descriptions.get(phase, phase.title())

        return (f"Great! Let's move to **{phase_name}**. This phase will help you"
                f"{'assess your risk tolerance' if phase == 'risk' else 
                    'build your optimized portfolio' if phase == 'portfolio' else
                    'select funds and ETFs' if phase == 'investment' else
                    'generate trading requests'
                }. ")

    @staticmethod
    def phase_explanation(phase: str) -> str:
        """Detailed breakdown of each phase, used when user asks questions."""
        explanations: Dict[str, str] = {
            "risk": "This phase helps you determine your tolerance for market volatility.",
            "portfolio": "Here we build your optimized investment portfolio.",
            "investment": "We select the most suitable funds or ETFs for your profile.",
            "trading": "We execute your investment plan through simulated or live trades."
        }
        return explanations.get(phase, f"Information about {phase.title()} phase is not available.")

    @staticmethod
    def proceed_confirmation(phase: str) -> str:
        """Used when AI confirm user's intent to continue"""
        return (f"Perfect! Let's proceed to the {phase.title()} phase "
                f"Type 'yes' to continue or ask about this phase")

    @staticmethod
    def unclear_intent() -> str:
        """Fallback message for unrecognized input."""
        return "I'm not sure what you'd like to do. You can say 'proceed' or ask me about any phase."

    @staticmethod
    def phase_complete_message() -> str:
        return "Congratulations! You've completed all phases of the investment planning process."

    @staticmethod
    def build_augmented_prompt(question: str, context: str) -> str:
        """
        Constructs the prompt sent to the LLM for 'learn_more' questions.
        """
        context_text = context if context else "No relevant information found."
        if len(context_text) > 4000:
            context_text = context_text[:3900] + "\n\n[...context truncated...]"

        return f"""
            Your task:
            Answer the user’s question using ONLY the information provided in the context below.
            Write in a clear, friendly, professional tone suitable for an investor.

            If the context does not contain enough information to answer, respond:
            "The context does not include enough information to answer this question."

            User question:
            {question}

            Context:
            {context_text}
            """

# ──────────────────────────────────────────────────────────────
# Stage summaries shown after completing each phase
# ──────────────────────────────────────────────────────────────

    STAGE_SUMMARIES = {
            "risk": "**Start Risk Assessment** ✅\n\nWe'll identify your goals and tolerance.",
            "portfolio": "**Start Portfolio Construction** ✅\n\nWe'll optimize your allocation.",
            "investment": "**Start Fund Selection** ✅\n\nWe'll pick the best funds for each class.",
            "trading": "**Start Trading Execution** ✅\n\nWe'll create precise buy/sell requests."
        }

    @staticmethod
    def get_stage_summary(stage: str) -> str:
        """Return summary plus next-step options."""
        summary = EntryMessages.STAGE_SUMMARIES.get(stage, f"Stage {stage.title()} completed.")
        return(f"{summary}\n\n"
               "**What would you like to do next?**\n"
               "• **Proceed** to the phase\n"
               "• **Learn more** by asking a question\n")

    @staticmethod
    def not_enough_info_for_learn_more() -> str:
        return "Sorry, I don’t have enough information to answer that right now."

    @staticmethod
    def fallback() -> str:
        return "Sorry, I couldn’t fetch an explanation for this question."

