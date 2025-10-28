"""
Reviewer Agent Prompts
"""

from pydantic import BaseModel, Field
from typing import Literal


class ReviewerIntent(BaseModel):
    """Intent classification for reviewer agent user input."""
    action: Literal[
        "validate",        # Normal validation flow (from proceed after completing a phase)
        "start_over",      # User wants to start over
        "finish",          # User wants to finish
        "unknown"          # Unclear intent
    ] = "unknown"


REVIEWER_SYSTEM_PROMPT = """You are a Reviewer Agent responsible for validating completion of each phase and managing the overall flow.

Your responsibilities:
1. Validate that each phase (risk, portfolio, investment, trading) is complete
2. Ask users if they want to proceed or edit each completed phase
3. Set the next_phase field to guide the entry agent
4. Manage the overall flow between phases

You should be helpful, clear, and guide users through the process."""

REVIEWER_VALIDATION_PROMPTS = {
    "risk": {
        "complete": "Risk assessment is complete with equity/bond allocation.",
        "incomplete": "Risk assessment needs completion. Please complete the risk questionnaire."
    },
    "portfolio": {
        "complete": "Portfolio optimization is complete with asset class weights.",
        "incomplete": "Portfolio construction needs completion. Please run the optimization."
    },
    "investment": {
        "complete": "Investment selection is complete with specific fund choices.",
        "incomplete": "Investment selection needs completion. Please select funds for your portfolio."
    },
    "trading": {
        "complete": "Trading requests are complete and ready for execution.",
        "incomplete": "Trading requests need completion. Please generate trading orders."
    }
}

REVIEWER_PROCEED_PROMPTS = {
    "risk": "Risk assessment is complete! Would you like to **proceed** to portfolio construction or **edit** your risk allocation?",
    "portfolio": "Portfolio optimization is complete! Would you like to **proceed** to investment selection or **edit** your portfolio weights?",
    "investment": "Investment selection is complete! Would you like to **proceed** to trading requests or **edit** your fund choices?",
    "trading": "Trading requests are complete! Your portfolio is ready for execution. Would you like to **review** your trading orders or **edit** them?"
}

REVIEWER_COMPLETION_MESSAGE = """🎉 **All Phases Complete!**

Congratulations! You have successfully completed all phases of the robo-advisor process:

✅ **Risk Assessment** - Your risk tolerance and asset allocation
✅ **Portfolio Construction** - Optimized asset class weights  
✅ **Investment Selection** - Specific funds and ETFs chosen
✅ **Trading Requests** - Ready-to-execute trading orders

Your personalized investment plan is now ready! You can review any phase or proceed with execution."""


# Intent classification prompt
INTENT_CLASSIFICATION_PROMPT = """
You are a reviewer assistant. Classify the user's intent from their input.

User input: "{user_input}"

Available actions:
- validate: Normal validation flow (when user just completed a phase and reviewer needs to validate)
- start_over: User wants to start fresh with a new portfolio (e.g., "start over", "new portfolio", "reset", "restart")
- finish: User wants to finish/complete the session (e.g., "finish", "done", "complete", "thank you", "exit")
- unknown: Intent is unclear or not related to reviewer actions

Examples:
- "start over" -> action: start_over
- "new portfolio" -> action: start_over
- "reset" -> action: start_over
- "finish" -> action: finish
- "done" -> action: finish
- "thank you" -> action: finish
- "exit" -> action: finish
- "hello" -> action: unknown
- "" -> action: validate (empty or no input means normal validation)
"""


class ReviewerMessages:
    """Reviewer agent system messages and responses."""
    
    @staticmethod
    def final_summary_with_options() -> str:
        """Final summary message with next step options."""
        return """🎉 **Portfolio Planning Complete!**

Your personalized investment plan is ready. All phases have been successfully completed.

**What would you like to do next?**
• **Start over** - Create a new portfolio from scratch
• **Finish** - Complete the session and exit"""
    
    @staticmethod
    def thank_you_message() -> str:
        """Thank you message when user finishes."""
        return "Thank you for using the Robo-Advisor! Your personalized investment plan has been created and is ready for execution."
    
    @staticmethod
    def start_over_message() -> str:
        """Message when starting over."""
        return "Great! Let's start fresh with a new portfolio. How can I assist you today?"
