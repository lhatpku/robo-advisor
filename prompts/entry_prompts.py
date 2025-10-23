"""
Entry agent prompts and messages.
"""

from pydantic import BaseModel
from typing import Optional, Literal

# Intent Classification Model
class EntryIntent(BaseModel):
    """Structured output for entry agent intent classification."""
    action: Literal["proceed", "learn_more"]  # Only these two options
    question: Optional[str] = None  # Structured question about what they want to learn

# Intent Classification Prompt
INTENT_CLASSIFICATION_PROMPT = """
You are an investment planning assistant. Classify the user's intent and extract relevant information.

User input: "{user_input}"

You must respond with a JSON object matching this exact structure:
{{
    "action": "proceed" or "learn_more",
    "question": "string or null"
}}

Available actions and their expected outputs:

**proceed** - User wants to continue to the next phase
- Examples: "yes", "proceed", "continue", "next", "go ahead", "start", "begin", "ok", "ready"
- Output: {{"action": "proceed", "question": null}}

**learn_more** - User wants to learn more about something
- Examples: "what is risk assessment", "tell me about portfolio", "explain investment selection", "how does trading work", "what happens in risk phase", "why do I need portfolio optimization"
- Output: {{"action": "learn_more", "question": "What is risk assessment and how does it work?"}}
- Convert user's question into a clear, structured question about the investment process

**Context Rules:**
1. If user input indicates they want to continue or move forward, use proceed
2. If user input asks questions about specific phases or wants explanations, use learn_more
3. Use null for question when action is proceed
4. Convert user's question into a clear, structured question when action is learn_more
5. Make questions specific and actionable (e.g., "What is risk assessment?" not "tell me about risk")

Respond with ONLY the JSON object, no other text.
"""

# Entry Messages Class
class EntryMessages:
    """Entry agent system messages and responses."""
    
    @staticmethod
    def welcome_message() -> str:
        """Welcome message for new users."""
        return "Welcome to the AI Robo-Advisor! I'll help you create a personalized investment plan through a structured process."
    
    @staticmethod
    def next_phase_intro(phase: str) -> str:
        """Introduction message for the next phase."""
        phase_descriptions = {
            "risk": "Risk Assessment",
            "portfolio": "Portfolio Construction", 
            "investment": "Fund Selection",
            "trading": "Trading Implementation"
        }
        
        phase_name = phase_descriptions.get(phase, phase.title())
        
        return f"Great! Let's move to **{phase_name}**. This phase will help you {'assess your risk tolerance and determine your asset allocation' if phase == 'risk' else 'build an optimized portfolio based on your risk profile' if phase == 'portfolio' else 'select specific funds and ETFs for your portfolio' if phase == 'investment' else 'generate trading requests to implement your investment plan'}."
    
    @staticmethod
    def phase_explanation(phase: str) -> str:
        """Detailed explanation of what a phase involves."""
        explanations = {
            "risk": """
**Risk Assessment Phase:**
• Complete a questionnaire about your investment goals and risk tolerance
• OR directly set your equity/bond allocation (e.g., 60% stocks, 40% bonds)
• This determines how aggressive or conservative your portfolio should be
• Based on your age, income, goals, and risk comfort level
            """,
            "portfolio": """
**Portfolio Construction Phase:**
• Uses mean-variance optimization to create an optimal asset allocation
• Considers your risk profile from the previous phase
• Allocates across different asset classes (large cap, small cap, international, bonds, etc.)
• Balances risk and return based on modern portfolio theory
            """,
            "investment": """
**Fund Selection Phase:**
• Converts your asset allocation into specific funds and ETFs
• Choose selection criteria: Balanced, Low Cost, High Performance, or Low Risk
• Selects the best funds for each asset class based on your criteria
• Provides detailed fund analysis and comparison options
            """,
            "trading": """
**Trading Implementation Phase:**
• Generates specific trading requests to implement your investment plan
• Considers your account value, current holdings, and tax implications
• Creates buy/sell orders with exact quantities and prices
• Handles rebalancing and position adjustments
            """
        }
        
        return explanations.get(phase, f"Information about {phase} phase is not available.")
    
    @staticmethod
    def proceed_confirmation(phase: str) -> str:
        """Confirmation message when user wants to proceed."""
        return f"Perfect! Let's proceed to the {phase} phase. Type 'yes' or 'proceed' to continue, or ask me about what this phase involves."
    
    @staticmethod
    def unclear_intent() -> str:
        """Message when user intent is unclear."""
        return "I'm not sure what you'd like to do. You can say 'proceed' to continue to the next phase, or ask me about any specific phase (risk assessment, portfolio construction, fund selection, or trading implementation)."
    
    @staticmethod
    def phase_complete_message() -> str:
        """Message when all phases are complete."""
        return "Congratulations! You've completed all phases of the investment planning process. Your personalized investment plan is ready."
    
    # Stage summaries dictionary
    STAGE_SUMMARIES = {
        "risk": "**Risk Assessment Complete** ✅\n\nYour risk profile has been determined based on your investment goals and risk tolerance. This will guide the portfolio construction in the next phase.",
        
        "portfolio": "**Portfolio Construction Complete** ✅\n\nYour optimized portfolio allocation has been created using mean-variance optimization. The next phase will select specific funds and ETFs for each asset class.",
        
        "investment": "**Fund Selection Complete** ✅\n\nYour investment portfolio is ready with specific funds and ETFs selected for each asset class. The final phase will generate trading requests to implement this portfolio.",
        
        "trading": "**Trading Implementation Complete** ✅\n\nYour trading plan is ready with specific buy/sell orders generated. Your complete investment plan is now ready for execution!"
    }
    
    @staticmethod
    def get_stage_summary(stage: str) -> str:
        """Get summary for a completed stage."""
        summary = EntryMessages.STAGE_SUMMARIES.get(stage, f"Stage {stage} completed.")
        
        return f"""{summary}

**What would you like to do next?**
• **Proceed** to the next phase
• **Learn more** about any phase by asking me questions
"""
