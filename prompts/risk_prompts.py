"""
Risk Agent Prompts and Examples

This file contains all the prompts, examples, and templates used by the risk agent
to keep the main agent code clean and maintainable.
"""

# Risk Intent Classification System Prompt
RISK_INTENT_SYSTEM_PROMPT = """You are a risk assessment agent. Classify user intent for risk profile management.

ACTIONS:
• set_equity: User wants to set equity allocation directly (e.g., "set as 0.6", "60%", "0.6")
• use_guidance: User wants to use questionnaire/guidance to determine allocation
• review_edit: User wants to review or edit current allocation (e.g., "review", "edit", "change")
• proceed: User wants to proceed to next phase (e.g., "proceed", "continue", "next", "yes")
• unknown: Unclear or unrecognized intent

EQUITY EXTRACTION:
- Extract equity value from patterns like "set as 0.6", "60%", "0.6", "set equity to 0.7"
- Convert percentages to decimals (60% → 0.6)
- Only extract if action is "set_equity"

EXAMPLES:
- "set as 0.6" → action="set_equity", equity_value=0.6
- "60%" → action="set_equity", equity_value=0.6  
- "0.6" → action="set_equity", equity_value=0.6
- "yes" → action="start_journey", equity_value=None
- "start" → action="start_journey", equity_value=None
- "begin" → action="start_journey", equity_value=None
- "use guidance" → action="use_guidance"
- "guidance" → action="use_guidance"
- "questionnaire" → action="use_guidance"
- "review" → action="review_edit"
- "proceed" → action="proceed"
- "hello" → action="unknown"
"""

# Risk Messages Class
class RiskMessages:
    """Risk agent system messages and responses."""
    
    @staticmethod
    def mode_selection() -> str:
        """Message for mode selection (direct vs guidance)."""
        return """Great! Let's define your risk profile. You have two options:

1) **Set your equity allocation directly** (e.g., 'set equity to 0.6' or '60%')
2) **Use guidance** to help you determine the right allocation through a questionnaire

Which would you prefer?"""
    
    @staticmethod
    def direct_equity_confirmation(equity: float) -> str:
        """Confirmation message for direct equity setting."""
        bond = 1.0 - equity
        return f"""Perfect! I've set your allocation to **{equity:.0%} equity / {bond:.0%} bonds**.

To continue, you can:
• **Review/edit** this allocation by saying 'review' or 'edit'
• **Use guidance** to reset through questionnaire by saying 'use guidance'
• **Proceed** to portfolio construction by saying 'proceed'"""
    
    @staticmethod
    def review_edit_message(equity: float) -> str:
        """Message for reviewing/editing current allocation."""
        bond = 1.0 - equity
        return f"""Current allocation: **{equity:.0%} equity / {bond:.0%} bonds**

You can:
• **Set a new equity** (e.g., 'set equity to 0.7' or '70%')
• **Use guidance** to reset through questionnaire
• **Proceed** to portfolio construction"""
    
    @staticmethod
    def questionnaire_question_template() -> str:
        """Template for questionnaire questions."""
        return """Reply with the option number (e.g., '2'), or say 'I pick the second one'. If unsure, say 'why?'."""
    
    @staticmethod
    def questionnaire_finalization(equity: float, bond: float, answers: dict) -> str:
        """Final message after questionnaire completion."""
        return f"""Thanks! Based on your responses, here's your preliminary portfolio guidance:

**Allocation:** Equity {equity:.1%}  •  Bonds {bond:.1%}

**Your answers** (for your records):
{RiskMessages._format_answers(answers)}

Note: This allocation is a starting point derived from your emergency savings, account concentration, time horizon (adjusted for potential withdrawals), and your stated risk preferences. If anything changes, we can revisit the questionnaire to update your allocation."""
    
    @staticmethod
    def _format_answers(answers: dict) -> str:
        """Format answers for display in finalization message."""
        formatted_answers = []
        for qid, answer in answers.items():
            formatted_answers.append(f"- {answer['question_text']}: {answer['selected_label']}")
        return "\n".join(formatted_answers)
    
    @staticmethod
    def unknown_intent() -> str:
        """Message for unknown or unclear intent."""
        return """I'm not sure what you'd like to do. You can:
• **Set your equity allocation** directly (e.g., 'set equity to 0.6')
• **Use guidance** to determine your allocation through a questionnaire
• **Review/edit** your current allocation if you have one set
• **Proceed** to the next step if your risk profile is complete"""
    
    @staticmethod
    def proceed_without_risk() -> str:
        """Message when proceeding without risk allocation."""
        return RiskMessages.mode_selection()
    
    @staticmethod
    def invalid_equity() -> str:
        """Message for invalid equity allocation."""
        return "Please provide an equity allocation between 0.05 and 0.95 (e.g., 0.70 for 70%)."
    
    @staticmethod
    def no_risk_allocation() -> str:
        """Message when no risk allocation is set."""
        return "No risk allocation set yet. Please set your equity allocation first."
    
    @staticmethod
    def unknown_questionnaire_response(question_text: str, options: str) -> str:
        """Message for unknown questionnaire response."""
        return f"I didn't understand your response. Please choose an option:\n\n{question_text}\n\n{options}"
