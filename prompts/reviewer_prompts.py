"""
Reviewer Agent Prompts
"""

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