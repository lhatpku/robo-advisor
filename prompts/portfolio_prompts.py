"""
Portfolio Agent Prompts

This module contains all the prompts and system messages used by the portfolio agent.
"""

# Intent classification prompt
INTENT_CLASSIFICATION_PROMPT = """
You are a portfolio optimization assistant. Classify the user's intent from their input.

User input: "{user_input}"

Available actions:
- set_lambda: User wants to set the lambda parameter (e.g., "set lambda to 1.5", "lambda 2")
- set_cash: User wants to set the cash reserve parameter (e.g., "set cash to 0.03", "cash 0.02")
- run_optimization: User wants to run portfolio optimization (e.g., "run", "optimize", "go")
- review: User wants to review current portfolio (e.g., "review", "show", "display")
- proceed: User wants to proceed to next step (e.g., "proceed", "next", "continue", "looks good", "satisfied", "go ahead", "move on", "fine as is", "I'm done")
- unknown: Intent is unclear or not related to portfolio optimization

Extract the specific values if setting parameters. For cash reserve, use values between 0.02 and 0.05.
For lambda, use positive values typically between 0.1 and 10.

Examples:
- "set cash as 0.02" -> action: set_cash, cash_value: 0.02
- "lambda 1.5" -> action: set_lambda, lambda_value: 1.5
- "run" -> action: run_optimization
- "review" -> action: review
- "proceed" -> action: proceed
- "looks good" -> action: proceed
- "I'm satisfied" -> action: proceed
- "hello" -> action: unknown
"""

# System messages
class PortfolioMessages:
    """Portfolio agent system messages and responses."""
    
    @staticmethod
    def need_risk_data() -> str:
        """Message when risk data is not available."""
        return "I need the equity/bond recommendation from the risk Agent before I can build the portfolio."
    
    @staticmethod
    def lambda_set_success(lambda_value: float, cash_reserve: float) -> str:
        """Message when lambda is successfully set."""
        return f"✅ Set lambda to {lambda_value}. Current parameters: • Lambda: {lambda_value} • Cash Reserve: {cash_reserve:.2f}\n\nSay 'run' to optimize or 'set cash to X' to adjust further."
    
    @staticmethod
    def lambda_set_missing_value() -> str:
        """Message when lambda value is not specified."""
        return "Please specify a lambda value. For example: 'set lambda to 1.5' or 'lambda 2'"
    
    @staticmethod
    def cash_set_success(cash_value: float, lambda_value: float) -> str:
        """Message when cash reserve is successfully set."""
        return f"✅ Set cash reserve to {cash_value:.2f}. Current parameters: • Lambda: {lambda_value} • Cash Reserve: {cash_value:.2f}\n\nSay 'run' to optimize or 'set lambda to X' to adjust further."
    
    @staticmethod
    def cash_set_invalid_value(cash_value: float, min_cash: float, max_cash: float) -> str:
        """Message when cash reserve value is invalid."""
        return f"❌ Cash reserve must be between {min_cash:.2f} and {max_cash:.2f}. You entered {cash_value:.2f}."
    
    @staticmethod
    def cash_set_missing_value(min_cash: float, max_cash: float) -> str:
        """Message when cash reserve value is not specified."""
        return f"Please specify a cash reserve value between {min_cash:.2f} and {max_cash:.2f}. For example: 'set cash to 0.03' or 'cash 0.02'"
    
    @staticmethod
    def optimization_success(portfolio_table: str, note: str = "") -> str:
        """Message when optimization is successful."""
        return f"✅ **Optimization complete**{note}. I've built your asset-class portfolio.\n\n{portfolio_table}\n\n**What would you like to do next?**\n• **Review** weights: say 'review'\n• **Proceed** to ETF selection: say 'proceed'"
    
    @staticmethod
    def optimization_failed() -> str:
        """Message when optimization fails."""
        return "❌ **Portfolio optimization failed.** Please try again or adjust your parameters."
    
    @staticmethod
    def review_current_portfolio(portfolio_table: str, lambda_value: float, cash_reserve: float) -> str:
        """Message for reviewing current portfolio."""
        return f"**Your current portfolio:**\n\n{portfolio_table}\n\n**Current parameters:** • Lambda: {lambda_value} • Cash Reserve: {cash_reserve:.2f}\n\n**What would you like to do next?**\n• **Edit** parameters: say 'set lambda to X' or 'set cash to Y'\n• **Re-optimize**: say 'run' to optimize with current parameters\n• **Proceed** to ETF selection: say 'proceed'"
    
    @staticmethod
    def intro_message(lambda_value: float, cash_reserve: float, max_cash: float) -> str:
        """Intro message when no portfolio exists."""
        return (
            "Here's the plan: I'll build an asset-class portfolio using mean-variance optimization.\n"
            f"Defaults are **lambda = {lambda_value}** and **cash_reserve = {cash_reserve:.2f}**.\n"
            f"Say \"set lambda to 1\", \"set cash to {max_cash:.2f}\", or just \"run\" to optimize now."
        )
    
    @staticmethod
    def proceed_success() -> str:
        """Message when user wants to proceed."""
        return "✅ **Great!** Proceeding to the next step."
