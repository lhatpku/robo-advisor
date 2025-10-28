"""
Trading Agent Prompts

This module contains all the prompts and system messages used by the trading agent.
"""

from pydantic import BaseModel, Field
from typing import Literal, Optional, Dict

# Scenario Selection Model
class ScenarioSelectionIntent(BaseModel):
    """Intent classification for scenario selection."""
    action: Literal[
        "select_scenario",        # User wants to select a scenario
        "custom_portfolio",       # User wants to use custom portfolio
        "unknown"                 # Unclear intent
    ] = "unknown"
    
    scenario_number: Optional[int] = Field(
        default=None,
        description="Scenario number (1-6) if action is select_scenario"
    )

# Intent Classification Model
class TradingIntent(BaseModel):
    """Intent classification for trading agent user input."""
    action: Literal[
        "set_tax_weight",         # User wants to set tax_weight parameter
        "set_ltcg_rate",          # User wants to set ltcg_rate parameter
        "set_integer_shares",     # User wants to set integer_shares parameter
        "run_rebalancing",        # User wants to execute rebalancing
        "review",                 # User wants to review current configuration
        "proceed",                # User wants to proceed to next phase
        "unknown"                 # Unclear intent
    ] = "unknown"
    
    tax_weight: Optional[float] = Field(
        default=None,
        description="Tax weight value if action is set_tax_weight (e.g., '0.5' for moderate, '2' for high importance)"
    )
    
    ltcg_rate: Optional[float] = Field(
        default=None,
        description="Long-term capital gains tax rate if action is set_ltcg_rate (range: 0 to 0.35)"
    )
    
    integer_shares: Optional[bool] = Field(
        default=None,
        description="Integer shares flag if action is set_integer_shares (True/False)"
    )


# Scenario selection classification prompt
SCENARIO_SELECTION_PROMPT = """
You are a portfolio scenario selection assistant. Classify the user's intent from their input.

User input: "{user_input}"

Available scenarios: 1-6 (Conservative Retiree, Young Professional, Mid-Career Balanced, High Net Worth, New Investor, Pre-Retirement)

Available actions:
- select_scenario: User wants to select a demo scenario (e.g., "1", "2", "3", "select 4", "choose scenario 5", "use scenario 2")
- custom_portfolio: User wants to use custom portfolio (e.g., "custom", "my portfolio", "actual", "real")
- unknown: Intent is unclear

Extract the scenario number (1-6) if user is selecting a scenario.

Examples:
- "1" -> action: select_scenario, scenario_number: 1
- "select 2" -> action: select_scenario, scenario_number: 2
- "choose scenario 5" -> action: select_scenario, scenario_number: 5
- "use scenario 3" -> action: select_scenario, scenario_number: 3
- "custom" -> action: custom_portfolio, scenario_number: null
- "my portfolio" -> action: custom_portfolio, scenario_number: null
- "hello" -> action: unknown, scenario_number: null
"""

# Intent classification prompt
INTENT_CLASSIFICATION_PROMPT = """
You are a trading execution assistant. Classify the user's intent from their input.

User input: "{user_input}"

Available actions:
- set_tax_weight: User wants to set tax weight preference (e.g., "set tax weight to 1.5", "override tax weight as 2", "tax_weight = 0.5")
- set_ltcg_rate: User wants to set long-term capital gains tax rate (e.g., "set ltcg to 0.20", "capital gains 15%", "override capital gain as 0.16")
- set_integer_shares: User wants to set integer shares option (e.g., "allow fractional shares", "integer shares only", "set fractional as false")
- run_rebalancing: User wants to execute trading rebalancing (e.g., "execute", "run", "run rebalancing", "generate trades", "create trades", "trade now")
- review: User wants to review current configuration (e.g., "review", "show", "display", "current settings")
- proceed: User wants to proceed to next phase (e.g., "proceed", "next", "continue", "looks good")
- unknown: Intent is unclear or not related to trading configuration

Extract the specific values if setting parameters. For ltcg_rate, use values between 0 and 0.35 (representing 0% to 35% tax rate).

Examples:
- "set tax weight to 1.5" -> action: set_tax_weight, tax_weight: 1.5
- "tax_weight = 0.5" -> action: set_tax_weight, tax_weight: 0.5
- "override tax weight as 2" -> action: set_tax_weight, tax_weight: 2
- "set ltcg to 0.20" -> action: set_ltcg_rate, ltcg_rate: 0.20
- "capital gains 15%" -> action: set_ltcg_rate, ltcg_rate: 0.15
- "override capital gain as 0.16" -> action: set_ltcg_rate, ltcg_rate: 0.16
- "allow fractional shares" -> action: set_integer_shares, integer_shares: False
- "integer shares only" -> action: set_integer_shares, integer_shares: True
- "execute" -> action: run_rebalancing
- "run" -> action: run_rebalancing
- "run rebalancing" -> action: run_rebalancing
- "execute the trading" -> action: run_rebalancing
- "generate trades" -> action: run_rebalancing
- "review" -> action: review
- "proceed" -> action: proceed
- "looks good" -> action: proceed
- "hello" -> action: unknown
"""


# System messages
class TradingMessages:
    """Trading agent system messages and responses."""
    
    @staticmethod
    def need_investment_data() -> str:
        """Message when investment data is not available."""
        return "I need the investment portfolio from the investment agent before I can generate trading requests. Please complete the investment selection first."
    
    @staticmethod
    def intro_message(tax_weight: float, ltcg_rate: float, integer_shares: bool, has_trades: bool = False) -> str:
        """Intro message for trading agent.
        
        Args:
            tax_weight: Tax weight parameter
            ltcg_rate: Long-term capital gains rate
            integer_shares: Whether to use integer shares
            has_trades: Whether trading requests have been generated (default: False)
        """
        shares_type = "Integer shares only" if integer_shares else "Fractional shares allowed"
        
        if has_trades:
            # Show all options after trades have been generated
            return (
                "I'll help you generate executable trading requests for your portfolio.\n\n"
                f"**Current Configuration:**\n"
                f"• Tax weight: {tax_weight}\n"
                f"• Long-term capital gains rate: {ltcg_rate*100:.1f}%\n"
                f"• Trading type: {shares_type}\n\n"
                f"**What would you like to do?**\n"
                f"• **Adjust** parameters: say 'set tax weight to X', 'set ltcg to Y', or 'integer shares only'\n"
                f"• **Execute** trading: say 'execute' or 'run'\n"
                f"• **Review** configuration: say 'review'\n"
                f"• **Proceed** with defaults: say 'proceed'"
            )
        else:
            # Hide review and proceed options before first run
            return (
                "I'll help you generate executable trading requests for your portfolio.\n\n"
                f"**Current Configuration:**\n"
                f"• Tax weight: {tax_weight}\n"
                f"• Long-term capital gains rate: {ltcg_rate*100:.1f}%\n"
                f"• Trading type: {shares_type}\n\n"
                f"**What would you like to do?**\n"
                f"• **Adjust** parameters: say 'set tax weight to X', 'set ltcg to Y', or 'integer shares only'\n"
                f"• **Execute** trading: say 'execute' or 'run'"
            )
    
    @staticmethod
    def tax_weight_set_success(tax_weight: float, ltcg_rate: float, integer_shares: bool) -> str:
        """Message when tax weight is successfully set."""
        shares_type = "Integer shares only" if integer_shares else "Fractional shares allowed"
        return (
            f"✅ Set tax weight to {tax_weight}.\n\n"
            f"**Current Configuration:**\n"
            f"• Tax weight: {tax_weight}\n"
            f"• Long-term capital gains rate: {ltcg_rate*100:.1f}%\n"
            f"• Trading type: {shares_type}\n\n"
            f"Say 'execute' to generate trading requests or adjust other parameters."
        )
    
    @staticmethod
    def tax_weight_invalid() -> str:
        """Message when tax weight value is invalid."""
        return "❌ Tax weight must be a positive number. Please try again (e.g., 'set tax weight to 1.5')."
    
    @staticmethod
    def ltcg_rate_set_success(ltcg_rate: float, tax_weight: float, integer_shares: bool) -> str:
        """Message when ltcg rate is successfully set."""
        shares_type = "Integer shares only" if integer_shares else "Fractional shares allowed"
        return (
            f"✅ Set long-term capital gains rate to {ltcg_rate*100:.1f}%.\n\n"
            f"**Current Configuration:**\n"
            f"• Tax weight: {tax_weight}\n"
            f"• Long-term capital gains rate: {ltcg_rate*100:.1f}%\n"
            f"• Trading type: {shares_type}\n\n"
            f"Say 'execute' to generate trading requests or adjust other parameters."
        )
    
    @staticmethod
    def ltcg_rate_invalid(ltcg_rate: float) -> str:
        """Message when ltcg rate value is invalid."""
        return (
            f"❌ Long-term capital gains rate must be between 0% and 35%. You entered {ltcg_rate*100:.1f}%.\n"
            f"Please enter a value between 0 and 0.35 (e.g., 'set ltcg to 0.15' for 15%)."
        )
    
    @staticmethod
    def integer_shares_set_success(integer_shares: bool, tax_weight: float, ltcg_rate: float) -> str:
        """Message when integer shares is successfully set."""
        shares_type = "Integer shares only" if integer_shares else "Fractional shares allowed"
        return (
            f"✅ Set trading type to {shares_type}.\n\n"
            f"**Current Configuration:**\n"
            f"• Tax weight: {tax_weight}\n"
            f"• Long-term capital gains rate: {ltcg_rate*100:.1f}%\n"
            f"• Trading type: {shares_type}\n\n"
            f"Say 'execute' to generate trading requests or adjust other parameters."
        )
    
    @staticmethod
    def review_configuration(tax_weight: float, ltcg_rate: float, integer_shares: bool) -> str:
        """Message for reviewing current configuration."""
        shares_type = "Integer shares only" if integer_shares else "Fractional shares allowed"
        return (
            f"**Current Trading Configuration:**\n\n"
            f"• **Tax weight:** {tax_weight}\n"
            f"  - Higher values prioritize tax savings over tracking error reduction\n"
            f"  - Lower values focus on minimizing tracking error\n\n"
            f"• **Long-term capital gains rate:** {ltcg_rate*100:.1f}%\n"
            f"  - Your expected tax rate on long-term capital gains\n\n"
            f"• **Trading type:** {shares_type}\n\n"
            f"**What would you like to do?**\n"
            f"• **Adjust** parameters\n"
            f"• **Execute** trading with these settings\n"
            f"• **Proceed** to final review"
        )
    
    @staticmethod
    def rebalancing_in_progress() -> str:
        """Message when rebalancing is starting."""
        return "⏳ Running portfolio rebalancing... This may take a moment."
    
    @staticmethod
    def rebalancing_success(trades_summary: str, result_summary) -> str:
        """Message when rebalancing is successful."""
        return (
            f"✅ Trading requests generated successfully!\n\n"
            f"{trades_summary}\n\n"
            f"**Rebalancing Summary:**\n"
            f"• Initial tracking error: {result_summary.get('initial_tracking_error', 0):.4f}\n"
            f"• Final tracking error: {result_summary.get('final_tracking_error', 0):.4f}\n"
            f"• Total trades: {len(result_summary.get('trades', []))}\n"
            f"• Realized gains: ${result_summary.get('realized_net_gains', 0):.2f}\n"
            f"• Estimated tax cost: ${result_summary.get('estimated_tax_cost', 0):.2f}\n\n"
            f"**Next Step?**\n"
            f"• **Adjust** parameters: say 'set tax weight to X', 'set ltcg to Y', or 'integer shares only'\n"
            f"• **Review** configuration: say 'review'\n"
            f"• **Proceed** with defaults: say 'proceed'"
        )
    
    @staticmethod
    def rebalancing_failed() -> str:
        """Message when rebalancing fails."""
        return "❌ Trading rebalancing failed. Please try again or adjust your parameters."
    
    @staticmethod
    def proceed_success() -> str:
        """Message when user wants to proceed."""
        return "Great! Proceeding to the final review phase."

