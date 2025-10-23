"""
Investment Agent Prompts

This module contains all the prompts, messages, and structured output models used by the investment agent.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field

# Intent Classification Prompt
INTENT_CLASSIFICATION_PROMPT = """
You are an investment fund selection assistant. Classify the user's intent and extract relevant information.

User input: "{user_input}"

You must respond with a JSON object matching this exact structure:
{{
    "action": "string",
    "criteria": "string or null",
    "asset_class": "string or null", 
    "ticker": "string or null"
}}

Available actions and their expected outputs:

**create_investment** - User wants to start fund selection process
- Examples: "yes", "proceed", "start", "go ahead", "begin"
- Output: {{"action": "create_investment", "criteria": null, "asset_class": null, "ticker": null}}

**select_criteria** - User wants to choose selection criteria (ONLY when selecting investment criteria, NOT when editing funds)
- Examples: "1", "2", "3", "4", "balanced", "low cost", "high performance", "low risk"
- Output: {{"action": "select_criteria", "criteria": "low_risk", "asset_class": null, "ticker": null}}
- Criteria values: "balanced", "low_cost", "high_performance", "low_risk"
- Use this when user is choosing HOW to select funds (criteria selection)
- Specific mappings: "1"→"balanced", "2"→"low_cost", "3"→"high_performance", "4"→"low_risk"

**edit_asset_class** - User wants to edit a specific asset class or select from fund options (ONLY when editing existing funds)
- Examples: "edit", "large cap growth", "large cap value", "mid term treasury", "1", "2", "3" (when shown fund options for specific asset class)
- Output: {{"action": "edit_asset_class", "criteria": null, "asset_class": "large_cap_value", "ticker": null}}
- Asset class values: "large_cap_growth", "large_cap_value", "small_cap_growth", "small_cap_value", "developed_market_equity", "emerging_market_equity", "short_term_treasury", "mid_term_treasury", "long_term_treasury", "corporate_bond", "tips", "cash"
- IMPORTANT: Always extract the asset class name from user input and map it to the correct format
- Example: User says "large cap value" → Output: {{"action": "edit_asset_class", "criteria": null, "asset_class": "large_cap_value", "ticker": null}}

**analyze_fund** - User wants to analyze a specific fund
- Examples: "analyze VUG", "analysis SPY", "tell me about VTI", "VUG analysis"
- Output: {{"action": "analyze_fund", "criteria": null, "asset_class": null, "ticker": "VUG"}}
- Ticker: 3-5 uppercase letters (VUG, SPY, VTI, etc.)

**review_investment** - User wants to see current investment portfolio
- Examples: "review", "show", "display", "see", "current", "portfolio"
- Output: {{"action": "review_investment", "criteria": null, "asset_class": null, "ticker": null}}

**proceed** - User wants to move to next phase
- Examples: "done", "ok", "proceed", "continue", "ready", "complete", "finished", "next"
- Output: {{"action": "proceed", "criteria": null, "asset_class": null, "ticker": null}}

**unknown** - Intent is unclear or not related to investment selection
- Examples: "hello", "help", "what", "how", unclear input
- Output: {{"action": "unknown", "criteria": null, "asset_class": null, "ticker": null}}

**Context Rules:**
1. If user input is just a single number (1-4) and the conversation is about choosing investment criteria (balanced, low cost, etc.), use select_criteria
2. If user input is just a single number (1-9) and the conversation is about selecting from a list of fund options for a specific asset class, use edit_asset_class
3. If user input is "edit" or contains asset class names (like "large cap growth"), use edit_asset_class with mapped asset_class
4. If user input contains ticker symbols (like "VUG", "SPY"), use analyze_fund with extracted ticker
5. If user input is "yes", "proceed", "start", etc. and no investment exists yet, use create_investment
6. If user input is "review", "show", "display", etc., use review_investment
7. If user input is "done", "ok", "proceed", etc. and investment exists, use proceed
8. Use null for fields that don't apply to the action

**Asset Class Mapping (CRITICAL for edit_asset_class):**
- "large cap growth" → "large_cap_growth"
- "large cap value" → "large_cap_value"  
- "small cap growth" → "small_cap_growth"
- "small cap value" → "small_cap_value"
- "developed market" → "developed_market_equity"
- "emerging market" → "emerging_market_equity"
- "short term treasury" → "short_term_treasury"
- "mid term treasury" → "mid_term_treasury"
- "long term treasury" → "long_term_treasury"
- "corporate bond" → "corporate_bond"
- "tips" → "tips"
- "cash" → "cash"

**Key Distinction:**
- select_criteria: User is choosing HOW to select funds (criteria selection phase)
- edit_asset_class: User is choosing WHICH fund to use for a specific asset class (fund selection phase)

Respond with ONLY the JSON object, no other text.
"""

# Investment Messages Class
class InvestmentMessages:
    """Investment agent system messages and responses."""
    
    @staticmethod
    def need_portfolio_data() -> str:
        """Message when portfolio data is not available."""
        return "I need a portfolio allocation from the portfolio agent before I can help you select specific funds."
    
    @staticmethod
    def intro_message() -> str:
        """Intro message when no investment exists."""
        return """Great! Now I'll help you convert your asset-class allocation into a tradeable portfolio with specific funds and ETFs.

I'll select appropriate funds for each asset class based on your allocation weights. Would you like me to proceed with fund selection?"""
    
    @staticmethod
    def criteria_selection_message() -> str:
        """Message for fund selection criteria options."""
        return """**How would you like me to select funds for your portfolio?**

I can choose funds using different criteria. Please select one:

*Note: Cash positions will be handled as sweep accounts for trading reserves.*

**1. Balanced (Recommended)**
   • **What it means:** Selects funds with the best risk-adjusted returns (Sharpe ratio)
   • **Best for:** Most investors who want good returns without excessive risk
   • **Focus:** Maximizes return per unit of risk taken

**2. Low Cost**
   • **What it means:** Selects funds with the lowest expense ratios
   • **Best for:** Cost-conscious investors who prioritize keeping fees low
   • **Focus:** Minimizes ongoing costs and fees

**3. High Performance**
   • **What it means:** Selects funds with the highest historical returns
   • **Best for:** Aggressive investors willing to take more risk for higher returns
   • **Focus:** Maximizes potential returns (may be more volatile)

**4. Low Risk**
   • **What it means:** Selects funds with the lowest volatility (most stable)
   • **Best for:** Conservative investors who prioritize capital preservation
   • **Focus:** Minimizes price fluctuations and downside risk

**Please reply with:**
• **"1" or "balanced"** for balanced approach
• **"2" or "low cost"** for cost-focused selection  
• **"3" or "high performance"** for return-focused selection
• **"4" or "low risk"** for stability-focused selection"""
    
    @staticmethod
    def invalid_criteria_selection() -> str:
        """Message for invalid criteria selection."""
        return "Please select a valid option (1-4) or type the criteria name (balanced, low cost, high performance, low risk)."
    
    @staticmethod
    def investment_created(criteria_name: str) -> str:
        """Message when investment portfolio is created."""
        return f"""I've created your tradeable portfolio using the **{criteria_name}** selection criteria. Would you like to review and edit any fund selections? You can say the asset class name (e.g., 'large cap growth') to see alternative options, or 'analyze [ticker]' for detailed fund analysis."""
    
    @staticmethod
    def edit_asset_class_prompt() -> str:
        """Message when asking which asset class to edit."""
        return "Which asset class would you like to edit? Please say the asset class name (e.g., 'large cap growth', 'mid term treasury')."
    
    @staticmethod
    def need_investment_first() -> str:
        """Message when trying to edit before investment exists."""
        return "I need to create your investment portfolio first. Would you like me to proceed with fund selection?"
    
    @staticmethod
    def proceed_cancelled() -> str:
        """Message when user cancels the process."""
        return "No problem! You can return to this step later when you're ready to select specific funds."
    
    @staticmethod
    def unclear_intent() -> str:
        """Message for unclear user input."""
        return "Please let me know if you'd like to proceed with fund selection or if you have any questions."
    
    @staticmethod
    def asset_class_not_found(asset_class: str) -> str:
        """Message when asset class is not found."""
        return f"Asset class '{asset_class}' not found in your portfolio. Please check the spelling and try again."
    
    @staticmethod
    def fund_options_header(asset_class: str, current_ticker: str = None) -> str:
        """Header for showing fund options."""
        current_text = f" (currently: {current_ticker})" if current_ticker else ""
        return f"Here are the available funds for {asset_class}{current_text}:"
    
    @staticmethod
    def fund_options_footer(num_options: int) -> str:
        """Footer for fund options selection."""
        return f"Please select a number (1-{num_options}):"
    
    @staticmethod
    def invalid_fund_selection(num_options: int, asset_class: str) -> str:
        """Message for invalid fund selection."""
        return f"Please select a number from 1 to {num_options} for {asset_class}."
    
    @staticmethod
    def asset_class_updated(asset_class: str, ticker: str) -> str:
        """Message when asset class is updated."""
        return f"Updated {asset_class} to use {ticker}. Would you like to edit another asset class or review your portfolio?"
    
    @staticmethod
    def fund_analysis_prompt() -> str:
        """Message when asking for fund ticker to analyze."""
        return "Please specify a fund ticker symbol to analyze (e.g., 'analyze VUG' or 'analysis SPY')."
    
    @staticmethod
    def investment_ready() -> str:
        """Message when investment is ready to proceed."""
        return "Perfect! Your investment portfolio is ready. Moving to the next phase..."
    
    @staticmethod
    def help_message() -> str:
        """General help message."""
        return "You can say 'review' to see your current portfolio, mention an asset class name to edit it (e.g., 'large cap growth'), 'analyze [ticker]' to get detailed fund analysis, or 'proceed' to move to trading."
    
    @staticmethod
    def portfolio_display_header() -> str:
        """Header for portfolio display."""
        return "**Your Tradeable Portfolio:**"
    
    @staticmethod
    def portfolio_display_footer() -> str:
        """Footer for portfolio display."""
        return "*Total: 100.0%*"
    
    @staticmethod
    def selection_criteria_header(criteria_name: str) -> str:
        """Header for selection criteria display."""
        return f"**Selection Criteria: {criteria_name}**"
    
    @staticmethod
    def next_steps_options() -> str:
        """Options for next steps."""
        return """**What would you like to do next?**
• **Edit** specific asset classes
• **Proceed** to trading
• **Go back** to portfolio construction"""
    
    @staticmethod
    def fund_analysis_error(ticker: str, error: str) -> str:
        """Error message for fund analysis."""
        return f"❌ Error analyzing {ticker}: {error}"
    
    @staticmethod
    def fund_analysis_header(ticker: str) -> str:
        """Header for fund analysis."""
        return f"📊 **Fund Analysis: {ticker}**"
    
    @staticmethod
    def fund_analysis_basic_info(name: str = None, category: str = None, expense_ratio: float = None, aum: float = None) -> str:
        """Basic fund information."""
        info = ""
        if name:
            info += f"**Name:** {name}\n"
        if category:
            info += f"**Category:** {category}\n"
        if expense_ratio:
            info += f"**Expense Ratio:** {expense_ratio:.2%}\n"
        if aum:
            info += f"**Assets Under Management:** ${aum:,.0f}\n"
        return info
    
    @staticmethod
    def fund_analysis_performance_header() -> str:
        """Header for performance metrics."""
        return "**Performance Metrics:**"
    
    @staticmethod
    def fund_analysis_management_header() -> str:
        """Header for management metrics."""
        return "**Management Metrics:**"
    
    @staticmethod
    def fund_analysis_data_quality(data_quality: str) -> str:
        """Data quality information."""
        return f"**Data Quality:** {data_quality}"
