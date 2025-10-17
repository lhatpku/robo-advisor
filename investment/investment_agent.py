# investment/investment_agent.py
from __future__ import annotations
from typing import Dict, Any, List, Optional
from langchain_openai import ChatOpenAI
from state import AgentState
from investment.fund_analyzer import FundAnalyzer
from investment.fund_analysis_tool import FUND_ANALYSIS_TOOLS
from investment.config import (
    get_fund_options, 
    get_selection_criteria, 
    is_cash_position, 
    get_cash_config,
    ASSET_CLASS_FUNDS
)


class InvestmentAgent:
    """
    Investment agent that handles the conversion of asset-class portfolios
    into tradeable portfolios with specific funds/ETFs.
    """
    
    def __init__(self, llm: ChatOpenAI):
        """
        Initialize the InvestmentAgent.
        
        Args:
            llm: ChatOpenAI instance for generating responses
        """
        self.llm = llm
        self.fund_analyzer = FundAnalyzer()
        
        # Local flags to control flow (similar to original GitHub version but local)
        self._investment_intro_done = False
        self._investment_criteria_selection = False
        self._investment_edit_mode = False
        self._investment_edit_asset_class = None
        self._investment_edit_options = None
    
    def step(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main step function for the investment agent.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated agent state
        """
        # Check if portfolio exists
        portfolio = state.get("portfolio", {})
        if not portfolio:
            state["messages"].append({
                "role": "ai", 
                "content": "I need a portfolio allocation from the portfolio agent before I can help you select specific funds."
            })
            return state
        
        # Only act on USER turns
        if not state.get("messages") or state["messages"][-1].get("role") != "user":
            return state

        last_user = state["messages"][-1].get("content", "").lower()
        
        # Get previous AI message
        prev_ai = ""
        if state.get("messages") and len(state["messages"]) > 1:
            for msg in reversed(state["messages"][:-1]):
                if msg.get("role") == "ai":
                    prev_ai = msg.get("content", "")
                    break

        # Handle review command first - show current investment portfolio
        if any(word in last_user for word in ["review", "show", "display", "see", "current"]):
            investment = state.get("investment")
            
            if investment and isinstance(investment, dict) and investment:
                # Show current investment portfolio
                self._display_investment_portfolio(state, investment)
            else:
                # Show intro message if no investment exists
                state["messages"].append({
                    "role":"ai",
                    "content": (
                        "Great! Now I'll help you convert your asset-class allocation into a tradeable portfolio "
                        "with specific funds and ETFs.\n\n"
                        "I'll select appropriate funds for each asset class based on your allocation weights. "
                        "Would you like me to proceed with fund selection?"
                    )
                })
            return state

        # Handle edit commands - allow editing specific asset classes
        if any(word in last_user for word in ["edit", "change", "modify", "adjust", "swap"]):
            if state.get("investment"):
                state["messages"].append({
                    "role":"ai",
                    "content": "Which asset class would you like to edit? Please say the asset class name (e.g., 'large cap growth', 'mid term treasury')."
                })
            else:
                state["messages"].append({
                    "role":"ai",
                    "content": "I need to create your investment portfolio first. Would you like me to proceed with fund selection?"
                })
            return state

        # Check if we're in criteria selection mode
        if self._investment_criteria_selection:
            return self._handle_criteria_selection(state)
        
        # Check if we're in edit mode (user wants to swap a ticker)
        if self._investment_edit_mode:
            return self._handle_edit_mode(state)
        
        # Check if investment already exists
        if state.get("investment"):
            return self._handle_existing_investment(state)
        
        # One-time intro on entry
        if not self._investment_intro_done:
            state["messages"].append({
                "role": "ai",
                "content": (
                    "Great! Now I'll help you convert your asset-class allocation into a tradeable portfolio "
                    "with specific funds and ETFs.\n\n"
                    "I'll select appropriate funds for each asset class based on your allocation weights. "
                    "Would you like me to proceed with fund selection?"
                )
            })
            self._investment_intro_done = True
            return state
        
        # Handle user input for initial fund selection
        if any(word in last_user for word in ["yes", "proceed", "continue", "go ahead", "start"]):
            return self._create_initial_investment(state)
        
        elif any(word in last_user for word in ["no", "cancel", "back", "return"]):
            state["messages"].append({
                "role": "ai",
                "content": "No problem! You can return to this step later when you're ready to select specific funds."
            })
            state["intent_to_investment"] = False
            return state
        
        else:
            state["messages"].append({
                "role": "ai",
                "content": "Please let me know if you'd like to proceed with fund selection or if you have any questions."
            })
            return state
    
    def _create_initial_investment(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Create initial investment portfolio with fund selections."""
        portfolio = state.get("portfolio", {})
        asset_weights = portfolio
        
        # Ask user to choose selection criteria
        state["messages"].append({
            "role": "ai",
            "content": self._get_criteria_selection_message()
        })
        
        # Set up criteria selection mode
        self._investment_criteria_selection = True
        
        return state
    
    def _get_criteria_selection_message(self) -> str:
        """Get the message explaining fund selection criteria options."""
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
• **"4" or "low risk"** for stability-focused selection
    """
    
    def _handle_criteria_selection(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle user selection of fund criteria."""
        last_user = state["messages"][-1].get("content", "").lower().strip()
        portfolio = state.get("portfolio", {})
        
        # Parse user selection
        criteria = None
        if last_user in ["1", "balanced"]:
            criteria = "balanced"
        elif last_user in ["2", "low cost", "lowcost"]:
            criteria = "low_cost"
        elif last_user in ["3", "high performance", "highperformance"]:
            criteria = "high_performance"
        elif last_user in ["4", "low risk", "lowrisk"]:
            criteria = "low_risk"
        
        if not criteria:
            state["messages"].append({
                "role": "ai",
                "content": "Please select a valid option (1-4) or type the criteria name (balanced, low cost, high performance, low risk)."
            })
            return state
        
        # Criteria selection completed
        
        # Create investment portfolio with selected criteria
        asset_weights = portfolio
        investment = {}
        
        for asset_class, weight in asset_weights.items():
            if weight > 0:  # Only include assets with positive weights
                # Special handling for CASH - use sweep_cash instead of fund analysis
                if asset_class == "cash":
                    investment[asset_class] = {
                        "weight": weight,
                        "ticker": "sweep_cash",
                        "analysis": {},
                        "selection_reason": "Cash position - using sweep account for trading reserve",
                        "criteria_used": "cash_reserve"
                    }
                else:
                    # Use fund analyzer to select best fund for this asset class with chosen criteria
                    selected_fund = self._select_best_fund_for_asset_class(asset_class, criteria)
                    investment[asset_class] = {
                        "weight": weight,
                        "ticker": selected_fund["ticker"],
                        "analysis": selected_fund.get("analysis", {}),
                        "selection_reason": selected_fund.get("reason", "Default selection"),
                        "criteria_used": criteria
                    }
        
        # Store investment in state
        state["investment"] = investment
        
        # Display the investment portfolio
        self._display_investment_portfolio(state, investment)
        
        # Show criteria used
        criteria_config = get_selection_criteria(criteria)
        criteria_name = criteria_config.get("name", criteria) if criteria_config else criteria
        
        state["messages"].append({
            "role": "ai",
            "content": (
                f"I've created your tradeable portfolio using the **{criteria_name}** selection criteria. "
                "Would you like to review and edit any fund selections? "
                "You can say the asset class name (e.g., 'large cap growth') to see alternative options, or 'analyze [ticker]' for detailed fund analysis."
            )
        })
        
        # Clear criteria selection mode
        self._investment_criteria_selection = False
        
        return state
    
    def _handle_existing_investment(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle user interaction when investment already exists."""
        last_user = state["messages"][-1].get("content", "").lower().strip()
        
        if any(word in last_user for word in ["review", "show", "display", "see"]):
            investment = state.get("investment", {})

            self._display_investment_portfolio(state, investment)
            return state
        
        elif any(word in last_user for word in ["edit", "change", "swap", "modify"]):
            state["messages"].append({
                "role": "ai",
                "content": "Which asset class would you like to edit? Please say the asset class name (e.g., 'large cap growth', 'mid term treasury')."
            })
            return state
        
        # Check if user wants to analyze a specific fund
        if "analyze" in last_user or "analysis" in last_user:
            return self._handle_fund_analysis_request(state, last_user)
        
        # Check if user wants to proceed to trading
        if state.get("portfolio") and state.get("investment") and state.get("messages") and state["messages"][-1].get("role") == "user":
            last_user = state["messages"][-1].get("content", "").lower()
            if last_user.strip() in ["done", "ok", "okay", "good", "fine", "next", "proceed", "continue", "ready", "complete", "finished"]:
                state["intent_to_investment"] = False
                state["intent_to_trading"] = True  # Route to trading agent
                state["messages"].append({
                    "role": "ai",
                    "content": "Perfect! Your investment portfolio is ready. Proceeding to generate trading requests..."
                })
                return state


        # if any(word in last_user for word in ["done", "ok", "okay", "good", "fine", "next", "proceed", "continue", "ready", "complete", "finished"]):
        #     state["messages"].append({
        #         "role": "ai",
        #         "content": "Perfect! Your investment portfolio is ready. You can now proceed to generate trading requests if you'd like to see how to execute this portfolio."
        #     })
        #     state["intent_to_investment"] = False
        #     state["intent_to_trading"] = True  # Route to trading agent
        #     state["done"] = True
        #     return state
        
        # Check if user mentioned a specific asset class
        asset_class = self._extract_asset_class(last_user)
        if asset_class:
            return self._show_asset_class_options(state, asset_class)
        
        state["messages"].append({
            "role": "ai",
            "content": "You can say 'review' to see your current portfolio, mention an asset class name to edit it (e.g., 'large cap growth'), 'analyze [ticker]' to get detailed fund analysis, or 'proceed' to move to trading."
        })
        return state
    
    def _handle_edit_mode(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle user input when in edit mode for a specific asset class."""
        last_user = state["messages"][-1].get("content", "").strip()
        
        # Check if user selected a fund option
        if last_user.isdigit():
            option_num = int(last_user)
            available_funds = self._investment_edit_options or []
            
            if 1 <= option_num <= len(available_funds):
                selected_fund = available_funds[option_num - 1]
                
                # Update the investment
                investment = state.get("investment", {})
                if self._investment_edit_asset_class in investment:
                    investment[self._investment_edit_asset_class]["ticker"] = selected_fund
                
                # Clear edit mode
                self._investment_edit_mode = False
                self._investment_edit_asset_class = None
                self._investment_edit_options = None
                
                state["messages"].append({
                    "role": "ai",
                    "content": f"Updated {self._investment_edit_asset_class} to use {selected_fund}. Would you like to edit another asset class or review your portfolio?"
                })
                return state
        
        # If not a valid selection, ask again
        state["messages"].append({
            "role": "ai",
            "content": f"Please select a number from 1 to {len(self._investment_edit_options or [])} for {self._investment_edit_asset_class}."
        })
        return state
    
    def _show_asset_class_options(self, state: Dict[str, Any], asset_class: str) -> Dict[str, Any]:
        """Show fund options for a specific asset class."""
        portfolio = state.get("portfolio", {})
        investment = state.get("investment", {})
        
        if asset_class not in investment:
            state["messages"].append({
                "role": "ai",
                "content": f"Asset class '{asset_class}' not found in your portfolio. Please check the spelling and try again."
            })
            return state
        
        # Get available funds for this asset class
        available_funds = get_fund_options(asset_class)
        current_ticker = investment[asset_class]["ticker"]
        
        # Set up edit mode
        self._investment_edit_mode = True
        self._investment_edit_asset_class = asset_class
        self._investment_edit_options = available_funds
        
        # Display options
        options_text = "\n".join([f"{i+1}. {fund}" for i, fund in enumerate(available_funds)])
        current_text = f" (currently: {current_ticker})" if current_ticker in available_funds else ""
        
        state["messages"].append({
            "role": "ai",
            "content": f"Here are the available funds for {asset_class}{current_text}:\n\n{options_text}\n\nPlease select a number (1-{len(available_funds)}):"
        })
        
        return state
    
    def _extract_asset_class(self, user_input: str) -> Optional[str]:
        """Extract asset class name from user input."""
        # Map common user terms to asset class names
        asset_class_mapping = {
            "large cap growth": "large_cap_growth",
            "large cap value": "large_cap_value", 
            "small cap growth": "small_cap_growth",
            "small cap value": "small_cap_value",
            "developed market": "developed_market_equity",
            "emerging market": "emerging_market_equity",
            "short term treasury": "short_term_treasury",
            "mid term treasury": "mid_term_treasury", 
            "long term treasury": "long_term_treasury",
            "corporate bond": "corporate_bond",
            "tips": "tips",
            "cash": "cash"
        }
        
        user_input_lower = user_input.lower()
        for user_term, asset_class in asset_class_mapping.items():
            if user_term in user_input_lower:
                return asset_class
        
        return None
    
    def _handle_fund_analysis_request(self, state: Dict[str, Any], user_input: str) -> Dict[str, Any]:
        """Handle user request to analyze a specific fund."""
        # Extract ticker from user input
        words = user_input.split()
        ticker = None
        
        # Look for ticker symbols (typically 3-5 uppercase letters)
        for word in words:
            if word.isupper() and 3 <= len(word) <= 5 and word.isalpha():
                ticker = word
                break
        
        if not ticker:
            state["messages"].append({
                "role": "ai",
                "content": "Please specify a fund ticker symbol to analyze (e.g., 'analyze VUG' or 'analysis SPY')."
            })
            return state
        
        # Analyze the fund
        analysis_summary = self._analyze_fund_for_user(ticker)
        state["messages"].append({
            "role": "ai",
            "content": analysis_summary
        })
        
        return state
    
    def _display_investment_portfolio(self, state: Dict[str, Any], investment: Dict[str, Any]) -> None:
        """Display the investment portfolio in a formatted table with analysis."""
        if not investment:
            return
        
        # Create formatted table
        lines = ["| Asset Class | Weight | Ticker | Analysis |", "|-------------|--------|--------|----------|"]
        
        for asset_class, data in investment.items():
            weight_pct = data["weight"] * 100
            ticker = data["ticker"]
            analysis = data.get("analysis", {})
            selection_reason = data.get("selection_reason", "Default")
            
            # Format asset class name for display
            display_name = asset_class.replace("_", " ").title()
            
            # Create analysis summary
            analysis_summary = ""
            if ticker == "sweep_cash":
                analysis_summary = "Sweep Account"
            elif analysis.get("performance_metrics", {}).get("sharpe_ratio"):
                sharpe = analysis["performance_metrics"]["sharpe_ratio"]
                analysis_summary = f"Sharpe: {sharpe:.2f}"
            elif analysis.get("management_metrics", {}).get("expense_ratio"):
                expense = analysis["management_metrics"]["expense_ratio"]
                analysis_summary = f"Expense: {expense:.2%}"
            else:
                analysis_summary = "Basic"
            
            lines.append(f"| {display_name} | {weight_pct:.1f}% | {ticker} | {analysis_summary} |")
        
        table_text = "\n".join(lines)
        
        # Add selection reasoning
        criteria_used = None
        for asset_class, data in investment.items():
            if data.get("criteria_used"):
                criteria_used = data["criteria_used"]
                break
        
        criteria_config = get_selection_criteria(criteria_used)
        criteria_name = criteria_config.get("name", criteria_used) if criteria_config else criteria_used
        
        reasoning_text = f"\n**Selection Criteria: {criteria_name}**\n"
        for asset_class, data in investment.items():
            if data.get("selection_reason"):
                display_name = asset_class.replace("_", " ").title()
                reasoning_text += f"• {display_name}: {data['selection_reason']}\n"
        
        portfolio_message = f"**Your Tradeable Portfolio:**\n\n{table_text}\n\n*Total: 100.0%*\n\n{reasoning_text}\n\n**What would you like to do next?**\n• **Edit** specific asset classes\n• **Proceed** to trading\n• **Go back** to portfolio construction"

        
        state["messages"].append({
            "role": "ai",
            "content": portfolio_message
        })

    
    def _select_best_fund_for_asset_class(self, asset_class: str, criteria: str = "balanced") -> Dict[str, Any]:
        """Select the best fund for a given asset class using fund analysis."""
        try:
            # Get fund options for this asset class
            fund_options = get_fund_options(asset_class)
            
            if not fund_options or fund_options[0] == "UNKNOWN":
                return {
                    "ticker": "UNKNOWN",
                    "reason": "No funds available for this asset class",
                    "analysis": {}
                }
            
            # Analyze all fund options
            comparison = self.fund_analyzer.compare_funds(fund_options)
            
            # Select the best fund based on chosen criteria
            summary = comparison.get("summary", {})
            
            if criteria == "balanced":
                # Select fund with highest Sharpe ratio
                highest_sharpe = summary.get("highest_sharpe", {})
                if highest_sharpe:
                    ticker = highest_sharpe["ticker"]
                    analysis = comparison.get("analyses", {}).get(ticker, {})
                    return {
                        "ticker": ticker,
                        "reason": f"Best risk-adjusted return (Sharpe: {highest_sharpe['sharpe_ratio']:.2f})",
                        "analysis": analysis
                    }
            
            elif criteria == "low_cost":
                # Select fund with lowest expense ratio
                lowest_cost = summary.get("lowest_cost", {})
                if lowest_cost:
                    ticker = lowest_cost["ticker"]
                    analysis = comparison.get("analyses", {}).get(ticker, {})
                    return {
                        "ticker": ticker,
                        "reason": f"Lowest expense ratio ({lowest_cost['expense_ratio']:.2%})",
                        "analysis": analysis
                    }
            
            elif criteria == "high_performance":
                # Select fund with highest return
                best_performers = summary.get("best_performers", {})
                if best_performers.get("highest_return"):
                    ticker = best_performers["highest_return"]["ticker"]
                    analysis = comparison.get("analyses", {}).get(ticker, {})
                    return {
                        "ticker": ticker,
                        "reason": f"Highest return ({best_performers['highest_return']['return']:.2f}%)",
                        "analysis": analysis
                    }
            
            elif criteria == "low_risk":
                # Select fund with lowest volatility
                lowest_risk = summary.get("lowest_risk", {})
                if lowest_risk:
                    ticker = lowest_risk["ticker"]
                    analysis = comparison.get("analyses", {}).get(ticker, {})
                    return {
                        "ticker": ticker,
                        "reason": f"Lowest volatility ({lowest_risk['volatility']:.2f}%)",
                        "analysis": analysis
                    }
            
            # Fallback to first available fund
            first_ticker = fund_options[0]
            analysis = comparison.get("analyses", {}).get(first_ticker, {})
            return {
                "ticker": first_ticker,
                "reason": "Default selection (first available)",
                "analysis": analysis
            }
            
        except Exception as e:
            # Fallback to first fund if analysis fails
            fund_options = get_fund_options(asset_class)
            return {
                "ticker": fund_options[0] if fund_options else "UNKNOWN",
                "reason": f"Analysis failed, using default: {str(e)}",
                "analysis": {}
            }
    
    def _analyze_fund_for_user(self, ticker: str) -> str:
        """Analyze a specific fund and return a user-friendly summary."""
        try:
            analysis = self.fund_analyzer.analyze_fund(ticker)
            
            if "error" in analysis:
                return f"❌ Error analyzing {ticker}: {analysis['error']}"
            
            fund_info = analysis.get("fund_info", {})
            performance = analysis.get("performance_metrics", {})
            management = analysis.get("management_metrics", {})
            
            summary = f"📊 **Fund Analysis: {ticker}**\n\n"
            
            # Basic info
            if fund_info.get("name"):
                summary += f"**Name:** {fund_info['name']}\n"
            if fund_info.get("category"):
                summary += f"**Category:** {fund_info['category']}\n"
            if management.get("expense_ratio"):
                summary += f"**Expense Ratio:** {management['expense_ratio']:.2%}\n"
            if management.get("aum"):
                summary += f"**Assets Under Management:** ${management['aum']:,.0f}\n"
            
            summary += "\n**Performance Metrics:**\n"
            if performance.get("annualized_return"):
                summary += f"• Annualized Return: {performance['annualized_return']:.2f}%\n"
            if performance.get("volatility"):
                summary += f"• Volatility: {performance['volatility']:.2f}%\n"
            if performance.get("sharpe_ratio"):
                summary += f"• Sharpe Ratio: {performance['sharpe_ratio']:.2f}\n"
            if performance.get("max_drawdown"):
                summary += f"• Max Drawdown: {performance['max_drawdown']:.2f}%\n"
            if performance.get("beta"):
                summary += f"• Beta: {performance['beta']:.2f}\n"
            
            summary += "\n**Management Metrics:**\n"
            if management.get("expense_ratio"):
                summary += f"• Expense Ratio: {management['expense_ratio']:.2%}\n"
            if management.get("aum"):
                summary += f"• Assets Under Management: ${management['aum']:,.0f}\n"
            if management.get("fund_age_years"):
                summary += f"• Fund Age: {management['fund_age_years']:.1f} years\n"
            if management.get("fund_family"):
                summary += f"• Fund Family: {management['fund_family']}\n"
            
            summary += f"\n**Data Quality:** {analysis.get('data_quality', 'Unknown')}"
            
            return summary
            
        except Exception as e:
            return f"❌ Error analyzing {ticker}: {str(e)}"
