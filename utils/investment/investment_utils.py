"""
Investment Agent Utilities

This module contains all the utility functions used by the investment agent,
separated from the main agent logic for better organization and reusability.
"""

from typing import Dict, Any, List, Optional
from langchain_openai import ChatOpenAI
from utils.investment.fund_analyzer import FundAnalyzer
import yfinance as yf
from utils.investment.config import (
    get_fund_options, 
    get_selection_criteria, 
    is_cash_position, 
    get_cash_config,
    ASSET_CLASS_FUNDS
)
from prompts.investment_prompts import InvestmentMessages


class InvestmentUtils:
    """Utility class containing all investment-related helper functions."""
    
    def __init__(self, llm: ChatOpenAI):
        """Initialize the InvestmentUtils."""
        self.llm = llm
        self.fund_analyzer = FundAnalyzer()
    
    def create_initial_investment(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # Ask user to choose selection criteria
        state["messages"].append({
            "role": "ai",
            "content": InvestmentMessages.criteria_selection_message()
        })
        
        return state
    
    def handle_criteria_selection(self, state: Dict[str, Any], criteria: str = None) -> Dict[str, Any]:
        """Handle user selection of fund criteria."""
        if not criteria:
            # Parse user selection from last message
            last_user = state["messages"][-1].get("content", "").lower().strip()
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
                "content": InvestmentMessages.invalid_criteria_selection()
            })
            return state
        
        # Criteria selection completed
        portfolio = state.get("portfolio", {})
        
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
                    selected_fund = self.select_best_fund_for_asset_class(asset_class, criteria)
                    
                    # Fetch current price using yfinance
                    current_price = self._fetch_current_price(selected_fund["ticker"])
                    
                    investment[asset_class] = {
                        "weight": weight,
                        "ticker": selected_fund["ticker"],
                        "price": current_price,  # Add current price
                        "analysis": selected_fund.get("analysis", {}),
                        "selection_reason": selected_fund.get("reason", "Default selection"),
                        "criteria_used": criteria
                    }
        
        # Store investment in state
        state["investment"] = investment
        
        # Display the investment portfolio
        self.display_investment_portfolio(state, investment)
        
        # Show criteria used
        criteria_config = get_selection_criteria(criteria)
        criteria_name = criteria_config.get("name", criteria) if criteria_config else criteria
        
        state["messages"].append({
            "role": "ai",
            "content": InvestmentMessages.investment_created(criteria_name)
        })
        
        return state
    
    def handle_edit_mode(self, state: Dict[str, Any], edit_mode_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle user input when in edit mode for a specific asset class."""
        last_user = state["messages"][-1].get("content", "").strip()
        available_funds = edit_mode_data.get("options", [])
        asset_class = edit_mode_data.get("asset_class", "")
        
        # Check if user selected a fund option
        if last_user.isdigit():
            option_num = int(last_user)
            
            if 1 <= option_num <= len(available_funds):
                selected_fund = available_funds[option_num - 1]
                
                # Update the investment
                investment = state.get("investment", {})
                if asset_class in investment:
                    investment[asset_class]["ticker"] = selected_fund
                
                state["messages"].append({
                    "role": "ai",
                    "content": InvestmentMessages.asset_class_updated(asset_class, selected_fund)
                })
                return state
        
        # If not a valid selection, ask again
        state["messages"].append({
            "role": "ai",
            "content": InvestmentMessages.invalid_fund_selection(len(available_funds), asset_class)
        })
        return state
    
    def show_asset_class_options(self, state: Dict[str, Any], asset_class: str) -> Dict[str, Any]:
        """Show fund options for a specific asset class."""
        investment = state.get("investment", {})
        if asset_class not in investment:
            state["messages"].append({
                "role": "ai",
                "content": InvestmentMessages.asset_class_not_found(asset_class)
            })
            return None
        
        # Get available funds for this asset class
        available_funds = get_fund_options(asset_class)
        current_ticker = investment[asset_class]["ticker"]
        
        # Display options
        options_text = "\n".join([f"{i+1}. {fund}" for i, fund in enumerate(available_funds)])
        current_text = f" (currently: {current_ticker})" if current_ticker in available_funds else ""
        
        state["messages"].append({
            "role": "ai",
            "content": f"{InvestmentMessages.fund_options_header(asset_class, current_ticker)}\n\n{options_text}\n\n{InvestmentMessages.fund_options_footer(len(available_funds))}"
        })
        
        return {
            "asset_class": asset_class,
            "options": available_funds
        }
    
    def extract_asset_class(self, user_input: str) -> Optional[str]:
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
    
    def handle_fund_analysis_request(self, state: Dict[str, Any], ticker: str = None) -> Dict[str, Any]:
        """Handle user request to analyze a specific fund."""
        if not ticker:
            # Extract ticker from user input
            user_input = state["messages"][-1].get("content", "")
            words = user_input.split()
            
            # Look for ticker symbols (typically 3-5 uppercase letters)
            for word in words:
                if word.isupper() and 3 <= len(word) <= 5 and word.isalpha():
                    ticker = word
                    break
        
        if not ticker:
            state["messages"].append({
                "role": "ai",
                "content": InvestmentMessages.fund_analysis_prompt()
            })
            return state
        
        # Analyze the fund
        analysis_summary = self.analyze_fund_for_user(ticker)
        state["messages"].append({
            "role": "ai",
            "content": analysis_summary
        })
        
        return state
    
    def display_investment_portfolio(self, state: Dict[str, Any], investment: Dict[str, Any]) -> None:
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
        
        reasoning_text = f"\n{InvestmentMessages.selection_criteria_header(criteria_name)}\n"
        for asset_class, data in investment.items():
            if data.get("selection_reason"):
                display_name = asset_class.replace("_", " ").title()
                reasoning_text += f"• {display_name}: {data['selection_reason']}\n"
        
        portfolio_message = f"{InvestmentMessages.portfolio_display_header()}\n\n{table_text}\n\n{InvestmentMessages.portfolio_display_footer()}\n\n{reasoning_text}\n\n{InvestmentMessages.next_steps_options()}"
        
        state["messages"].append({
            "role": "ai",
            "content": portfolio_message
        })
    
    def select_best_fund_for_asset_class(self, asset_class: str, criteria: str = "balanced") -> Dict[str, Any]:
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
    
    def analyze_fund_for_user(self, ticker: str) -> str:
        """Analyze a specific fund and return a user-friendly summary."""
        try:
            analysis = self.fund_analyzer.analyze_fund(ticker)
            
            if "error" in analysis:
                return InvestmentMessages.fund_analysis_error(ticker, analysis['error'])
            
            fund_info = analysis.get("fund_info", {})
            performance = analysis.get("performance_metrics", {})
            management = analysis.get("management_metrics", {})
            
            summary = InvestmentMessages.fund_analysis_header(ticker) + "\n\n"
            
            # Basic info
            summary += InvestmentMessages.fund_analysis_basic_info(
                fund_info.get("name"),
                fund_info.get("category"),
                management.get("expense_ratio"),
                management.get("aum")
            )
            
            summary += f"\n{InvestmentMessages.fund_analysis_performance_header()}\n"
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
            
            summary += f"\n{InvestmentMessages.fund_analysis_management_header()}\n"
            if management.get("expense_ratio"):
                summary += f"• Expense Ratio: {management['expense_ratio']:.2%}\n"
            if management.get("aum"):
                summary += f"• Assets Under Management: ${management['aum']:,.0f}\n"
            if management.get("fund_age_years"):
                summary += f"• Fund Age: {management['fund_age_years']:.1f} years\n"
            if management.get("fund_family"):
                summary += f"• Fund Family: {management['fund_family']}\n"
            
            summary += f"\n{InvestmentMessages.fund_analysis_data_quality(analysis.get('data_quality', 'Unknown'))}"
            
            return summary
            
        except Exception as e:
            return InvestmentMessages.fund_analysis_error(ticker, str(e))
    
    def _fetch_current_price(self, ticker: str) -> float:
        """
        Fetch the current/latest close price for a ticker using yfinance.
        
        Args:
            ticker: Fund ticker symbol (e.g., 'VUG', 'SPY')
            
        Returns:
            Current price as float. Returns None if fetch fails.
        """
        try:
            from operation.retry.retry import retry_with_backoff
            from operation.retry.retry_config import YFINANCE_RETRY_CONFIG
            
            @retry_with_backoff(
                max_attempts=YFINANCE_RETRY_CONFIG.max_attempts,
                initial_delay=YFINANCE_RETRY_CONFIG.initial_delay,
                max_delay=YFINANCE_RETRY_CONFIG.max_delay,
                multiplier=YFINANCE_RETRY_CONFIG.multiplier,
                jitter=YFINANCE_RETRY_CONFIG.jitter,
                retryable_exceptions=YFINANCE_RETRY_CONFIG.retryable_exceptions,
                strategy=YFINANCE_RETRY_CONFIG.strategy
            )
            def _fetch_price():
                # Fetch ticker data
                fund = yf.Ticker(ticker)
                # Get latest close price (last trading day)
                hist = fund.history(period="1d")
                if not hist.empty:
                    return float(hist['Close'].iloc[-1])
                else:
                    # Fallback: try to get current price from info
                    info = fund.info
                    price = info.get('regularMarketPrice') or info.get('currentPrice')
                    return float(price) if price else None
            
            return _fetch_price()
                
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Error fetching price for {ticker}: {e}")
            return None
