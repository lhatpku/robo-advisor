# investment/fund_analysis_tool.py
from __future__ import annotations
from typing import Dict, Any, List
from langchain.tools import tool
from investment.fund_analyzer import FundAnalyzer

@tool("analyze_fund_performance")
def analyze_fund_performance(ticker: str) -> Dict[str, Any]:
    """
    Analyze fund performance and management metrics for a given ticker.
    
    Args:
        ticker: Fund ticker symbol (e.g., 'VUG', 'SPY', 'VTI')
        
    Returns:
        Dictionary containing comprehensive fund analysis
    """
    analyzer = FundAnalyzer()
    return analyzer.analyze_fund(ticker)


@tool("compare_fund_options")
def compare_fund_options(tickers: List[str]) -> Dict[str, Any]:
    """
    Compare multiple fund options side by side.
    
    Args:
        tickers: List of fund ticker symbols to compare
        
    Returns:
        Dictionary containing comparison analysis
    """
    analyzer = FundAnalyzer()
    return analyzer.compare_funds(tickers)


@tool("get_fund_recommendation")
def get_fund_recommendation(asset_class: str, criteria: str = "balanced") -> Dict[str, Any]:
    """
    Get fund recommendation for a specific asset class based on criteria.
    
    Args:
        asset_class: Asset class name (e.g., 'large_cap_growth', 'mid_term_treasury')
        criteria: Selection criteria ('balanced', 'low_cost', 'high_performance', 'low_risk')
        
    Returns:
        Dictionary containing fund recommendation and analysis
    """
    # Import the centralized asset class funds data
    from investment.config import get_fund_options
    
    # Get fund options for the asset class from centralized source
    fund_options = get_fund_options(asset_class)
    
    if not fund_options:
        return {
            "error": f"Unknown asset class: {asset_class}",
            "recommendation": None
        }
    
    # Analyze all options
    analyzer = FundAnalyzer()
    comparison = analyzer.compare_funds(fund_options)
    
    # Select best fund based on criteria
    recommendation = _select_best_fund(comparison, criteria)
    
    return {
        "asset_class": asset_class,
        "criteria": criteria,
        "fund_options": fund_options,
        "analysis": comparison,
        "recommendation": recommendation
    }

def _select_best_fund(comparison: Dict[str, Any], criteria: str) -> Dict[str, Any]:
    """Select the best fund based on specified criteria."""
    analyses = comparison.get("analyses", {})
    summary = comparison.get("summary", {})
    
    if not analyses:
        return {"error": "No fund analysis available"}
    
    # Remove funds with errors
    valid_analyses = {k: v for k, v in analyses.items() if "error" not in v}
    
    if not valid_analyses:
        return {"error": "No valid fund analyses available"}
    
    if criteria == "low_cost":
        # Select fund with lowest expense ratio
        lowest_cost = summary.get("lowest_cost", {})
        if lowest_cost:
            ticker = lowest_cost["ticker"]
            return {
                "ticker": ticker,
                "reason": f"Lowest expense ratio: {lowest_cost['expense_ratio']:.2%}",
                "analysis": valid_analyses.get(ticker, {})
            }
    
    elif criteria == "high_performance":
        # Select fund with highest return
        best_performers = summary.get("best_performers", {})
        if best_performers.get("highest_return"):
            ticker = best_performers["highest_return"]["ticker"]
            return {
                "ticker": ticker,
                "reason": f"Highest return: {best_performers['highest_return']['return']:.2f}%",
                "analysis": valid_analyses.get(ticker, {})
            }
    
    elif criteria == "low_risk":
        # Select fund with lowest volatility
        lowest_risk = summary.get("lowest_risk", {})
        if lowest_risk:
            ticker = lowest_risk["ticker"]
            return {
                "ticker": ticker,
                "reason": f"Lowest volatility: {lowest_risk['volatility']:.2f}%",
                "analysis": valid_analyses.get(ticker, {})
            }
    
    elif criteria == "balanced":
        # Select fund with highest Sharpe ratio (risk-adjusted return)
        highest_sharpe = summary.get("highest_sharpe", {})
        if highest_sharpe:
            ticker = highest_sharpe["ticker"]
            return {
                "ticker": ticker,
                "reason": f"Highest Sharpe ratio: {highest_sharpe['sharpe_ratio']:.2f}",
                "analysis": valid_analyses.get(ticker, {})
            }
    
    # Default: return first valid fund
    first_ticker = list(valid_analyses.keys())[0]
    return {
        "ticker": first_ticker,
        "reason": "Default selection (first available)",
        "analysis": valid_analyses[first_ticker]
    }


# Tool registry for easy integration
FUND_ANALYSIS_TOOLS = [
    analyze_fund_performance,
    compare_fund_options,
    get_fund_recommendation
]
