# investment/fund_analyzer.py
from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json

class FundAnalyzer:
    """
    Fund analysis tool that retrieves performance and management metrics
    for mutual funds and ETFs using various data sources.
    """
    
    def __init__(self, alpha_vantage_key: Optional[str] = None, morningstar_key: Optional[str] = None):
        """
        Initialize the FundAnalyzer.
        
        Args:
            alpha_vantage_key: API key for Alpha Vantage (optional)
            morningstar_key: API key for Morningstar (optional)
        """
        self.alpha_vantage_key = alpha_vantage_key
        self.morningstar_key = morningstar_key
    
    def analyze_fund(self, ticker: str) -> Dict[str, Any]:
        """
        Perform comprehensive fund analysis.
        
        Args:
            ticker: Fund ticker symbol (e.g., 'VUG', 'SPY')
            
        Returns:
            Dictionary containing fund analysis metrics
        """
        try:
            # Get basic fund data
            fund_data = self._get_basic_fund_data(ticker)
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(ticker)
            
            # Get management metrics
            management_metrics = self._get_management_metrics(ticker)
            
            # Combine all metrics
            analysis = {
                "ticker": ticker,
                "fund_info": fund_data,
                "performance_metrics": performance_metrics,
                "management_metrics": management_metrics,
                "analysis_date": datetime.now().isoformat(),
                "data_quality": self._assess_data_quality(fund_data, performance_metrics, management_metrics)
            }
            
            return analysis
            
        except Exception as e:
            return {
                "ticker": ticker,
                "error": f"Failed to analyze fund: {str(e)}",
                "analysis_date": datetime.now().isoformat()
            }
    
    def _get_basic_fund_data(self, ticker: str) -> Dict[str, Any]:
        """Get basic fund information from yfinance."""
        try:
            fund = yf.Ticker(ticker)
            info = fund.info
            
            # Extract relevant fund information (only fields reliably available in yfinance)
            fund_data = {
                "name": info.get("longName", "Unknown"),
                "category": info.get("category", "Unknown"),
                "expense_ratio": info.get("expenseRatio", None),
                "aum": info.get("totalAssets", None),
                "inception_date": info.get("firstTradeDateEpochUtc", None),
                "currency": info.get("currency", "USD"),
                "exchange": info.get("exchange", "Unknown"),
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
                "description": info.get("longBusinessSummary", ""),
                "website": info.get("website", ""),
                "current_price": info.get("regularMarketPrice", None),
                "previous_close": info.get("previousClose", None),
                "day_change": info.get("regularMarketChange", None),
                "day_change_percent": info.get("regularMarketChangePercent", None),
                "fund_family": info.get("fundFamily", "Unknown")
            }
            
            # Convert epoch to date if available
            if fund_data["inception_date"]:
                fund_data["inception_date"] = datetime.fromtimestamp(fund_data["inception_date"]).strftime("%Y-%m-%d")
            
            return fund_data
            
        except Exception as e:
            return {"error": f"Failed to get basic fund data: {str(e)}"}
    
    def _calculate_performance_metrics(self, ticker: str, period: str = "5y") -> Dict[str, Any]:
        """Calculate performance metrics from historical data."""
        try:
            fund = yf.Ticker(ticker)
            hist = fund.history(period=period)
            
            if hist.empty:
                return {"error": "No historical data available"}
            
            # Calculate returns
            returns = hist['Close'].pct_change().dropna()
            
            # Basic performance metrics
            total_return = (hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1) * 100
            annualized_return = ((hist['Close'].iloc[-1] / hist['Close'].iloc[0]) ** (252 / len(hist)) - 1) * 100
            volatility = returns.std() * np.sqrt(252) * 100
            
            # Risk metrics
            sharpe_ratio = (annualized_return / 100) / (volatility / 100) if volatility > 0 else 0
            
            # Drawdown analysis
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min() * 100
            
            # Beta calculation (using S&P 500 as benchmark)
            beta = self._calculate_beta(ticker, returns)
            
            return {
                "total_return_5y": round(total_return, 2),
                "annualized_return": round(annualized_return, 2),
                "volatility": round(volatility, 2),
                "sharpe_ratio": round(sharpe_ratio, 2),
                "max_drawdown": round(max_drawdown, 2),
                "beta": round(beta, 2),
                "data_points": len(hist)
            }
            
        except Exception as e:
            return {"error": f"Failed to calculate performance metrics: {str(e)}"}
    
    def _calculate_beta(self, ticker: str, returns: pd.Series) -> float:
        """Calculate beta relative to S&P 500 (simplified)."""
        try:
            # Get S&P 500 data for comparison
            spy = yf.Ticker("SPY")
            spy_hist = spy.history(period="5y")
            spy_returns = spy_hist['Close'].pct_change().dropna()
            
            # Align the data
            common_dates = returns.index.intersection(spy_returns.index)
            if len(common_dates) < 30:  # Need at least 30 data points
                return 1.0  # Default beta
            
            aligned_returns = returns.loc[common_dates]
            aligned_spy = spy_returns.loc[common_dates]
            
            # Calculate beta
            covariance = np.cov(aligned_returns, aligned_spy)[0, 1]
            spy_variance = np.var(aligned_spy)
            
            beta = covariance / spy_variance if spy_variance > 0 else 1.0
            return beta
            
        except Exception as e:
            return 1.0  # Default beta if calculation fails
    
    def _get_management_metrics(self, ticker: str) -> Dict[str, Any]:
        """Get management-related metrics."""
        try:
            fund = yf.Ticker(ticker)
            info = fund.info
            
            # Extract management metrics from available data (yfinance limitations noted)
            management_metrics = {
                "expense_ratio": info.get("expenseRatio", None),
                "aum": info.get("totalAssets", None),
                "inception_date": info.get("firstTradeDateEpochUtc", None),
                "fund_family": info.get("fundFamily", "Unknown"),
                "management_company": info.get("companyName", "Unknown"),
                "minimum_investment": info.get("minimumInitialInvestment", None),
                "redemption_fee": info.get("redemptionFee", None),
                "load": info.get("load", None),
                # Note: These fields are not available in yfinance
                "portfolio_turnover": "Not available via yfinance",
                "management_tenure": "Not available via yfinance", 
                "number_of_holdings": "Not available via yfinance",
                "manager_ownership": "Not available via yfinance"
            }
            
            # Convert epoch to date if available
            if management_metrics["inception_date"]:
                inception_date = datetime.fromtimestamp(management_metrics["inception_date"])
                management_metrics["inception_date"] = inception_date.strftime("%Y-%m-%d")
                # Calculate fund age
                fund_age_years = (datetime.now() - inception_date).days / 365.25
                management_metrics["fund_age_years"] = round(fund_age_years, 1)
            
            return management_metrics
            
        except Exception as e:
            return {"error": f"Failed to get management metrics: {str(e)}"}
    
    def _assess_data_quality(self, fund_data: Dict, performance: Dict, management: Dict) -> str:
        """Assess the quality of available data from yfinance."""
        score = 0
        max_score = 8  # Adjusted for yfinance limitations
        
        # Check basic data availability
        if fund_data.get("name") and fund_data["name"] != "Unknown":
            score += 1
        if fund_data.get("expense_ratio") is not None:
            score += 1
        if fund_data.get("aum") is not None:
            score += 1
        
        # Check performance data availability (calculated from historical data)
        if performance.get("annualized_return") is not None:
            score += 2
        if performance.get("volatility") is not None:
            score += 1
        if performance.get("sharpe_ratio") is not None:
            score += 1
        
        # Check management data availability
        if management.get("inception_date") is not None:
            score += 1
        
        percentage = (score / max_score) * 100
        
        if percentage >= 80:
            return "High"
        elif percentage >= 60:
            return "Medium"
        else:
            return "Low"
    
    def compare_funds(self, tickers: List[str]) -> Dict[str, Any]:
        """
        Compare multiple funds side by side.
        
        Args:
            tickers: List of fund ticker symbols
            
        Returns:
            Dictionary containing comparison data
        """
        analyses = {}
        
        for ticker in tickers:
            analyses[ticker] = self.analyze_fund(ticker)
        
        # Create comparison summary
        comparison = {
            "funds_analyzed": len(tickers),
            "analyses": analyses,
            "summary": self._create_comparison_summary(analyses),
            "comparison_date": datetime.now().isoformat()
        }
        
        return comparison
    
    def _create_comparison_summary(self, analyses: Dict[str, Dict]) -> Dict[str, Any]:
        """Create a summary comparison of multiple funds."""
        summary = {
            "best_performers": {},
            "lowest_cost": {},
            "lowest_risk": {},
            "highest_sharpe": {}
        }
        
        valid_analyses = {k: v for k, v in analyses.items() if "error" not in v}
        
        if not valid_analyses:
            return summary
        
        # Find best performers by return
        returns = {}
        for ticker, analysis in valid_analyses.items():
            perf = analysis.get("performance_metrics", {})
            if "annualized_return" in perf:
                returns[ticker] = perf["annualized_return"]
        
        if returns:
            best_return_ticker = max(returns, key=returns.get)
            summary["best_performers"]["highest_return"] = {
                "ticker": best_return_ticker,
                "return": returns[best_return_ticker]
            }
        
        # Find lowest cost funds
        costs = {}
        for ticker, analysis in valid_analyses.items():
            mgmt = analysis.get("management_metrics", {})
            if "expense_ratio" in mgmt and mgmt["expense_ratio"] is not None:
                costs[ticker] = mgmt["expense_ratio"]
        
        if costs:
            lowest_cost_ticker = min(costs, key=costs.get)
            summary["lowest_cost"] = {
                "ticker": lowest_cost_ticker,
                "expense_ratio": costs[lowest_cost_ticker]
            }
        
        # Find lowest risk funds
        volatilities = {}
        for ticker, analysis in valid_analyses.items():
            perf = analysis.get("performance_metrics", {})
            if "volatility" in perf:
                volatilities[ticker] = perf["volatility"]
        
        if volatilities:
            lowest_risk_ticker = min(volatilities, key=volatilities.get)
            summary["lowest_risk"] = {
                "ticker": lowest_risk_ticker,
                "volatility": volatilities[lowest_risk_ticker]
            }
        
        # Find highest Sharpe ratio
        sharpe_ratios = {}
        for ticker, analysis in valid_analyses.items():
            perf = analysis.get("performance_metrics", {})
            if "sharpe_ratio" in perf:
                sharpe_ratios[ticker] = perf["sharpe_ratio"]
        
        if sharpe_ratios:
            highest_sharpe_ticker = max(sharpe_ratios, key=sharpe_ratios.get)
            summary["highest_sharpe"] = {
                "ticker": highest_sharpe_ticker,
                "sharpe_ratio": sharpe_ratios[highest_sharpe_ticker]
            }
        
        return summary


# Example usage and testing
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = FundAnalyzer()
    
    # Analyze a single fund
    print("Analyzing VUG (Vanguard Growth ETF)...")
    analysis = analyzer.analyze_fund("VUG")
    print(json.dumps(analysis, indent=2, default=str))
    
    # Compare multiple funds
    print("\nComparing multiple funds...")
    comparison = analyzer.compare_funds(["VUG", "VTV", "VBK", "VBR"])
    print(json.dumps(comparison, indent=2, default=str))
