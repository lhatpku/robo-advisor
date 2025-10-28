"""
Trading Agent Utilities

This module contains all the utility functions used by the trading agent,
separated from the main agent logic for better organization and reusability.
"""

from typing import Dict, Any, List, Optional
import numpy as np
from langchain_openai import ChatOpenAI
from state import AgentState
from trading.rebalance import SoftObjectiveRebalancer
from trading.config import DEFAULT_REBALANCE_CONFIG, COVARIANCE_MATRIX_DATA, ASSET_ORDER
from trading.trading_scenarios import ALL_SCENARIOS, get_scenario_by_index
from prompts.trading_prompts import TradingMessages


class TradingUtils:
    """Utility class containing all trading-related helper functions."""
    
    def __init__(self, llm: ChatOpenAI):
        """Initialize the TradingUtils."""
        self.llm = llm
    
    def show_scenario_selection(self, state: AgentState) -> AgentState:
        """Show scenario selection options."""
        from trading.trading_scenarios import _calculate_account_value
        
        scenarios_text = "\n\n".join([
            f"{i+1}. {scenario['name']}\n"
            f"   • Total Account Value: ${_calculate_account_value(scenario.get('cash', 0), scenario.get('holdings', {}), scenario.get('cost_basis', {})):,}\n"
            f"   • Risk: {scenario['risk_tolerance'].capitalize()} | Tax Sensitivity: {scenario['tax_sensitivity'].capitalize()}\n"
            f"   • Current Holdings: {len(scenario.get('holdings', {}))} positions"
            for i, scenario in enumerate(ALL_SCENARIOS)
        ])
        
        message = (
            "I need some information about your current portfolio to generate accurate trading requests.\n\n"
            "**Available Demo Scenarios:**\n\n"
            f"{scenarios_text}\n\n"
            "📋 **How to proceed:**\n"
            "• Type a number (1-6) to select a scenario\n"
            "• Type 'custom' to use your actual portfolio (requires additional input)\n\n"
            "⚠️ **Note:** Demo scenarios are recommended for testing."
        )
        
        state["messages"].append({"role": "ai", "content": message})
        return state
    
    def build_positions_from_investment(self, investment: Dict[str, Any], selected_scenario: Optional[Dict[str, Any]]) -> List[Dict[str, float]]:
        """
        Build positions list for rebalancer from investment agent output.
        
        Args:
            investment: Investment portfolio from investment agent (includes ticker, weight, price)
            selected_scenario: Selected trading scenario (includes holdings, cost_basis, cash, account_value)
            
        Returns:
            List of position dictionaries for rebalancer
        """
        positions = []
        
        # If we have a selected scenario, use it for holdings and cost basis
        if selected_scenario:
            holdings = selected_scenario.get('holdings', {})
            cost_basis = selected_scenario.get('cost_basis', {})
            cash = selected_scenario.get('cash', 0)
        else:
            # New investor scenario (no existing holdings)
            holdings = {}
            cost_basis = {}
            cash = 2000  # default cash
        
        for asset_class, data in investment.items():
            if asset_class == "cash":
                # Cash position
                positions.append({
                    "ticker": "CASH",
                    "target_weight": data["weight"],
                    "quantity": cash,
                    "cost_basis": 1.0,
                    "price": 1.0
                })
            else:
                ticker = data["ticker"]
                target_weight = data["weight"]
                price = data.get("price", 100.0)  # fallback if no price
                
                # Get current holdings from scenario
                quantity = holdings.get(ticker, 0)
                
                # Get cost basis from scenario (or use price if not available)
                cost_basis_price = cost_basis.get(ticker, price)
                
                positions.append({
                    "ticker": ticker,
                    "target_weight": target_weight,
                    "quantity": quantity,
                    "cost_basis": cost_basis_price,
                    "price": price
                })
        
        return positions
    
    def build_covariance_subset(self, investment: Dict[str, Any]) -> np.ndarray:
        """
        Build covariance matrix subset for selected tickers.
        
        Args:
            investment: Investment portfolio from investment agent
            
        Returns:
            Covariance matrix subset as numpy array
        """
        # Get all tickers (excluding cash)
        tickers = [data["ticker"] for asset_class, data in investment.items() if asset_class != "cash"]
        
        # Create a simple diagonal covariance matrix (simplified approach)
        n = len(tickers)
        cov_subset = np.eye(n) * 0.04  # 4% variance for all assets
        
        return cov_subset
    
    def execute_rebalancing(
        self,
        state: AgentState,
        investment: Dict[str, Any],
        tax_weight: float,
        ltcg_rate: float,
        integer_shares: bool,
        selected_scenario: Optional[Dict[str, Any]]
    ) -> AgentState:
        """
        Execute portfolio rebalancing using SoftObjectiveRebalancer.
        
        Args:
            state: Current agent state
            investment: Investment portfolio from investment agent
            tax_weight: Tax weight parameter
            ltcg_rate: Long-term capital gains rate
            integer_shares: Whether to use integer shares only
            selected_scenario: Selected trading scenario
            
        Returns:
            Updated agent state with trading requests
        """
        try:
            state["messages"].append({
                "role": "ai",
                "content": TradingMessages.rebalancing_in_progress()
            })
            
            # Build positions from investment
            positions = self.build_positions_from_investment(investment, selected_scenario)
            
            # Build covariance matrix subset
            cov_matrix = self.build_covariance_subset(investment)
            
            # Initialize rebalancer
            rebalancer = SoftObjectiveRebalancer(
                cov_matrix=cov_matrix,
                tax_weight=tax_weight,
                ltcg_rate=ltcg_rate,
                integer_shares=integer_shares,
                min_cash_pct=DEFAULT_REBALANCE_CONFIG.get('min_cash_pct', 0.02)
            )
            
            # Execute rebalancing
            result = rebalancer.rebalance(positions)
            
            # Store in state with the format expected by reviewer
            state["trading_requests"] = {
                "trading_requests": result.get('trades', []),
                "summary": result
            }
            
            # Format trades summary
            trades = result.get('trades', [])
            trades_summary = self.format_trades_summary(trades)
            
            # Show success message
            state["messages"].append({
                "role": "ai",
                "content": TradingMessages.rebalancing_success(trades_summary, result)
            })
            
        except Exception as e:
            print(f"Rebalancing error: {e}")
            state["messages"].append({
                "role": "ai",
                "content": TradingMessages.rebalancing_failed()
            })
        
        return state
    
    def format_trades_summary(self, trades: List[Dict[str, Any]]) -> str:
        """
        Format trades for display.
        
        Args:
            trades: List of trade dictionaries
            
        Returns:
            Formatted trades summary string
        """
        if not trades:
            return "No trades required."
        
        lines = ["**Trading Requests:**\n"]
        for i, trade in enumerate(trades[:10], 1):  # Show first 10
            shares = trade.get('Shares', 0)
            ticker = trade.get('Ticker', '')
            price = trade.get('Price', 0)
            side = trade.get('Side', '')
            
            lines.append(f"{i}. {side} {shares:.4f} {ticker} @ ${price:.2f}")
        
        if len(trades) > 10:
            lines.append(f"\n... and {len(trades) - 10} more trades")
        
        return "\n".join(lines)

