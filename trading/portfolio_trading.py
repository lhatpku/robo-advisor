"""
Portfolio Trading Integration

This module integrates the enhanced rebalancing functionality with the investment agent
to generate executable trading requests from portfolio optimization results.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from .rebalance import Rebalancer, RebalanceConfig, Position, TaxRates
from investment.fund_analyzer import FundAnalyzer
from .config import (
    get_rebalance_config, 
    get_tax_rates, 
    create_simple_covariance_matrix,
    BASE_VARIANCE,
    AVERAGE_CORRELATION
)


class PortfolioTradingManager:
    """
    Manages the integration between portfolio optimization and trading execution
    """
    
    def __init__(self, rebalance_config: RebalanceConfig = None):
        self.rebalancer = Rebalancer(rebalance_config)
        self.fund_analyzer = FundAnalyzer()
        
    def generate_trading_requests(
        self, 
        investment_portfolio: Dict[str, Any],
        current_positions: List[Dict[str, Any]] = None,
        account_value: float = None
    ) -> Dict[str, Any]:
        """
        Generate trading requests from investment portfolio results
        
        Parameters
        ----------
        investment_portfolio : Dict[str, Any]
            Investment portfolio from investment agent with tickers and weights
        current_positions : List[Dict[str, Any]], optional
            Current account positions for rebalancing
        account_value : float, optional
            Total account value for position sizing
            
        Returns
        -------
        Dict with trading requests and analysis
        """
        # Extract target weights from investment portfolio
        target_weights = self._extract_target_weights(investment_portfolio)
        
        # Check if we have any target weights
        if not target_weights:
            raise ValueError("No target weights found in investment portfolio. Please complete investment selection first.")
        
        if current_positions is None:
            # Create initial positions if none provided
            current_positions = self._create_initial_positions(target_weights, account_value or 100000)
        
        # Convert to Position objects
        positions = self._convert_to_positions(current_positions)
        
        # Check if we have any positions
        if not positions:
            raise ValueError("No positions available for rebalancing. Please use a demo scenario.")
        
        # Set up risk model if covariance data available
        self._setup_risk_model(positions)
        
        # Run enhanced rebalancing
        rebalance_result = self.rebalancer.rebalance_portfolio(positions, target_weights)
        
        # Generate trading requests
        trading_requests = self._generate_trading_requests(rebalance_result, investment_portfolio)
        
        return {
            'trading_requests': trading_requests,
            'rebalance_analysis': rebalance_result,
            'target_weights': target_weights,
            'execution_summary': self._create_execution_summary(trading_requests, rebalance_result)
        }
    
    def _extract_target_weights(self, investment_portfolio: Dict[str, Any]) -> Dict[str, float]:
        """Extract target weights from investment portfolio"""
        target_weights = {}
        
        for asset_class, data in investment_portfolio.items():
            if isinstance(data, dict) and 'weight' in data:
                target_weights[data['ticker']] = data['weight']
        
        return target_weights
    
    def _create_initial_positions(self, target_weights: Dict[str, float], account_value: float) -> List[Dict[str, Any]]:
        """Create initial positions for new accounts"""
        positions = []
        
        for ticker, weight in target_weights.items():
            if ticker == 'sweep_cash':
                # Cash position
                positions.append({
                    'ticker': ticker,
                    'quantity': weight * account_value,  # Cash amount
                    'cost_basis': 1.0,  # Cash has no cost basis
                    'price': 1.0,
                    'acquisition_date': datetime.now() - timedelta(days=400)  # Long-term
                })
            else:
                # Fund position - assume no current holdings
                positions.append({
                    'ticker': ticker,
                    'quantity': 0.0,
                    'cost_basis': 0.0,
                    'price': 100.0,  # Placeholder price
                    'acquisition_date': datetime.now() - timedelta(days=400)
                })
        
        return positions
    
    def _convert_to_positions(self, current_positions: List[Dict[str, Any]]) -> List[Position]:
        """Convert position dictionaries to Position objects"""
        positions = []
        
        for pos in current_positions:
            positions.append(Position(
                ticker=pos['ticker'],
                target_weight=pos.get('target_weight', 0.0),
                quantity=pos['quantity'],
                cost_basis=pos['cost_basis'],
                price=pos['price'],
                acquisition_date=pos.get('acquisition_date', datetime.now() - timedelta(days=400)),
                lot_id=pos.get('lot_id')
            ))
        
        return positions
    
    def _setup_risk_model(self, positions: List[Position]):
        """Set up risk model with covariance matrix using configuration"""
        # This is a simplified implementation using configuration assumptions
        # In practice, you would load actual covariance data
        
        n_assets = len(positions)
        tickers = [p.ticker for p in positions]
        
        # Create covariance matrix using configuration parameters
        covariance_matrix = create_simple_covariance_matrix(
            n_assets=n_assets,
            base_variance=BASE_VARIANCE,
            correlation=AVERAGE_CORRELATION
        )
        
        # Set the risk model
        self.rebalancer.set_risk_model(covariance_matrix)
    
    def _generate_trading_requests(self, rebalance_result: Dict[str, Any], 
                                 investment_portfolio: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate executable trading requests from rebalancing results"""
        trading_requests = []
        
        for trade in rebalance_result['trades']:
            # Get fund analysis data for the ticker
            ticker = trade['ticker']
            fund_analysis = self._get_fund_analysis(ticker, investment_portfolio)
            
            # Create trading request
            trading_request = {
                'ticker': ticker,
                'side': trade['side'],
                'shares': trade['shares'],
                'price': trade.get('price', 0.0),
                'proceeds': trade['proceeds'],
                'realized_gain': trade['realized_gain'],
                'fund_analysis': fund_analysis,
                'execution_priority': self._calculate_execution_priority(trade, fund_analysis),
                'risk_metrics': self._calculate_risk_metrics(trade, fund_analysis),
                'tax_implications': self._calculate_tax_implications(trade)
            }
            
            trading_requests.append(trading_request)
        
        # Sort by execution priority
        trading_requests.sort(key=lambda x: x['execution_priority'], reverse=True)
        
        return trading_requests
    
    def _get_fund_analysis(self, ticker: str, investment_portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """Get fund analysis data for a ticker"""
        if ticker == 'sweep_cash':
            return {
                'ticker': 'sweep_cash',
                'name': 'Cash Sweep Account',
                'type': 'cash',
                'analysis_available': False
            }
        
        # Find the fund in investment portfolio
        for asset_class, data in investment_portfolio.items():
            if isinstance(data, dict) and data.get('ticker') == ticker:
                return data.get('analysis', {})
        
        # If not found, return empty analysis
        return {'analysis_available': False}
    
    def _calculate_execution_priority(self, trade: Dict[str, Any], fund_analysis: Dict[str, Any]) -> float:
        """Calculate execution priority for a trade"""
        priority = 0.0
        
        # Base priority on trade size
        trade_value = abs(trade['proceeds'])
        priority += trade_value * 0.1
        
        # Adjust for fund liquidity (if analysis available)
        if fund_analysis.get('analysis_available', False):
            # Higher priority for more liquid funds
            aum = fund_analysis.get('management_metrics', {}).get('aum', 0)
            if aum > 0:
                priority += min(aum / 1e9, 10.0)  # Cap at 10 for very large funds
        
        # Adjust for tax implications
        if trade['side'] == 'SELL' and trade['realized_gain'] < 0:
            # Higher priority for tax loss harvesting
            priority += abs(trade['realized_gain']) * 0.2
        
        return priority
    
    def _calculate_risk_metrics(self, trade: Dict[str, Any], fund_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate risk metrics for a trade"""
        risk_metrics = {
            'trade_value': abs(trade['proceeds']),
            'volatility': 0.0,
            'beta': 1.0,
            'tracking_error': 0.0
        }
        
        if fund_analysis.get('analysis_available', False):
            perf_metrics = fund_analysis.get('performance_metrics', {})
            risk_metrics.update({
                'volatility': perf_metrics.get('volatility', 0.0),
                'beta': perf_metrics.get('beta', 1.0),
                'tracking_error': perf_metrics.get('tracking_error', 0.0)
            })
        
        return risk_metrics
    
    def _calculate_tax_implications(self, trade: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate tax implications for a trade"""
        if trade['side'] != 'SELL':
            return {'tax_cost': 0.0, 'tax_rate': 0.0, 'net_proceeds': trade['proceeds']}
        
        realized_gain = trade['realized_gain']
        
        if realized_gain <= 0:
            # Loss - can offset gains
            return {
                'tax_cost': 0.0,
                'tax_rate': 0.0,
                'net_proceeds': trade['proceeds'],
                'tax_benefit': abs(realized_gain) * 0.2  # Assume 20% tax benefit
            }
        else:
            # Gain - subject to tax
            # Assume short-term rate for simplicity
            tax_rate = 0.37  # 37% short-term rate
            tax_cost = realized_gain * tax_rate
            
            return {
                'tax_cost': tax_cost,
                'tax_rate': tax_rate,
                'net_proceeds': trade['proceeds'] - tax_cost
            }
    
    def _create_execution_summary(self, trading_requests: List[Dict[str, Any]], 
                                rebalance_result: Dict[str, Any]) -> Dict[str, Any]:
        """Create execution summary for trading requests"""
        total_buy_value = sum(abs(req['proceeds']) for req in trading_requests if req['side'] == 'BUY')
        total_sell_value = sum(req['proceeds'] for req in trading_requests if req['side'] == 'SELL')
        total_tax_cost = sum(req['tax_implications']['tax_cost'] for req in trading_requests)
        
        return {
            'total_trades': len(trading_requests),
            'buy_trades': len([req for req in trading_requests if req['side'] == 'BUY']),
            'sell_trades': len([req for req in trading_requests if req['side'] == 'SELL']),
            'total_buy_value': total_buy_value,
            'total_sell_value': total_sell_value,
            'net_cash_flow': total_sell_value - total_buy_value,
            'total_tax_cost': total_tax_cost,
            'tracking_error': rebalance_result.get('tracking_error', 0.0),
            'execution_priority_high': len([req for req in trading_requests if req['execution_priority'] > 5.0])
        }


def generate_trading_requests_from_investment(
    investment_portfolio: Dict[str, Any],
    current_positions: List[Dict[str, Any]] = None,
    account_value: float = None,
    rebalance_config: RebalanceConfig = None
) -> Dict[str, Any]:
    """
    Convenience function to generate trading requests from investment portfolio
    
    Parameters
    ----------
    investment_portfolio : Dict[str, Any]
        Investment portfolio from investment agent
    current_positions : List[Dict[str, Any]], optional
        Current account positions
    account_value : float, optional
        Total account value
    rebalance_config : RebalanceConfig, optional
        Rebalancing configuration
        
    Returns
    -------
    Dict with trading requests and analysis
    """
    manager = PortfolioTradingManager(rebalance_config)
    return manager.generate_trading_requests(investment_portfolio, current_positions, account_value)
