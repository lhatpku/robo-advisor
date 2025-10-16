"""
Trading Module

This module provides portfolio rebalancing and trading functionality
with tax-aware optimization and execution management.
"""

from .rebalance import (
    Rebalancer,
    RebalanceConfig,
    Position,
    TaxRates,
    TaxStatus,
    rebalance_portfolio
)

from .portfolio_trading import (
    PortfolioTradingManager,
    generate_trading_requests_from_investment
)

from .trading_agent import TradingAgent

__all__ = [
    'Rebalancer',
    'RebalanceConfig', 
    'Position',
    'TaxRates',
    'TaxStatus',
    'rebalance_portfolio',
    'PortfolioTradingManager',
    'generate_trading_requests_from_investment',
    'TradingAgent'
]
