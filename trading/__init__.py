"""
Trading Module

This module provides portfolio rebalancing and trading functionality
with tax-aware optimization and execution management.
"""

from .rebalance import (
    SoftObjectiveRebalancer
)

from .trading_agent import TradingAgent

__all__ = [
    'SoftObjectiveRebalancer',
    'TradingAgent'
]
