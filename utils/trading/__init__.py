"""
Trading Module

This module provides portfolio rebalancing and trading functionality
with tax-aware optimization and execution management.
"""

from .rebalance import (
    SoftObjectiveRebalancer
)

__all__ = [
    'SoftObjectiveRebalancer',
]
