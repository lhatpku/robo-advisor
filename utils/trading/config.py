"""
Trading Module Configuration

This file contains all the assumed parameters and data used in the trading module.
These are simplified assumptions for demo purposes and should be replaced with 
real market data in production.

Author: Robo-Advisor System
Date: 2024
"""

import numpy as np
from typing import Dict, Any
import sys
import os

# Import portfolio config
from utils.portfolio.config import COVARIANCE_MATRIX_DATA, EXPECTED_RETURNS, ASSET_CLASSES, ASSET_ORDER

# =============================================================================
# REBALANCING CONFIGURATION
# =============================================================================

# Default rebalancing configuration parameters
DEFAULT_REBALANCE_CONFIG = {

    'ltcg_rate': 0.15,                 # Long-term capital gains tax rate (15%)
    'stcg_rate': 0.22,                 # Short-term capital gains tax rate (22%)
    'tax_weight': 1.0,                 # Weight for tax penalty in objective function
    

    'min_cash_pct': 0.02,              # Minimum cash % of total portfolio (2%)

}


