"""
Portfolio Module Configuration

This file contains all the asset statistics and configuration parameters
for the portfolio module. This replaces the Excel file dependencies
and centralizes all portfolio-related data.

Author: Robo-Advisor System
Date: 2024
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

# =============================================================================
# ASSET CLASS DEFINITIONS
# =============================================================================

# Asset class names and their standard representations
ASSET_CLASSES = [
    "large_cap_growth",
    "large_cap_value", 
    "small_cap_growth",
    "small_cap_value",
    "developed_market_equity",
    "emerging_market_equity",
    "short_term_treasury",
    "mid_term_treasury",
    "long_term_treasury",
    "corporate_bond",
    "tips",
    "cash"
]

# Asset class display names for user interface
ASSET_CLASS_DISPLAY_NAMES = {
    "large_cap_growth": "Large Cap Growth",
    "large_cap_value": "Large Cap Value",
    "small_cap_growth": "Small Cap Growth", 
    "small_cap_value": "Small Cap Value",
    "developed_market_equity": "Developed Market Equity",
    "emerging_market_equity": "Emerging Market Equity",
    "short_term_treasury": "Short Term Treasury",
    "mid_term_treasury": "Mid Term Treasury",
    "long_term_treasury": "Long Term Treasury",
    "corporate_bond": "Corporate Bond",
    "tips": "TIPS",
    "cash": "Cash"
}

# =============================================================================
# EXPECTED RETURNS (MU) - From Excel file
# =============================================================================

# Expected annual returns for each asset class (exact from asset_stats.xlsx)
EXPECTED_RETURNS = {
    "large_cap_growth": 0.075,     # 7.5% - From Excel
    "large_cap_value": 0.070,      # 7.0% - From Excel
    "small_cap_growth": 0.085,     # 8.5% - From Excel
    "small_cap_value": 0.080,      # 8.0% - From Excel
    "developed_market_equity": 0.070,  # 7.0% - From Excel
    "emerging_market_equity": 0.090,  # 9.0% - From Excel
    "short_term_treasury": 0.035,  # 3.5% - From Excel
    "mid_term_treasury": 0.045,    # 4.5% - From Excel
    "long_term_treasury": 0.047,   # 4.7% - From Excel
    "corporate_bond": 0.050,       # 5.0% - From Excel
    "tips": 0.045,                 # 4.5% - From Excel
    "cash": 0.020                  # 2.0% - From Excel
}

# =============================================================================
# COVARIANCE MATRIX - From Excel file
# =============================================================================

# Covariance matrix from asset_stats.xlsx (12x12 matrix)
# This is the exact covariance matrix from the Excel file
COVARIANCE_MATRIX_DATA = np.array([
    [0.036100, 0.024320, 0.036480, 0.033440, 0.027360, 0.033440, -0.000855, -0.001425, -0.002565, -0.001995, -0.001425, -0.000048],
    [0.024320, 0.025600, 0.030720, 0.028160, 0.023040, 0.028160, -0.000720, -0.001200, -0.002160, -0.001680, -0.001200, -0.000040],
    [0.036480, 0.030720, 0.057600, 0.042240, 0.034560, 0.042240, -0.001080, -0.001800, -0.003240, -0.002520, -0.001800, -0.000060],
    [0.033440, 0.028160, 0.042240, 0.048400, 0.031680, 0.038720, -0.000990, -0.001650, -0.002970, -0.002310, -0.001650, -0.000055],
    [0.027360, 0.023040, 0.034560, 0.031680, 0.032400, 0.031680, -0.000810, -0.001350, -0.002430, -0.001890, -0.001350, -0.000045],
    [0.033440, 0.028160, 0.042240, 0.038720, 0.031680, 0.048400, -0.000990, -0.001650, -0.002970, -0.002310, -0.001650, -0.000055],
    [-0.000855, -0.000720, -0.001080, -0.000990, -0.000810, -0.000990, 0.000900, 0.000900, 0.001620, 0.001260, 0.000900, 0.000015],
    [-0.001425, -0.001200, -0.001800, -0.001650, -0.001350, -0.001650, 0.000900, 0.002500, 0.002700, 0.002100, 0.001500, 0.000025],
    [-0.002565, -0.002160, -0.003240, -0.002970, -0.002430, -0.002970, 0.001620, 0.002700, 0.008100, 0.003780, 0.002700, 0.000045],
    [-0.001995, -0.001680, -0.002520, -0.002310, -0.001890, -0.002310, 0.001260, 0.002100, 0.003780, 0.004900, 0.002100, 0.000035],
    [-0.001425, -0.001200, -0.001800, -0.001650, -0.001350, -0.001650, 0.000900, 0.001500, 0.002700, 0.002100, 0.002500, 0.000025],
    [-0.000048, -0.000040, -0.000060, -0.000055, -0.000045, -0.000055, 0.000015, 0.000025, 0.000045, 0.000035, 0.000025, 0.000025]
])

# Asset class order (matches the covariance matrix rows/columns)
ASSET_ORDER = [
    "large_cap_growth",
    "large_cap_value", 
    "small_cap_growth",
    "small_cap_value",
    "developed_market_equity",
    "emerging_market_equity",
    "short_term_treasury",
    "mid_term_treasury",
    "long_term_treasury",
    "corporate_bond",
    "tips",
    "cash"
]

# =============================================================================
# PORTFOLIO OPTIMIZATION PARAMETERS
# =============================================================================

# Default values for portfolio optimization
DEFAULT_LAMBDA = 1.0               # Risk aversion parameter (higher = more conservative)
DEFAULT_CASH_RESERVE = 0.05        # 5% cash reserve

# Cash reserve constraints
CASH_RESERVE_MIN = 0.02            # Minimum 2% cash reserve
CASH_RESERVE_MAX = 0.05            # Maximum 5% cash reserve

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_expected_returns() -> np.ndarray:
    """Get expected returns as numpy array in asset class order."""
    return np.array([EXPECTED_RETURNS[asset] for asset in ASSET_ORDER])

def get_covariance_matrix() -> np.ndarray:
    """Get the exact covariance matrix from Excel file."""
    return COVARIANCE_MATRIX_DATA.copy()

def get_standard_deviations() -> np.ndarray:
    """Get standard deviations derived from diagonal of covariance matrix."""
    return np.sqrt(np.diag(COVARIANCE_MATRIX_DATA))

def get_asset_class_index(asset_class: str) -> int:
    """Get the index of an asset class in the standard ordering."""
    try:
        return ASSET_ORDER.index(asset_class)
    except ValueError:
        raise ValueError(f"Unknown asset class: {asset_class}")

def get_asset_class_name(index: int) -> str:
    """Get the asset class name for a given index."""
    if 0 <= index < len(ASSET_ORDER):
        return ASSET_ORDER[index]
    else:
        raise IndexError(f"Asset class index {index} out of range")

def get_display_name(asset_class: str) -> str:
    """Get the display name for an asset class."""
    return ASSET_CLASS_DISPLAY_NAMES.get(asset_class, asset_class.replace("_", " ").title())

def get_cash_reserve_constraints() -> Tuple[float, float]:
    """
    Get cash reserve constraints.
    
    Returns:
        Tuple of (min_cash_reserve, max_cash_reserve)
    """
    return CASH_RESERVE_MIN, CASH_RESERVE_MAX

def validate_cash_reserve(cash_reserve: float) -> bool:
    """
    Validate if cash reserve is within allowed range.
    
    Args:
        cash_reserve: Cash reserve value to validate
        
    Returns:
        True if valid, False otherwise
    """
    return CASH_RESERVE_MIN <= cash_reserve <= CASH_RESERVE_MAX

def create_asset_stats_dataframe() -> pd.DataFrame:
    """Create a DataFrame with asset statistics for compatibility."""
    data = []
    std_devs = get_standard_deviations()
    
    for i, asset in enumerate(ASSET_ORDER):
        data.append({
            "Asset Class": get_display_name(asset),
            "Expected Return": EXPECTED_RETURNS[asset],
            "Volatility": std_devs[i],
            "Risk Level": "High" if std_devs[i] > 0.15 else "Medium" if std_devs[i] > 0.05 else "Low"
        })
    
    return pd.DataFrame(data)

# =============================================================================
# CONFIGURATION NOTES
# =============================================================================

CONFIGURATION_NOTES = """
PORTFOLIO MODULE CONFIGURATION
==============================

This configuration file centralizes all portfolio optimization data
and parameters, replacing the Excel file dependencies.

EXPECTED RETURNS:
- Based on historical averages and forward-looking estimates
- Conservative estimates for long-term planning
- Updated annually or as market conditions change

CORRELATION MATRIX:
- Based on historical correlation analysis
- Simplified for computational efficiency
- Assumes stable relationships over time

VOLATILITY ESTIMATES:
- Annualized standard deviation of returns
- Based on historical data and market expectations
- Conservative estimates for risk management

OPTIMIZATION PARAMETERS:
- Risk aversion (lambda): Higher = more conservative
- Cash reserve: Minimum cash allocation
- Weight constraints: Maximum allocation per asset class
- Rebalancing: Thresholds and turnover limits

RISK TOLERANCE MAPPINGS:
- Conservative: High lambda, low max weights, high cash
- Moderate: Balanced parameters
- Aggressive: Low lambda, high max weights, low cash

RECOMMENDATIONS FOR PRODUCTION:
1. Update expected returns quarterly
2. Recalibrate correlations annually
3. Adjust volatility estimates based on market regime
4. Implement regime detection for dynamic parameters
5. Add transaction cost modeling
6. Include liquidity constraints
7. Add ESG screening parameters
8. Implement factor-based risk models
9. Add currency hedging considerations
10. Include tax optimization parameters
"""

if __name__ == "__main__":
    print(CONFIGURATION_NOTES)
    print("\nAsset Statistics:")
    print(create_asset_stats_dataframe())
