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

# =============================================================================
# CONFIGURATION TOGGLES
# =============================================================================

# Toggle for using assumed data vs real data
USE_ASSUMED_DATA = True  # Set to False when implementing real market data

# =============================================================================
# RISK MODEL ASSUMPTIONS
# =============================================================================

# Base variance assumption for all assets (4% annual variance)
# This is a simplified assumption - in reality, different asset classes have different volatilities
BASE_VARIANCE = 0.04

# Average correlation between assets (30%)
# This is a simplified assumption - in reality, correlations vary significantly
# between asset classes and change over time
AVERAGE_CORRELATION = 0.3

# =============================================================================
# ASSET CLASS RISK CHARACTERISTICS
# =============================================================================

# Asset class risk profiles (variance and beta assumptions)
# These are simplified assumptions based on typical asset class characteristics
ASSET_CLASS_RISK = {
    # Equity Asset Classes
    'large_cap_growth': {
        'variance': 0.16,  # 16% annual variance (4% monthly)
        'beta': 1.2,       # 20% more volatile than market
        'description': 'Large cap growth stocks - higher volatility, growth-oriented'
    },
    'large_cap_value': {
        'variance': 0.12,  # 12% annual variance (3.5% monthly)
        'beta': 0.9,       # 10% less volatile than market
        'description': 'Large cap value stocks - moderate volatility, value-oriented'
    },
    'small_cap_growth': {
        'variance': 0.25,  # 25% annual variance (5% monthly)
        'beta': 1.4,       # 40% more volatile than market
        'description': 'Small cap growth stocks - high volatility, growth-oriented'
    },
    'small_cap_value': {
        'variance': 0.20,  # 20% annual variance (4.5% monthly)
        'beta': 1.1,       # 10% more volatile than market
        'description': 'Small cap value stocks - high volatility, value-oriented'
    },
    'emerging_market_equity': {
        'variance': 0.30,  # 30% annual variance (5.5% monthly)
        'beta': 1.3,       # 30% more volatile than market
        'description': 'Emerging market equities - very high volatility, international exposure'
    },
    'developed_market_equity': {
        'variance': 0.18,  # 18% annual variance (4.2% monthly)
        'beta': 1.0,       # Market volatility
        'description': 'Developed market equities - moderate volatility, international exposure'
    },
    
    # Fixed Income Asset Classes
    'mid_term_treasury': {
        'variance': 0.01,  # 1% annual variance (0.3% monthly)
        'beta': 0.0,       # No market correlation
        'description': 'Mid-term treasury bonds - low volatility, interest rate sensitive'
    },
    'long_term_treasury': {
        'variance': 0.04,  # 4% annual variance (0.6% monthly)
        'beta': 0.0,       # No market correlation
        'description': 'Long-term treasury bonds - moderate volatility, highly interest rate sensitive'
    },
    'short_term_treasury': {
        'variance': 0.001, # 0.1% annual variance (0.1% monthly)
        'beta': 0.0,       # No market correlation
        'description': 'Short-term treasury bonds - very low volatility, minimal interest rate sensitivity'
    },
    'tips': {
        'variance': 0.02,  # 2% annual variance (0.4% monthly)
        'beta': 0.0,       # No market correlation
        'description': 'TIPS (Treasury Inflation-Protected Securities) - low volatility, inflation protection'
    },
    'corporate_bond': {
        'variance': 0.03,  # 3% annual variance (0.5% monthly)
        'beta': 0.2,       # Slight market correlation
        'description': 'Corporate bonds - low volatility, credit risk exposure'
    },
    
    # Cash and Alternatives
    'sweep_cash': {
        'variance': 0.0,   # 0% variance (risk-free)
        'beta': 0.0,       # No market correlation
        'description': 'Cash sweep account - risk-free, no volatility'
    }
}

# =============================================================================
# REBALANCING CONFIGURATION
# =============================================================================

# Default rebalancing configuration parameters
DEFAULT_REBALANCE_CONFIG = {
    # Risk model parameters
    'risk_aversion': 1.0,              # Risk aversion coefficient
    'tracking_error_weight': 1.0,      # Weight for tracking error in objective function
    'tax_penalty_weight': 0.5,         # Weight for tax penalty in objective function
    'friction_weight': 0.1,            # Weight for transaction costs in objective function
    'cash_band_penalty_weight': 0.2,   # Weight for cash band penalty in objective function
    
    # Tax parameters
    'soft_tax_cap': 10000.0,           # Soft tax cap in dollars
    'tax_penalty_exponent': 2.0,       # Exponent for increasing penalty
    
    # Cash management
    'cash_sweep_band_min': 0.02,       # Minimum cash weight (2%)
    'cash_sweep_band_max': 0.05,       # Maximum cash weight (5%)
    
    # Trading parameters
    'spread_cost_bps': 5.0,            # Bid-ask spread cost in basis points
    'turnover_penalty_bps': 2.0,       # Turnover penalty in basis points
    'integer_shares': False,           # Whether to use integer shares only
    
    # Optimization parameters
    'max_iterations': 1000,            # Maximum optimization iterations
    'tolerance': 1e-6,                 # Optimization tolerance
}

# =============================================================================
# TAX RATE ASSUMPTIONS
# =============================================================================

# Tax rates for different holding periods and income levels
# These are simplified assumptions - in reality, tax rates vary by jurisdiction and income
TAX_RATES = {
    'short_term_capital_gains': 0.22,  # 22% for short-term gains (< 1 year)
    'long_term_capital_gains': 0.15,  # 15% for long-term gains (>= 1 year)
    'short_term_capital_losses': 0.22, # 22% for short-term losses (< 1 year)
    'long_term_capital_losses': 0.15,  # 15% for long-term losses (>= 1 year)
    'ordinary_income': 0.25,           # 25% for ordinary income
    'state_tax_rate': 0.05,            # 5% state tax rate
    'medicare_surtax': 0.038           # 3.8% Medicare surtax on investment income
}

# =============================================================================
# DEMO SCENARIO CONFIGURATION
# =============================================================================

# Default parameters for demo scenarios
DEMO_SCENARIO_DEFAULTS = {
    'default_account_value': 100000,   # Default account value for new scenarios
    'default_risk_tolerance': 'moderate',  # Default risk tolerance
    'default_tax_sensitivity': 'neutral',  # Default tax sensitivity
    'default_integer_shares': True,    # Default to integer shares
    'default_cash_reserve': 0.05       # Default 5% cash reserve
}

# =============================================================================
# FUND ANALYSIS CONFIGURATION
# =============================================================================

# Parameters for fund analysis and selection
FUND_ANALYSIS_CONFIG = {
    'min_data_quality_score': 0.7,    # Minimum data quality score for fund analysis
    'max_expense_ratio': 0.01,        # Maximum expense ratio (1%) for fund selection
    'min_aum': 100000000,             # Minimum AUM ($100M) for fund selection
    'max_tracking_error': 0.05,       # Maximum tracking error (5%) for fund selection
    
    'performance_lookback_days': 252,  # Days to look back for performance analysis
    'volatility_lookback_days': 252,   # Days to look back for volatility calculation
    'correlation_lookback_days': 252,  # Days to look back for correlation calculation
    
    'benchmark_ticker': 'SPY',        # Benchmark for performance comparison
    'risk_free_rate': 0.02,           # Risk-free rate (2% annual)
    'market_return': 0.10             # Expected market return (10% annual)
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_asset_class_risk(asset_class: str) -> Dict[str, Any]:
    """Get risk characteristics for an asset class"""
    return ASSET_CLASS_RISK.get(asset_class, {
        'variance': BASE_VARIANCE,
        'beta': 1.0,
        'description': 'Unknown asset class - using default assumptions'
    })

def get_rebalance_config(**overrides) -> Dict[str, Any]:
    """Get rebalancing configuration with optional overrides"""
    config = DEFAULT_REBALANCE_CONFIG.copy()
    config.update(overrides)
    return config

def get_tax_rates(**overrides) -> Dict[str, float]:
    """Get tax rates with optional overrides"""
    rates = TAX_RATES.copy()
    rates.update(overrides)
    return rates

def create_simple_covariance_matrix(n_assets: int, 
                                  base_variance: float = BASE_VARIANCE,
                                  correlation: float = AVERAGE_CORRELATION) -> np.ndarray:
    """Create a simple covariance matrix with uniform assumptions"""
    if not USE_ASSUMED_DATA:
        raise NotImplementedError("Real market data implementation not yet available. Set USE_ASSUMED_DATA=True for demo purposes.")
    
    covariance_matrix = np.zeros((n_assets, n_assets))
    
    for i in range(n_assets):
        for j in range(n_assets):
            if i == j:
                covariance_matrix[i, j] = base_variance
            else:
                covariance_matrix[i, j] = base_variance * correlation
    
    return covariance_matrix

# =============================================================================
# CONFIGURATION NOTES
# =============================================================================

CONFIGURATION_NOTES = """
CONFIGURATION ASSUMPTIONS AND NOTES
===================================

This configuration file contains simplified assumptions for demo purposes.
In a production system, these should be replaced with real market data.

RISK MODEL ASSUMPTIONS:
- All assets assumed to have 4% annual variance (simplified)
- All asset pairs assumed to have 30% correlation (simplified)
- No time-varying volatility or correlation
- No regime changes or market stress scenarios

ASSET CLASS RISK CHARACTERISTICS:
- Based on typical historical asset class behavior
- Variance estimates from long-term historical data
- Beta estimates relative to S&P 500
- No consideration of current market conditions

REBALANCING CONFIGURATION:
- Weights chosen for balanced optimization
- Transaction costs based on typical retail trading
- Cash band designed for practical trading
- Tax assumptions based on US tax code

TAX RATE ASSUMPTIONS:
- Based on US federal tax rates (2024)
- Simplified to single rates (no progressive brackets)
- No consideration of state-specific rates
- No consideration of tax-loss harvesting benefits

DEMO SCENARIO DEFAULTS:
- Conservative assumptions for demo purposes
- Account values chosen for realistic scenarios
- Risk tolerances simplified to three levels

FUND ANALYSIS CONFIGURATION:
- Conservative thresholds for fund selection
- Performance metrics based on standard industry practices
- Data quality requirements for reliable analysis

RECOMMENDATIONS FOR PRODUCTION:
1. Replace synthetic covariance matrix with real market data
2. Implement dynamic risk models that update with market conditions
3. Use actual tax rates based on user's jurisdiction and income
4. Implement real-time fund data feeds
5. Add stress testing and scenario analysis
6. Consider transaction cost models based on actual broker data
7. Implement tax-loss harvesting optimization
8. Add ESG and sustainability criteria
9. Implement factor models for better risk attribution
10. Add currency hedging for international assets
"""

if __name__ == "__main__":
    print(CONFIGURATION_NOTES)
