"""
Investment Module Configuration

This file contains all the fund selection data and configuration parameters
for the investment module. This centralizes all fund options and criteria
to ensure consistency across the investment agent and fund analysis tools.

Author: Robo-Advisor System
Date: 2024
"""

# =============================================================================
# ASSET CLASS FUND MAPPINGS
# =============================================================================

# Mapping of asset classes to available fund tickers
# Each asset class has multiple fund options for selection
ASSET_CLASS_FUNDS = {
    "large_cap_growth": ["VUG", "MGK", "VUG", "QQQ"],
    "large_cap_value": ["VTV", "VYM", "VTV", "SPYV"],
    "small_cap_growth": ["VBK", "IJR", "VBK", "IJS"],
    "small_cap_value": ["VBR", "IJS", "VBR", "SLYV"],
    "developed_market_equity": ["VEA", "EFA", "VEA", "IEFA"],
    "emerging_market_equity": ["VWO", "EEM", "VWO", "IEMG"],
    "short_term_treasury": ["SHY", "VGSH", "SHY", "SCHR"],
    "mid_term_treasury": ["IEF", "VGIT", "IEF", "SCHM"],
    "long_term_treasury": ["TLT", "VGLT", "TLT", "SCHQ"],
    "corporate_bond": ["LQD", "VCIT", "LQD", "SCHI"],
    "tips": ["TIP", "VTEB", "TIP", "SCHP"],
    "cash": ["BIL", "SHV", "BIL", "SCHO"]
}

# =============================================================================
# FUND SELECTION CRITERIA
# =============================================================================

# Available selection criteria with descriptions
SELECTION_CRITERIA = {
    "balanced": {
        "name": "Balanced",
        "description": "Best risk-adjusted returns (highest Sharpe ratio)",
        "metric": "sharpe_ratio",
        "direction": "highest"
    },
    "low_cost": {
        "name": "Low Cost", 
        "description": "Lowest expense ratio for cost efficiency",
        "metric": "expense_ratio",
        "direction": "lowest"
    },
    "high_performance": {
        "name": "High Performance",
        "description": "Highest historical returns",
        "metric": "annualized_return", 
        "direction": "highest"
    },
    "low_risk": {
        "name": "Low Risk",
        "description": "Lowest volatility for stability",
        "metric": "volatility",
        "direction": "lowest"
    }
}

# =============================================================================
# FUND ANALYSIS CONFIGURATION
# =============================================================================

# Parameters for fund analysis and selection
FUND_ANALYSIS_CONFIG = {
    "min_data_quality_score": 0.7,    # Minimum data quality score for fund analysis
    "max_expense_ratio": 0.01,        # Maximum expense ratio (1%) for fund selection
    "min_aum": 100000000,             # Minimum AUM ($100M) for fund selection
    "max_tracking_error": 0.05,       # Maximum tracking error (5%) for fund selection
    
    "performance_lookback_days": 252,  # Days to look back for performance analysis
    "volatility_lookback_days": 252,   # Days to look back for volatility calculation
    "correlation_lookback_days": 252,  # Days to look back for correlation calculation
    
    "benchmark_ticker": "SPY",        # Benchmark for performance comparison
    "risk_free_rate": 0.02,           # Risk-free rate (2% annual)
    "market_return": 0.10             # Expected market return (10% annual)
}

# =============================================================================
# CASH POSITION CONFIGURATION
# =============================================================================

# Special handling for cash positions
CASH_CONFIG = {
    "ticker": "sweep_cash",
    "description": "Cash position - using sweep account for trading reserve",
    "exclude_from_analysis": True,  # Don't analyze cash positions
    "selection_reason": "Cash position - using sweep account for trading reserve",
    "criteria_used": "cash_reserve"
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_fund_options(asset_class: str) -> list:
    """Get available fund options for an asset class"""
    return ASSET_CLASS_FUNDS.get(asset_class, [])

def get_selection_criteria(criteria: str) -> dict:
    """Get selection criteria configuration"""
    return SELECTION_CRITERIA.get(criteria, {})

def is_cash_position(asset_class: str) -> bool:
    """Check if an asset class is a cash position"""
    return asset_class.lower() == "cash"

def get_cash_config() -> dict:
    """Get cash position configuration"""
    return CASH_CONFIG

# =============================================================================
# CONFIGURATION NOTES
# =============================================================================

CONFIGURATION_NOTES = """
INVESTMENT MODULE CONFIGURATION
===============================

This configuration file centralizes all fund selection data and criteria
for the investment module. This ensures consistency across all components.

ASSET CLASS FUND MAPPINGS:
- Each asset class has 4 fund options for selection
- Options are ordered by preference (first option is typically preferred)
- All tickers are valid and tradeable on major exchanges
- Mix of Vanguard, iShares, and other major providers

SELECTION CRITERIA:
- Balanced: Uses Sharpe ratio for risk-adjusted returns
- Low Cost: Uses expense ratio for cost efficiency  
- High Performance: Uses annualized return for growth
- Low Risk: Uses volatility for stability

FUND ANALYSIS CONFIGURATION:
- Conservative thresholds for fund selection
- Performance metrics based on standard industry practices
- Data quality requirements for reliable analysis

CASH POSITION CONFIGURATION:
- Special handling for cash positions
- Uses sweep account instead of fund analysis
- Excluded from fund selection process

RECOMMENDATIONS FOR PRODUCTION:
1. Update fund options based on current market availability
2. Adjust selection criteria based on user preferences
3. Implement real-time fund data validation
4. Add ESG and sustainability criteria
5. Consider fund family preferences
6. Add minimum investment requirements
7. Implement fund closure monitoring
8. Add performance attribution analysis
9. Consider tax efficiency for different account types
10. Add currency hedging for international funds
"""

if __name__ == "__main__":
    print(CONFIGURATION_NOTES)
