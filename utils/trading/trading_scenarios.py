"""
Trading Scenarios

This file contains hardcoded demo scenarios for testing and demonstration purposes.
Each scenario includes account value, current holdings, and cost basis.

The account_value is calculated as: cash + sum(holdings[ticker] * cost_basis[ticker])
"""

from typing import Dict, Any, List

def _calculate_account_value(cash: float, holdings: Dict[str, float], cost_basis: Dict[str, float]) -> float:
    """Calculate account value as cash + sum of all positions."""
    positions_value = sum(holdings.get(ticker, 0) * cost_basis.get(ticker, 0) for ticker in holdings.keys())
    return cash + positions_value

# Scenario 1: Conservative Retiree
SCENARIO_CONSERVATIVE_RETIREE = {
    "name": "Conservative Retiree",
    "cash": 5000,
    "risk_tolerance": "conservative",
    "tax_sensitivity": "high",
    "holdings": {
        # Format: ticker -> quantity
        "VTV": 300,   # Large Cap Value
        "BND": 1500,  # Bonds (Corporate)
        "VGSH": 400,  # Short Term Treasury
        "VTIP": 200   # TIPS
    },
    "cost_basis": {
        # Format: ticker -> cost per share
        "VTV": 100.0,
        "BND": 75.0,
        "VGSH": 60.0,
        "VTIP": 50.0
    }
}
# Calculate account value
SCENARIO_CONSERVATIVE_RETIREE["account_value"] = _calculate_account_value(
    SCENARIO_CONSERVATIVE_RETIREE["cash"],
    SCENARIO_CONSERVATIVE_RETIREE["holdings"],
    SCENARIO_CONSERVATIVE_RETIREE["cost_basis"]
)

# Scenario 2: Young Professional
SCENARIO_YOUNG_PROFESSIONAL = {
    "name": "Young Professional",
    "cash": 3000,
    "risk_tolerance": "aggressive",
    "tax_sensitivity": "low",
    "holdings": {
        "VUG": 100,   # Large Cap Growth
        "VBK": 50,    # Small Cap Growth
        "VEA": 40     # Developed Market Equity
    },
    "cost_basis": {
        "VUG": 150.0,
        "VBK": 100.0,
        "VEA": 50.0
    }
}
# Calculate account value
SCENARIO_YOUNG_PROFESSIONAL["account_value"] = _calculate_account_value(
    SCENARIO_YOUNG_PROFESSIONAL["cash"],
    SCENARIO_YOUNG_PROFESSIONAL["holdings"],
    SCENARIO_YOUNG_PROFESSIONAL["cost_basis"]
)

# Scenario 3: Mid-Career Balanced
SCENARIO_MID_CAREER_BALANCED = {
    "name": "Mid-Career Balanced",
    "cash": 10000,
    "risk_tolerance": "moderate",
    "tax_sensitivity": "moderate",
    "holdings": {
        "VUG": 200,   # Large Cap Growth
        "VTV": 150,   # Large Cap Value
        "VBK": 100,   # Small Cap Growth
        "VEA": 80,    # Developed Market Equity
        "BND": 400    # Corporate Bonds
    },
    "cost_basis": {
        "VUG": 145.0,
        "VTV": 98.0,
        "VBK": 95.0,
        "VEA": 48.0,
        "BND": 72.0
    }
}
# Calculate account value
SCENARIO_MID_CAREER_BALANCED["account_value"] = _calculate_account_value(
    SCENARIO_MID_CAREER_BALANCED["cash"],
    SCENARIO_MID_CAREER_BALANCED["holdings"],
    SCENARIO_MID_CAREER_BALANCED["cost_basis"]
)

# Scenario 4: High Net Worth Tax-Sensitive
SCENARIO_HIGH_NET_WORTH = {
    "name": "High Net Worth Tax-Sensitive",
    "cash": 100000,
    "risk_tolerance": "moderate",
    "tax_sensitivity": "high",
    "holdings": {
        "VUG": 500,   # Large Cap Growth
        "VTV": 400,   # Large Cap Value
        "VBK": 200,   # Small Cap Growth
        "VB": 150,    # Small Cap Value
        "VEA": 150,   # Developed Market Equity
        "VWO": 100,   # Emerging Market Equity
        "BND": 1500,  # Corporate Bonds
        "VGIT": 300,  # Mid Term Treasury
        "VTIP": 200   # TIPS
    },
    "cost_basis": {
        "VUG": 140.0,
        "VTV": 96.0,
        "VBK": 93.0,
        "VB": 85.0,
        "VEA": 47.0,
        "VWO": 42.0,
        "BND": 74.0,
        "VGIT": 70.0,
        "VTIP": 49.0
    }
}
# Calculate account value
SCENARIO_HIGH_NET_WORTH["account_value"] = _calculate_account_value(
    SCENARIO_HIGH_NET_WORTH["cash"],
    SCENARIO_HIGH_NET_WORTH["holdings"],
    SCENARIO_HIGH_NET_WORTH["cost_basis"]
)

# Scenario 5: New Investor
SCENARIO_NEW_INVESTOR = {
    "name": "New Investor",
    "cash": 500,
    "risk_tolerance": "aggressive",
    "tax_sensitivity": "low",
    "holdings": {
        "VTI": 50     # Total Stock Market (only position)
    },
    "cost_basis": {
        "VTI": 220.0
    }
}
# Calculate account value
SCENARIO_NEW_INVESTOR["account_value"] = _calculate_account_value(
    SCENARIO_NEW_INVESTOR["cash"],
    SCENARIO_NEW_INVESTOR["holdings"],
    SCENARIO_NEW_INVESTOR["cost_basis"]
)

# Scenario 6: Pre-Retirement Transition
SCENARIO_PRE_RETIREMENT = {
    "name": "Pre-Retirement Transition",
    "cash": 50000,
    "risk_tolerance": "conservative",
    "tax_sensitivity": "high",
    "holdings": {
        "VUG": 200,   # Large Cap Growth
        "VTV": 250,   # Large Cap Value
        "VBK": 100,   # Small Cap Growth
        "VB": 80,     # Small Cap Value
        "VEA": 100,   # Developed Market Equity
        "VWO": 60,    # Emerging Market Equity
        "BND": 1200,  # Corporate Bonds
        "VGIT": 400,  # Mid Term Treasury
        "VGIT": 300,  # Long Term Treasury
        "VTIP": 350   # TIPS
    },
    "cost_basis": {
        "VUG": 142.0,
        "VTV": 94.0,
        "VBK": 92.0,
        "VB": 83.0,
        "VEA": 46.0,
        "VWO": 40.0,
        "BND": 73.0,
        "VGIT": 71.0,
        "VTIP": 48.0
    }
}
# Calculate account value
SCENARIO_PRE_RETIREMENT["account_value"] = _calculate_account_value(
    SCENARIO_PRE_RETIREMENT["cash"],
    SCENARIO_PRE_RETIREMENT["holdings"],
    SCENARIO_PRE_RETIREMENT["cost_basis"]
)

# All scenarios as a list
ALL_SCENARIOS = [
    SCENARIO_CONSERVATIVE_RETIREE,
    SCENARIO_YOUNG_PROFESSIONAL,
    SCENARIO_MID_CAREER_BALANCED,
    SCENARIO_HIGH_NET_WORTH,
    SCENARIO_NEW_INVESTOR,
    SCENARIO_PRE_RETIREMENT
]

# Helper function to get scenario by name
def get_scenario_by_name(name: str) -> Dict[str, Any]:
    """Get a scenario by its name."""
    for scenario in ALL_SCENARIOS:
        if scenario["name"] == name:
            return scenario
    return None

# Helper function to get scenario by index
def get_scenario_by_index(index: int) -> Dict[str, Any]:
    """Get a scenario by its index (0-based)."""
    if 0 <= index < len(ALL_SCENARIOS):
        return ALL_SCENARIOS[index]
    return None

# Helper function to format scenario for display
def format_scenario_display(scenario: Dict[str, Any]) -> str:
    """Format a scenario for display in the UI."""
    holdings_count = len(scenario.get("holdings", {}))
    cash = scenario.get("cash", 0)
    holdings = scenario.get("holdings", {})
    cost_basis = scenario.get("cost_basis", {})
    
    # Calculate account value using the helper function
    account_value = _calculate_account_value(cash, holdings, cost_basis)
    
    # Calculate positions value
    positions_value = sum(holdings.get(ticker, 0) * cost_basis.get(ticker, 0) for ticker in holdings.keys())
    
    return (
        f"**{scenario['name']}**\n"
        f"• Total Account Value: ${account_value:,}\n"
        f"  - Cash: ${cash:,}\n"
        f"  - Holdings Value: ${positions_value:,}\n"
        f"• Risk: {scenario['risk_tolerance'].capitalize()} | Tax Sensitivity: {scenario['tax_sensitivity'].capitalize()}\n"
        f"• Current Holdings: {holdings_count} positions"
    )
