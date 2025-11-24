"""
Risk Module Configuration

This file contains all the risk assessment questions, glidepath data,
and portfolio index mappings for the risk module. This replaces the
Excel file dependencies and centralizes all risk-related data.

Author: Robo-Advisor System
Date: 2024
"""

import pandas as pd
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

# =============================================================================
# RISK ASSESSMENT QUESTIONS (Exact from RiskManager)
# =============================================================================

@dataclass
class MCQuestion:
    id: str
    text: str
    label: str
    options: List[str]
    guidance: str

# Exact questions from RiskManager._load_questions()
RISK_QUESTIONS = [
    MCQuestion(
        id="q1",
        text="How much emergency savings do you currently have set aside?",
        label="Emergency Savings",
        options=[
            "Less than 3 months of salary",
            "3-6 months of salary",
            "more than 6 months",
        ],
        guidance=(
            "Having an adequate emergency fund ensures you can cover unexpected expenses without liquidating your investments prematurely. Typically, 3–6 months of essential living expenses is recommended to maintain financial stability."
        ),
    ),
    MCQuestion(
        id="q2",
        text="What portion of your total investable assets does this managed account represent?",
        label="Managed Account Representation Percentage",
        options=[
            "less than 25%",
            "25% to 50%",
            "more than 50%",
        ],
        guidance=(
            "Understanding how much of your overall wealth this account represents helps us determine how much risk is appropriate. If this is a small portion of your assets, you may tolerate higher risk; if it's your main portfolio, a more balanced approach may be suitable."
        ),
    ),
    MCQuestion(
        id="q3",
        text="What is your total investment  horizon for this account?",
        label="Investment Horizon",
        options=[
            "less than 5 years",
            "5-10 year",
            "10-15 year",
            "15-20 year",
            "20-25 year",
            "25-30 year",
            "30 year +",
        ],
        guidance=(
            "Your investment time horizon—the number of years before you expect to withdraw funds—is a key factor in determining the right asset allocation. Longer horizons allow for more growth-oriented investments."
        ),
    ),
    MCQuestion(
        id="q4",
        text="How likely are you to make early withdrawals from this account?",
        label="Early Withdrawals",
        options=[
            "No",
            "Less Likely",
            "Likely",
        ],
        guidance=(
            "Frequent or early withdrawals can affect your investment strategy. If withdrawals are likely, we may recommend maintaining a more liquid or conservative portfolio to avoid selling assets at unfavorable times."
        ),
    ),
    MCQuestion(
        id="q5",
        text="How would you describe your level of investment knowledge?",
        label="Investment Knowledge",
        options=[
            "A little",
            "Normal ",
            "Expert",
        ],
        guidance=(
            "Your investment knowledge helps us tailor the advice and explanations you receive. It ensures that recommendations are communicated in a way that matches your familiarity with financial concepts."
        ),
    ),
    MCQuestion(
        id="q6",
        text="How do you value portfolio growth versus income guarantee",
        label="Growth vs Income Preserve",
        options=[
            "Value growth more",
            "Treat them equal",
            "Value income guarantee more",
        ],
        guidance=(
            "This indicates the investment objective as to grow assets or guarantee income (risky versus conservative), the answer will impact how the final portfolio derailed from the neutral equity calculated before"
        ),
    ),
    MCQuestion(
        id="q7",
        text="If a market crashes and your account value drops a lot, what would be your action item afterwards",
        label="Action when market crashes",
        options=[
            "I would continue investing the same way, I believe in the long term the market will bounce back",
            "I will investment less than half of my account in a more conservative portfolio",
            "I will investment more than half of my account in a more conservative portfolio",
        ],
        guidance=(
            "This indicates when market crashes what actions the investor will take which will indicate "
            "his/her risk preference. "
        ),
    ),
]

# =============================================================================
# GLIDEPATH DATA (From Excel file - Glidepath sheet)
# =============================================================================

# Glidepath data with horizon as index and Path columns
# This data structure matches the Excel file format exactly
GLIDEPATH_DATA = {
    # Horizon: {Path 1, Path 2, Path 3, Path 4}
    1: {"Path 1": 2, "Path 2": 2, "Path 3": 3, "Path 4": 4},
    2: {"Path 1": 2, "Path 2": 2, "Path 3": 3, "Path 4": 4},
    3: {"Path 1": 2, "Path 2": 2, "Path 3": 3, "Path 4": 4},
    4: {"Path 1": 2, "Path 2": 3, "Path 3": 3, "Path 4": 4},
    5: {"Path 1": 2, "Path 2": 3, "Path 3": 3, "Path 4": 4},
    6: {"Path 1": 2, "Path 2": 3, "Path 3": 4, "Path 4": 5},
    7: {"Path 1": 3, "Path 2": 3, "Path 3": 4, "Path 4": 5},
    8: {"Path 1": 3, "Path 2": 3, "Path 3": 4, "Path 4": 5},
    9: {"Path 1": 3, "Path 2": 4, "Path 3": 4, "Path 4": 5},
    10: {"Path 1": 3, "Path 2": 4, "Path 3": 5, "Path 4": 6},
    11: {"Path 1": 4, "Path 2": 4, "Path 3": 5, "Path 4": 6},
    12: {"Path 1": 4, "Path 2": 5, "Path 3": 5, "Path 4": 6},
    13: {"Path 1": 4, "Path 2": 5, "Path 3": 5, "Path 4": 6},
    14: {"Path 1": 5, "Path 2": 5, "Path 3": 6, "Path 4": 7},
    15: {"Path 1": 5, "Path 2": 5, "Path 3": 6, "Path 4": 7},
    16: {"Path 1": 5, "Path 2": 6, "Path 3": 6, "Path 4": 7},
    17: {"Path 1": 6, "Path 2": 6, "Path 3": 6, "Path 4": 7},
    18: {"Path 1": 6, "Path 2": 6, "Path 3": 6, "Path 4": 7},
    19: {"Path 1": 6, "Path 2": 6, "Path 3": 7, "Path 4": 8},
    20: {"Path 1": 6, "Path 2": 7, "Path 3": 7, "Path 4": 8},
    21: {"Path 1": 6, "Path 2": 7, "Path 3": 7, "Path 4": 8},
    22: {"Path 1": 7, "Path 2": 7, "Path 3": 7, "Path 4": 8},
    23: {"Path 1": 7, "Path 2": 7, "Path 3": 8, "Path 4": 9},
    24: {"Path 1": 7, "Path 2": 8, "Path 3": 8, "Path 4": 9},
    25: {"Path 1": 7, "Path 2": 8, "Path 3": 8, "Path 4": 9},
    26: {"Path 1": 7, "Path 2": 8, "Path 3": 8, "Path 4": 9},
    27: {"Path 1": 8, "Path 2": 9, "Path 3": 9, "Path 4": 10},
    28: {"Path 1": 8, "Path 2": 9, "Path 3": 9, "Path 4": 10},
    29: {"Path 1": 8, "Path 2": 9, "Path 3": 9, "Path 4": 10},
    30: {"Path 1": 8, "Path 2": 9, "Path 3": 9, "Path 4": 10},
}

# =============================================================================
# PORTFOLIO INDEX DATA (From Excel file - PortfolioIndex sheet)
# =============================================================================

# Portfolio index data with index as key and equity allocation as value
# This data structure matches the Excel file format exactly
PORTFOLIO_INDEX_DATA = {
    1: 0.05,
    2: 0.15,
    3: 0.25,
    4: 0.35,
    5: 0.45,
    6: 0.55,
    7: 0.65,
    8: 0.75,
    9: 0.85,
    10: 0.95,
}

# =============================================================================
# UTILITY FUNCTIONS (Exact from RiskManager)
# =============================================================================

def get_questions() -> List[MCQuestion]:
    """Get the list of risk assessment questions."""
    return RISK_QUESTIONS

def _map_path_from_q1_q2(q1_idx: int, q2_idx: int) -> int:
    """
    Map Q1 and Q2 answers to a path (1-4).
    
    Args:
        q1_idx: Q1 selected index (0-2)
        q2_idx: Q2 selected index (0-2)
        
    Returns:
        Path number (1-4)
    """
    q1_score_map = {0: 0, 1: 1, 2: 2}
    q2_score_map = {0: 2, 1: 1, 2: 0}
    s1 = q1_score_map.get(q1_idx)
    s2 = q2_score_map.get(q2_idx)
    if s1 is None or s2 is None:
        raise ValueError("Q1/Q2 selected_index out of expected range (0..2).")
    total = s1 + s2
    if total >= 4:
        return 4
    if total == 3:
        return 3
    if total == 2:
        return 3
    if total == 1:
        return 2
    return 1  # total == 0

def _map_horizon_from_q3_q4(q3_idx: int, q4_idx: int) -> int:
    """
    Map Q3 and Q4 answers to horizon in years.
    
    Args:
        q3_idx: Q3 selected index (0-6)
        q4_idx: Q4 selected index (0-2)
        
    Returns:
        Horizon in years
    """
    q3_map = {0: 2.5, 1: 7.5, 2: 12.5, 3: 17.5, 4: 22.5, 5: 27.5, 6: 30.0}
    q4_mult = {0: 1.0, 1: 0.75, 2: 0.5}
    base = q3_map.get(q3_idx)
    mult = q4_mult.get(q4_idx)
    if base is None or mult is None:
        raise ValueError("Q3/Q4 selected_index out of expected range.")
    return int(round(base * mult))

def _bounds_from_q5(q5_idx: int) -> Tuple[int, int]:
    """
    Get risk adjustment bounds from Q5 answer.
    
    Args:
        q5_idx: Q5 selected index (0-2)
        
    Returns:
        Tuple of (upper_bound, lower_bound)
    """
    mapping = {
        0: (1, -1),
        1: (1, -2),
        2: (2, -2),
    }
    if q5_idx not in mapping:
        raise ValueError("Q5 selected_index out of expected range (0..2).")
    return mapping[q5_idx]

def _risk_adjustment_from_q6_q7(q6_idx: int, q7_idx: int) -> int:
    """
    Calculate risk adjustment from Q6 and Q7 answers.
    
    Args:
        q6_idx: Q6 selected index (0-2)
        q7_idx: Q7 selected index (0-2)
        
    Returns:
        Risk adjustment value
    """
    tri_map = {0: 1, 1: 0, 2: -1}
    a = tri_map.get(q6_idx)
    b = tri_map.get(q7_idx)
    if a is None or b is None:
        raise ValueError("Q6/Q7 selected_index out of expected range (0..2).")
    return a + b

def create_glidepath_dataframe() -> pd.DataFrame:
    """Create a DataFrame with glidepath data for compatibility."""
    df = pd.DataFrame(GLIDEPATH_DATA).T
    df.index.name = 'Horizon'
    return df

def create_portfolio_index_dataframe() -> pd.DataFrame:
    """Create a DataFrame with portfolio index data for compatibility."""
    df = pd.DataFrame(list(PORTFOLIO_INDEX_DATA.items()), columns=['Index', 'Equity']).set_index('Index')
    return df

# =============================================================================
# CONFIGURATION NOTES
# =============================================================================

CONFIGURATION_NOTES = """
RISK MODULE CONFIGURATION
=========================

This configuration file centralizes all risk assessment data and
parameters, replacing the Excel file dependencies.

RISK ASSESSMENT QUESTIONS:
- 6 comprehensive questions covering key risk factors
- Weighted scoring system for different question importance
- Covers time horizon, risk tolerance, goals, loss capacity, monitoring, experience

GLIDEPATH DATA:
- Age-based allocation recommendations
- Conservative approach for retirement planning
- Smooth transitions between age brackets
- Based on target date fund methodologies

PORTFOLIO INDEX MAPPINGS:
- Risk score ranges mapped to portfolio allocations
- 5 risk categories from conservative to aggressive
- Clear descriptions and suitability guidelines
- Balanced approach to risk-return tradeoffs

RISK SCORING PARAMETERS:
- Configurable weights for different question types
- Age adjustment factors for lifecycle planning
- Experience bonuses for sophisticated investors
- Default scores for missing data

RECOMMENDATIONS FOR PRODUCTION:
1. Validate questions with behavioral finance research
2. Update glidepath based on current life expectancy
3. Calibrate risk scores with historical performance
4. Add market regime adjustments
5. Include ESG and sustainability factors
6. Add tax optimization considerations
7. Implement dynamic rebalancing triggers
8. Add stress testing scenarios
9. Include alternative investment options
10. Add international diversification factors
"""

if __name__ == "__main__":
    print(CONFIGURATION_NOTES)
    print("\nGlidepath Data:")
    print(create_glidepath_dataframe())
    print("\nPortfolio Index:")
    print(create_portfolio_index_dataframe())
