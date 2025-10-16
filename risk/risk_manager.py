# risk/risk_manager.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from pathlib import Path
import math
import pandas as pd
from langchain.tools import tool
import os


@dataclass
class MCQuestion:
    id: str
    text: str
    label: str
    options: List[str]
    guidance: str


@dataclass
class MCAnswer:
    selected_index: int
    selected_label: str
    raw_user_text: str


class RiskManager:
    """
    Comprehensive risk assessment class that combines question management
    and risk calculation functionality.
    """
    
    def __init__(self):
        """Initialize the RiskManager with questions and configuration."""
        self.questions = self._load_questions()
        self._config_path = self._get_config_path()
    
    def _load_questions(self) -> List[MCQuestion]:
        """Load the predefined risk assessment questions."""
        return [
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
    
    def _get_config_path(self) -> Path:
        """Get the path to the configuration Excel file."""
        return os.path.join(Path(__file__).parent, "config", "general_investing_config.xlsx")
    
    def _load_config(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load configuration from Excel file.
        
        Returns:
            Tuple of (glidepath_df, portfolio_index_df)
        """
        path = self._config_path
        glide = pd.read_excel(path, sheet_name="Glidepath")
        port = pd.read_excel(path, sheet_name="PortfolioIndex")

        # Set indices based on the first column of each sheet
        glide = glide.set_index(glide.columns[0])

        # Equity assumed to be the second column
        port = port.set_index(port.columns[0])
        equity_col = port.columns[1] if len(port.columns) > 1 else port.columns[0]
        equity = port[equity_col]
        if equity.max() > 1.0:
            equity = equity / 100.0
        port = equity.to_frame(name="Equity")

        return glide, port
    
    def _map_path_from_q1_q2(self, q1_idx: int, q2_idx: int) -> int:
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
    
    def _map_horizon_from_q3_q4(self, q3_idx: int, q4_idx: int) -> int:
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
    
    def _bounds_from_q5(self, q5_idx: int) -> Tuple[int, int]:
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
    
    def _risk_adjustment_from_q6_q7(self, q6_idx: int, q7_idx: int) -> int:
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
    
    def calculate_risk_allocation(self, answers: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate equity/bond allocation based on questionnaire answers.
        
        Args:
            answers: Dict mapping question IDs to answer data
            
        Returns:
            Dict with 'equity' and 'bond' allocations
        """
        required = ["q1", "q2", "q3", "q4", "q5", "q6", "q7"]
        for q in required:
            if q not in answers or "selected_index" not in answers[q]:
                raise ValueError(f"Missing or malformed answers for {q}")

        # Load config tables
        glide, port_index = self._load_config()

        # 1+2) Choose path using Q1, Q2
        path = self._map_path_from_q1_q2(answers["q1"]["selected_index"], answers["q2"]["selected_index"])

        # 3) Compute horizon using Q3, Q4 and look up base index from Glidepath
        horizon_year = self._map_horizon_from_q3_q4(answers["q3"]["selected_index"], answers["q4"]["selected_index"])

        # If horizon not in index, try to clamp to nearest available within [min,max]
        if horizon_year not in glide.index:
            min_h, max_h = glide.index.min(), glide.index.max()
            horizon_year = min(max(horizon_year, min_h), max_h)

        path_col = f"Path {path}"
        if path_col not in glide.columns:
            raise ValueError(f"Expected '{path_col}' in Glidepath columns: {list(glide.columns)}")

        # This value is the "portfolio index" baseline before risk adjustments
        base_index_val = glide.loc[horizon_year, path_col]
        try:
            base_index = int(round(float(base_index_val)))
        except Exception:
            raise ValueError(f"Glidepath value at horizon={horizon_year}, {path_col} is not numeric: {base_index_val}")

        # 4) Risk adjustment bounds from Q5
        upper, lower = self._bounds_from_q5(answers["q5"]["selected_index"])
        # Sum of Q6/Q7 adjustments
        risk_adj = self._risk_adjustment_from_q6_q7(answers["q6"]["selected_index"], answers["q7"]["selected_index"])
        # Clamp within bounds
        risk_adj = max(lower, min(upper, risk_adj))

        # Final index = base + risk_adj, clamped to [1..10]
        final_index = max(1, min(10, base_index + risk_adj))

        # 5) Lookup equity allocation in PortfolioIndex
        if final_index not in port_index.index:
            min_i, max_i = port_index.index.min(), port_index.index.max()
            final_index = min(max(final_index, min_i), max_i)

        equity = float(port_index.loc[final_index, "Equity"])
        # Ensure 0..1
        if equity > 1.0:
            equity = equity / 100.0
        equity = max(0.0, min(1.0, equity))

        return {"equity": round(equity, 4), "bond": round(1.0 - equity, 4)}
    
    def get_question(self, index: int) -> MCQuestion:
        """Get a question by index."""
        if 0 <= index < len(self.questions):
            return self.questions[index]
        raise IndexError(f"Question index {index} out of range")
    
    def get_question_by_id(self, question_id: str) -> MCQuestion:
        """Get a question by ID."""
        for q in self.questions:
            if q.id == question_id:
                return q
        raise ValueError(f"Question with ID '{question_id}' not found")
    
    def get_total_questions(self) -> int:
        """Get the total number of questions."""
        return len(self.questions)


# LangChain tool wrapper for backward compatibility
@tool("general_investing_risk")
def general_investing_risk_tool(answers: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    """Compute equity/bond allocation using config in 'config/general_investing_config.xlsx'."""
    manager = RiskManager()
    return manager.calculate_risk_allocation(answers)


# Backward compatibility - expose questions as module-level constants
QUESTIONS = RiskManager().questions
