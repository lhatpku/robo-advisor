# risk/risk_manager.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import os
from pathlib import Path
from langchain.tools import tool
from utils.risk.config import (
    get_questions, 
    create_glidepath_dataframe,
    create_portfolio_index_dataframe,
    _map_path_from_q1_q2,
    _map_horizon_from_q3_q4,
    _bounds_from_q5,
    _risk_adjustment_from_q6_q7
)


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
        """Load the predefined risk assessment questions from config."""
        return get_questions()
    
    def _get_config_path(self) -> Path:
        """Get the path to the configuration Excel file."""
        return os.path.join(Path(__file__).parent, "config", "general_investing_config.xlsx")
    
    def _load_config(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load configuration from config file.
        
        Returns:
            Tuple of (glidepath_df, portfolio_index_df)
        """
        glide = create_glidepath_dataframe()
        port = create_portfolio_index_dataframe()
        return glide, port
    
    def _map_path_from_q1_q2(self, q1_idx: int, q2_idx: int) -> int:
        """Map Q1 and Q2 answers to a path (1-4) using config function."""
        return _map_path_from_q1_q2(q1_idx, q2_idx)
    
    def _map_horizon_from_q3_q4(self, q3_idx: int, q4_idx: int) -> int:
        """Map Q3 and Q4 answers to horizon in years using config function."""
        return _map_horizon_from_q3_q4(q3_idx, q4_idx)
    
    def _bounds_from_q5(self, q5_idx: int) -> Tuple[int, int]:
        """Get risk adjustment bounds from Q5 answer using config function."""
        return _bounds_from_q5(q5_idx)
    
    def _risk_adjustment_from_q6_q7(self, q6_idx: int, q7_idx: int) -> int:
        """Calculate risk adjustment from Q6 and Q7 answers using config function."""
        return _risk_adjustment_from_q6_q7(q6_idx, q7_idx)
    
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
