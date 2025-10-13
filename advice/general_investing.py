# advice/general_investing.py
from __future__ import annotations
from typing import Dict, Any
from pathlib import Path
import math
import pandas as pd
from langchain.tools import tool
import os 


def _config_path() -> Path:
    # The Excel file is expected to live at: <this_file_dir>/config/general_investing_config.xlsx
    return os.path.join(Path(__file__).parent, "config", "general_investing_config.xlsx")


def _load_config() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simplest loader: assume the Excel is well-formed.
      - 'Glidepath' sheet: first column is the horizon (already clean). We set it as the index.
      - 'PortfolioIndex' sheet: first column is the index (1..10), second column is Equity.
        If Equity values look like percentages (>1), convert to 0..1.
    """
    path = _config_path()
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

def _map_path_from_q1_q2(q1_idx: int, q2_idx: int) -> int:
    """
    Q1 options -> scores: [0,1,2] for indices [0,1,2]
    Q2 options -> scores: [2,1,0] for indices [0,1,2]
    total=4 -> Path 4
    total=3 -> Path 3
    total=2 -> Path 3
    total=1 -> Path 2
    total=0 -> Path 1
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
    Q3 options (index 0..6) -> [2.5, 7.5, 12.5, 17.5, 22.5, 27.5, 30]
    Q4 options multiplier (index 0..2 for No, Less Likely, Likely) -> [1.0, 0.75, 0.5]
    horizon_value = round( horizon[q3_idx] * multiplier[q4_idx] )  -> integer years
    """
    q3_map = {0: 2.5, 1: 7.5, 2: 12.5, 3: 17.5, 4: 22.5, 5: 27.5, 6: 30.0}
    q4_mult = {0: 1.0, 1: 0.75, 2: 0.5}
    base = q3_map.get(q3_idx)
    mult = q4_mult.get(q4_idx)
    if base is None or mult is None:
        raise ValueError("Q3/Q4 selected_index out of expected range.")
    return int(round(base * mult))


def _bounds_from_q5(q5_idx: int) -> tuple[int, int]:
    """
    Q5:
      - A little  -> (upper=1, lower=-1)
      - Normal    -> (upper=1, lower=-2)
      - Expert    -> (upper=2, lower=-2)
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
    Q6 (Value growth more, Treat them equal, Value income guarantee more) -> (1, 0, -1)
    Q7 (continue investing; less than half conservative; more than half conservative) -> (1, 0, -1)
    Sum them, then later clamp within Q5 bounds.
    """
    tri_map = {0: 1, 1: 0, 2: -1}
    a = tri_map.get(q6_idx)
    b = tri_map.get(q7_idx)
    if a is None or b is None:
        raise ValueError("Q6/Q7 selected_index out of expected range (0..2).")
    return a + b


@tool("general_investing_advice")
def general_investing_advice_tool(answers: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute equity/bond allocation using config in 'config/general_investing_config.xlsx'.
    Inputs:
        answers[qid] = {
            "selected_index": int,
            "selected_label": str,
            "raw_user_text": str
        }
      where qid in {"q1","q2","q3","q4","q5","q6","q7"}.
    Logic:
      1) Read 'Glidepath' and 'PortfolioIndex' sheets.
      2) Use Q1+Q2 to pick Path (1..4).
      3) Use Q3 adjusted by Q4 to compute an integer horizon, then look up Glidepath[horizon, Path] to get a base portfolio index.
      4) Q5 sets upper/lower bounds for risk adjustment; Q6 & Q7 map to {-1,0,1} and sum to a risk_adjustment,
         clamped within those bounds. Add to base index, clamp to [1,10].
      5) Use the final index to fetch Equity allocation from PortfolioIndex.
    Returns:
        {"equity": float, "bond": float}
    """
    required = ["q1", "q2", "q3", "q4", "q5", "q6", "q7"]
    for q in required:
        if q not in answers or "selected_index" not in answers[q]:
            raise ValueError(f"Missing or malformed answers for {q}")

    # Load config tables
    glide, port_index = _load_config()  # glide has columns: "Path 1".."Path 4"; index=horizon int

    # 1+2) Choose path using Q1, Q2
    path = _map_path_from_q1_q2(answers["q1"]["selected_index"], answers["q2"]["selected_index"])

    # 3) Compute horizon using Q3, Q4 and look up base index from Glidepath
    horizon_year = _map_horizon_from_q3_q4(answers["q3"]["selected_index"], answers["q4"]["selected_index"])

    # If horizon not in index, try to clamp to nearest available within [min,max]
    if horizon_year not in glide.index:
        # clamp to nearest horizon available
        min_h, max_h = glide.index.min(), glide.index.max()
        horizon_year = min(max(horizon_year, min_h), max_h)

    path_col = f"Path {path}"
    if path_col not in glide.columns:
        raise ValueError(f"Expected '{path_col}' in Glidepath columns: {list(glide.columns)}")

    # This value is the "portfolio index" baseline before risk adjustments
    # It should be an integer in [1..10]. If not, we'll coerce/clamp below.
    base_index_val = glide.loc[horizon_year, path_col]
    try:
        base_index = int(round(float(base_index_val)))
    except Exception:
        raise ValueError(f"Glidepath value at horizon={horizon_year}, {path_col} is not numeric: {base_index_val}")

    # 4) Risk adjustment bounds from Q5
    upper, lower = _bounds_from_q5(answers["q5"]["selected_index"])
    # Sum of Q6/Q7 adjustments
    risk_adj = _risk_adjustment_from_q6_q7(answers["q6"]["selected_index"], answers["q7"]["selected_index"])
    # Clamp within bounds
    risk_adj = max(lower, min(upper, risk_adj))

    # Final index = base + risk_adj, clamped to [1..10]
    final_index = max(1, min(10, base_index + risk_adj))

    # 5) Lookup equity allocation in PortfolioIndex
    if final_index not in port_index.index:
        # clamp to nearest available index
        min_i, max_i = port_index.index.min(), port_index.index.max()
        final_index = min(max(final_index, min_i), max_i)

    equity = float(port_index.loc[final_index, "Equity"])
    # Ensure 0..1
    if equity > 1.0:
        equity = equity / 100.0
    equity = max(0.0, min(1.0, equity))

    return {"equity": round(equity, 4), "bond": round(1.0 - equity, 4)}
