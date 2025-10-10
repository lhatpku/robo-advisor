# general_investing.py
from __future__ import annotations
from typing import Dict, Any
from langchain.tools import tool

@tool("general_investing_advice")
def general_investing_advice_tool(answers: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute a preliminary equity/bond allocation based on the 7 answers.
    Input: answers[qid] = {"selected_index": int, "selected_label": str, "raw_user_text": str}
    Return: {"equity": float, "bond": float}
    ---
    """
    # --- Placeholder logic (replace me) ---
    equity = 0.60
    # Horizon tilt (q3)
    if "q3" in answers:
        equity += 0.04 * answers["q3"]["selected_index"]  # later option ≈ longer horizon
    # Withdrawal penalty (q4)
    if "q4" in answers and answers["q4"]["selected_index"] == 2:  # "Likely"
        equity -= 0.08
    # Clamp, round
    equity = max(0.20, min(0.90, round(equity, 2)))
    return {"equity": equity, "bond": round(1 - equity, 2)}
