from __future__ import annotations
from typing import Dict, Any
from langchain.tools import tool

@tool("set_investment_param")
def set_investment_param(param: str, value: float, current: Dict[str, float] | None = None) -> Dict[str, Any]:
    """Validate/update an investment parameter.

    Args:
      param: one of ["lambda", "cash_reserve"]
      value: desired numeric value
      current: optional current dict with keys "lambda" and "cash_reserve"

    Returns:
      {
        "ok": bool,
        "param": str,
        "new_value": float,
        "note": str
      }

    Notes:
      - We do minimal validation here; the agent can clamp cash later for the optimizer.
      - Lambda must be > 0; cash should be between 0 and 0.05.
    """
    if param not in {"lambda", "cash_reserve"}:
        return {"ok": False, "param": param, "new_value": None, "note": "Unsupported parameter."}

    if param == "lambda":
        if value <= 0:
            return {"ok": False, "param": param, "new_value": None, "note": "Lambda must be > 0."}
        return {"ok": True, "param": param, "new_value": float(value), "note": "Updated lambda."}

    if param == "cash_reserve":
        if not (0.0 <= value <= 0.05):
            return {"ok": False, "param": param, "new_value": None, "note": "Cash reserve should be within 0.0–0.05."}
        return {"ok": True, "param": param, "new_value": float(value), "note": "Updated cash reserve."}
