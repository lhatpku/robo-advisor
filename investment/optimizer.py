
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import os
from langchain.tools import tool

EQUITY = ["large_cap_growth","large_cap_value","small_cap_growth","small_cap_value","developed_market_equity","emerging_market_equity"]
BONDS  = ["short_term_treasury","mid_term_treasury","long_term_treasury","corporate_bond","tips","cash"]
ALL_ASSETS = EQUITY + BONDS

def _read_mu_cov_from_excel(path: str) -> Tuple[Dict[str,float], np.ndarray, List[str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Excel file not found: {path}")
    mu_df = pd.read_excel(path, sheet_name="mu")
    cov_df = pd.read_excel(path, sheet_name="cov", index_col=0)

    # Validate names alignment
    names = list(mu_df["asset"])
    if set(names) != set(cov_df.index) or set(names) != set(cov_df.columns):
        raise ValueError("Mismatch between mu assets and covariance matrix labels.")

    mu_map = dict(zip(mu_df["asset"], mu_df["mean"]))
    # Reorder covariance to the same order as names
    cov_df = cov_df.loc[names, names]
    Sigma = cov_df.values
    return mu_map, Sigma, names

def _solve_bucket(mu: np.ndarray, Sigma: np.ndarray, lam: float) -> np.ndarray:
    n = len(mu)
    inv = np.linalg.pinv(Sigma + 1e-8*np.eye(n))
    ones = np.ones(n)
    A = inv @ ones
    B = inv @ mu
    a = ones @ A
    b = ones @ B
    nu = (b - lam) / (a + 1e-12)
    w = (1.0/lam) * (B - nu * A)
    w = np.clip(w, 0.0, None)
    s = w.sum()
    if s <= 1e-12:
        w = np.ones(n)/n
    else:
        w /= s
    return w

def optimize_portfolio_from_file(
    mu_cov_xlsx_path: str,
    advice_equity: float,
    advice_bonds: float,
    lam: float,
    cash_reserve: float
) -> Dict[str, float]:
    """
    Returns {asset_class: weight} across ALL classes.
    Reads mu (vector) and cov (matrix) from Excel.
    """
    if lam <= 0:
        raise ValueError("lambda must be positive (try 1; higher = more conservative).")
    if not (0.0 <= cash_reserve <= 0.05):
        raise ValueError("cash_reserve must be between 0.0 and 0.05 (0%–5%).")

    mu_map, Sigma, names = _read_mu_cov_from_excel(mu_cov_xlsx_path)
    idx = {n:i for i,n in enumerate(names)}

    # Determine available sets based on file (fallback to expected lists)
    EQU = [n for n in names if n in EQUITY]
    BND = [n for n in names if n in BONDS and n != "cash"]
    has_cash = "cash" in names

    mu_eq = np.array([mu_map[n] for n in EQU])
    Sig_eq = Sigma[np.ix_([idx[n] for n in EQU], [idx[n] for n in EQU])]
    w_eq   = _solve_bucket(mu_eq, Sig_eq, lam) if len(EQU) else np.array([])

    mu_fi = np.array([mu_map[n] for n in BND])
    Sig_fi = Sigma[np.ix_([idx[n] for n in BND], [idx[n] for n in BND])]
    w_fi   = _solve_bucket(mu_fi, Sig_fi, lam) if len(BND) else np.array([])

    bonds_ex_cash_target = max(0.0, advice_bonds - (cash_reserve if has_cash else 0.0))
    if advice_equity + advice_bonds > 1.0001:
        scale = 1.0 / (advice_equity + advice_bonds)
        advice_equity *= scale
        advice_bonds  *= scale
        bonds_ex_cash_target = max(0.0, advice_bonds - (cash_reserve if has_cash else 0.0))

    out: Dict[str, float] = {}
    for n, w in zip(EQU, w_eq):
        out[n] = float(w * advice_equity)
    for n, w in zip(BND, w_fi):
        out[n] = float(w * bonds_ex_cash_target)
    if has_cash:
        out["cash"] = float(cash_reserve)

    s = sum(out.values())
    if s > 0:
        for k in list(out.keys()):
            out[k] = out[k]/s
    return out

# -------- LangChain Tool wrapper --------

@tool("mean_variance_optimizer")
def mean_variance_optimizer(
    mu_cov_xlsx_path: str,
    advice_equity: float,
    advice_bonds: float,
    lam: float,
    cash_reserve: float
) -> Dict[str, float]:
    """Optimize asset-class weights via mean-variance, reading 'mu' and 'cov' sheets from an Excel file.
    Args:
      mu_cov_xlsx_path: Path to Excel with sheets 'mu' (asset, mean) and 'cov' (covariance matrix with headers).
      advice_equity: Equity bucket target (e.g., 0.7).
      advice_bonds: Bond bucket target (e.g., 0.3).
      lam: Risk-aversion parameter; larger means more conservative (try 5–20).
      cash_reserve: Cash proportion between 0.03 and 0.06.
    Returns:
      Dict[str, float]: mapping from asset class to portfolio weight that sums to ~1.0.
    """
    return optimize_portfolio_from_file(
        mu_cov_xlsx_path=mu_cov_xlsx_path,
        advice_equity=float(advice_equity),
        advice_bonds=float(advice_bonds),
        lam=float(lam),
        cash_reserve=float(cash_reserve),
    )
