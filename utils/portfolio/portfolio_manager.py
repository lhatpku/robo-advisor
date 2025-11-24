from __future__ import annotations
from typing import Dict, List, Tuple, Callable, Any
import numpy as np
import pandas as pd
import os
import json
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from utils.portfolio.config import get_expected_returns, get_covariance_matrix, ASSET_CLASSES, get_cash_reserve_constraints, validate_cash_reserve, DEFAULT_LAMBDA, DEFAULT_CASH_RESERVE


class PortfolioManager:
    """
    Comprehensive portfolio management class that combines optimization,
    parameter setting, and tool execution functionality.
    """
    
    # Asset class definitions
    EQUITY = ["large_cap_growth", "large_cap_value", "small_cap_growth", "small_cap_value", 
              "developed_market_equity", "emerging_market_equity"]
    BONDS = ["short_term_treasury", "mid_term_treasury", "long_term_treasury", 
             "corporate_bond", "tips", "cash"]
    ALL_ASSETS = EQUITY + BONDS
    
    def __init__(self, llm: ChatOpenAI = None):
        """
        Initialize the PortfolioManager.
        
        Args:
            llm: Optional ChatOpenAI instance for tool planning
        """
        self.llm = llm
        self._tool_registry = self._create_tool_registry()
    
    def _read_mu_cov_from_excel(self, path: str) -> Tuple[Dict[str, float], np.ndarray, List[str]]:
        """
        Read mean returns and covariance matrix from Excel file.
        
        Args:
            path: Path to Excel file with 'mu' and 'cov' sheets
            
        Returns:
            Tuple of (mu_map, Sigma, names)
        """
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
    
    def _read_mu_cov_from_config(self) -> Tuple[Dict[str, float], np.ndarray, List[str]]:
        """
        Read mean returns and covariance matrix from config file.
        
        Returns:
            Tuple of (mu_map, Sigma, names)
        """
        # Get expected returns and covariance matrix from config
        mu_array = get_expected_returns()
        Sigma = get_covariance_matrix()
        names = ASSET_CLASSES
        
        # Convert to dictionary format
        mu_map = dict(zip(names, mu_array))
        
        return mu_map, Sigma, names
    
    def _solve_bucket(self, mu: np.ndarray, Sigma: np.ndarray, lam: float) -> np.ndarray:
        """
        Solve mean-variance optimization for a single asset bucket.
        
        Args:
            mu: Mean returns vector
            Sigma: Covariance matrix
            lam: Risk aversion parameter
            
        Returns:
            Optimal weights vector
        """
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
    
    def mean_variance_optimizer(
        self,
        risk_equity: float,
        risk_bonds: float,
        lam: float,
        cash_reserve: float
    ) -> Dict[str, float]:
        """
        Optimize portfolio weights using mean-variance optimization.
        
        Args:
            risk_equity: Equity bucket target (e.g., 0.7)
            risk_bonds: Bond bucket target (e.g., 0.3)
            lam: Risk-aversion parameter (higher = more conservative)
            cash_reserve: Cash proportion between 0.02 and 0.05
            
        Returns:
            Dict mapping asset class to portfolio weight
        """
        if lam <= 0:
            raise ValueError("lambda must be positive (try 1; higher = more conservative).")
        min_cash, max_cash = get_cash_reserve_constraints()
        if not validate_cash_reserve(cash_reserve):
            raise ValueError(f"cash_reserve must be between {min_cash:.2f} and {max_cash:.2f} ({min_cash*100:.0f}%–{max_cash*100:.0f}%).")

        # Use config instead of Excel file
        mu_map, Sigma, names = self._read_mu_cov_from_config()
        idx = {n: i for i, n in enumerate(names)}

        # Determine available sets based on file (fallback to expected lists)
        EQU = [n for n in names if n in self.EQUITY]
        BND = [n for n in names if n in self.BONDS and n != "cash"]
        has_cash = "cash" in names

        mu_eq = np.array([mu_map[n] for n in EQU])
        Sig_eq = Sigma[np.ix_([idx[n] for n in EQU], [idx[n] for n in EQU])]
        w_eq = self._solve_bucket(mu_eq, Sig_eq, lam) if len(EQU) else np.array([])

        mu_fi = np.array([mu_map[n] for n in BND])
        Sig_fi = Sigma[np.ix_([idx[n] for n in BND], [idx[n] for n in BND])]
        w_fi = self._solve_bucket(mu_fi, Sig_fi, lam) if len(BND) else np.array([])

        bonds_ex_cash_target = max(0.0, risk_bonds - (cash_reserve if has_cash else 0.0))
        if risk_equity + risk_bonds > 1.0001:
            scale = 1.0 / (risk_equity + risk_bonds)
            risk_equity *= scale
            risk_bonds *= scale
            bonds_ex_cash_target = max(0.0, risk_bonds - (cash_reserve if has_cash else 0.0))

        out: Dict[str, float] = {}
        for n, w in zip(EQU, w_eq):
            out[n] = float(w * risk_equity)
        for n, w in zip(BND, w_fi):
            out[n] = float(w * bonds_ex_cash_target)
        if has_cash:
            out["cash"] = float(cash_reserve)

        s = sum(out.values())
        if s > 0:
            for k in list(out.keys()):
                out[k] = out[k]/s
        return out
    
    def set_portfolio_param(
        self, 
        param: str, 
        value: float, 
        current: Dict[str, float] | None = None
    ) -> Dict[str, Any]:
        """
        Validate and update a portfolio parameter.
        
        Args:
            param: One of ["lambda", "cash_reserve"]
            value: Desired numeric value
            current: Optional current dict with keys "lambda" and "cash_reserve"
            
        Returns:
            Dict with validation result and new value
        """
        if param not in {"lambda", "cash_reserve"}:
            return {"ok": False, "param": param, "new_value": None, "note": "Unsupported parameter."}

        if param == "lambda":
            if value <= 0:
                return {"ok": False, "param": param, "new_value": None, "note": "Lambda must be > 0."}
            return {"ok": True, "param": param, "new_value": float(value), "note": "Updated lambda."}

        if param == "cash_reserve":
            min_cash, max_cash = get_cash_reserve_constraints()
            if not validate_cash_reserve(value):
                return {"ok": False, "param": param, "new_value": None, "note": f"Cash reserve should be within {min_cash:.2f}–{max_cash:.2f}."}
            return {"ok": True, "param": param, "new_value": float(value), "note": "Updated cash reserve."}
    
    def _create_tool_registry(self) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
        """Create registry of available tools."""
        return {
            "mean_variance_optimizer": lambda args: self.mean_variance_optimizer(**args),
            "set_portfolio_param": lambda args: self.set_portfolio_param(**args),
        }
    
    def execute_tool_call(self, call: Dict[str, Any]) -> Any:
        """
        Execute a single tool call.
        
        Args:
            call: Dict with "tool" and "args" keys
            
        Returns:
            Tool execution result
        """
        name = call.get("tool")
        args = call.get("args", {})
        if name not in self._tool_registry:
            return {"error": f"Unknown tool: {name}"}
        return self._tool_registry[name](args)
    
    def _plan_tools_with_llm(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Use LLM to plan tool calls based on user input.
        
        Args:
            state: Current application state
            
        Returns:
            List of tool calls to execute
        """
        if not self.llm:
            return []
            
        inv = state.get("portfolio") or {}
        risk = state.get("risk") or {}
        user_text = ""
        if state.get("messages") and state["messages"][-1].get("role") == "user":
            user_text = state["messages"][-1].get("content", "")

        # Get current parameters from state (passed from portfolio agent)
        current_lambda = state.get("current_lambda", DEFAULT_LAMBDA)
        current_cash_reserve = state.get("current_cash_reserve", DEFAULT_CASH_RESERVE)
        
        # Get dynamic constraints from config
        min_cash, max_cash = get_cash_reserve_constraints()

        system = (
            "Plan tool calls for portfolio optimization based on user input.\n\n"
            "TOOLS:\n"
            "• set_portfolio_param(param, value) - Update lambda or cash_reserve\n"
            "• mean_variance_optimizer(...) - Run portfolio optimization\n\n"
            "RULES:\n"
            "• Output JSON array: [{\"tool\":\"name\", \"args\":{...}}]\n"
            f"• Parse terse inputs: 'lambda 1', 'cash {max_cash:.2f}', 'run', 'proceed'\n"
            "• Multiple params → multiple set_portfolio_param calls\n"
            "• Include optimization as final call if user wants to run\n"
            "• Ambiguous/chat → return []\n"
            "• Use current params for optimization if not specified\n"
            f"• Cash reserve must be between {min_cash:.2f} and {max_cash:.2f}"
        )

        # Few-shot examples to reduce ambiguity
        examples = [
            {
                "user": "set lambda to 12 and run",
                "calls": [
                    {"tool": "set_portfolio_param", "args": {"param": "lambda", "value": 12}},
                    {"tool": "mean_variance_optimizer", "args": "<auto-fill current params>"}
                ]
            },
            {
                "user": f"cash {max_cash:.2f}, lambda 8",
                "calls": [
                    {"tool": "set_portfolio_param", "args": {"param": "cash_reserve", "value": max_cash}},
                    {"tool": "set_portfolio_param", "args": {"param": "lambda", "value": 8}}
                ]
            },
            {
                "user": "run",
                "calls": [
                    {"tool": "mean_variance_optimizer", "args": "<auto-fill current params>"}
                ]
            },
            {
                "user": "what is lambda?",
                "calls": []
            }
        ]

        exemplar = json.dumps(examples, indent=2)
        user = (
            f"Current parameters: lambda={current_lambda}, cash_reserve={current_cash_reserve}.\n"
            f"risk split: equity={risk.get('equity', 0.0)}, bonds={risk.get('bond', 0.0)}.\n\n"
            f"Examples (for guidance, not for output):\n{exemplar}\n\n"
            f"Latest user message:\n{user_text}\n\n"
            "Now output ONLY the JSON array of tool calls."
        )

        resp = self.llm.invoke([{"role": "system", "content": system}, {"role": "user", "content": user}])
        content = getattr(resp, "content", "") if resp else ""
        try:
            calls = json.loads(content)
            return calls if isinstance(calls, list) else []
        except Exception:
            return []
    
    def plan_and_execute_tools(self, state: Dict[str, Any]) -> List[Any]:
        """
        Plan and execute tools based on current state.
        
        Args:
            state: Current application state
            
        Returns:
            List of tool execution results
        """
        tool_calls = self._plan_tools_with_llm(state)
        results = []
        for call in tool_calls:
            result = self.execute_tool_call(call)
            results.append(result)
        return results


# LangChain tool wrappers for backward compatibility
@tool("mean_variance_optimizer")
def mean_variance_optimizer(
    risk_equity: float,
    risk_bonds: float,
    lam: float,
    cash_reserve: float
) -> Dict[str, float]:
    """Optimize asset-class weights via mean-variance using config data."""
    manager = PortfolioManager()
    return manager.mean_variance_optimizer(
        risk_equity=float(risk_equity),
        risk_bonds=float(risk_bonds),
        lam=float(lam),
        cash_reserve=float(cash_reserve),
    )


@tool("set_portfolio_param")
def set_portfolio_param(
    param: str, 
    value: float, 
    current: Dict[str, float] | None = None
) -> Dict[str, Any]:
    """Validate/update a portfolio parameter."""
    manager = PortfolioManager()
    return manager.set_portfolio_param(param, value, current)


# Backward compatibility functions
def create_tool_registry() -> Dict[str, Callable[[Dict[str, Any]], Any]]:
    """Map tool names to callables that accept a dict args and return result."""
    manager = PortfolioManager()
    return manager._create_tool_registry()


def execute_tool_call(call: Dict[str, Any], registry: Dict[str, Callable]) -> Any:
    """Executes a single tool call of shape {"tool": <name>, "args": {...}}."""
    manager = PortfolioManager()
    return manager.execute_tool_call(call)


def _plan_tools_with_llm(llm, state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Ask the LLM to propose a list of tool calls."""
    manager = PortfolioManager(llm)
    return manager._plan_tools_with_llm(state)
