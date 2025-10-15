from __future__ import annotations
from typing import Dict, Callable, Any
from portfolio.optimizer import mean_variance_optimizer
from portfolio.set_portfolio_param import set_portfolio_param
import json

def create_tool_registry() -> Dict[str, Callable[[Dict[str, Any]], Any]]:
    """Map tool names to callables that accept a dict args and return result."""
    return {
        # Wrap LangChain @tool tools with .invoke for a consistent call signature
        "mean_variance_optimizer": lambda args: mean_variance_optimizer.invoke(args),
        "set_portfolio_param":   lambda args: set_portfolio_param.invoke(args),
    }

def execute_tool_call(call: Dict[str, Any], registry: Dict[str, Callable]) -> Any:
    """Executes a single tool call of shape {"tool": <name>, "args": {...}}."""
    name = call.get("tool")
    args = call.get("args", {})
    if name not in registry:
        return {"error": f"Unknown tool: {name}"}
    return registry[name](args)

def _plan_tools_with_llm(llm, state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Ask the LLM to propose a list of tool calls.
    Tools available:
      - set_portfolio_param(param:str in {"lambda","cash_reserve"}, value: float)
      - mean_variance_optimizer(mu_cov_xlsx_path:str, risk_equity:float, risk_bonds:float, lam:float, cash_reserve:float)

    Contract:
      - Return ONLY a JSON array of tool calls. No prose.
      - If user says "run"/"optimize"/"go ahead", call mean_variance_optimizer once.
      - If user asks to change param(s) (e.g., "lambda 1", "cash 0.05"), emit set_portfolio_param for each change.
      - If user both changes params AND says to run, emit set_portfolio_param calls first, then mean_variance_optimizer.
      - If unclear, return [].
    """
    inv = state.get("portfolio") or {}
    risk = state.get("risk") or {}
    user_text = ""
    if state.get("messages") and state["messages"][-1].get("role") == "user":
        user_text = state["messages"][-1].get("content", "")

    system = (
        "You are a tool planner. Decide which tools to call next given the user's latest message.\n"
        "Tools:\n"
        '1) set_portfolio_param(param:str, value:float)\n'
        '2) mean_variance_optimizer(mu_cov_xlsx_path:str, risk_equity:float, risk_bonds:float, lam:float, cash_reserve:float)\n'
        "Rules:\n"
        "- Only output JSON (no surrounding text): a list like [{\"tool\":\"set_portfolio_param\",\"args\":{...}}, ...].\n"
        "- Interpret terse inputs like “lambda 1”, “cash 0.05”, “run”, “go ahead”, “proceed”.\n"
        "- If user updates multiple params, emit multiple set_portfolio_param calls in the order mentioned.\n"
        "- If they also want to run, emit mean_variance_optimizer as the last call.\n"
        "- If message is ambiguous or just chit-chat, return []."
    )

    # Few-shot examples to reduce ambiguity
    examples = [
        {
            "user": "set lambda to 12 and run",
            "calls": [
                {"tool":"set_portfolio_param","args":{"param":"lambda","value":12}},
                {"tool":"mean_variance_optimizer","args":"<auto-fill current params>"}
            ]
        },
        {
            "user": "cash 0.05, lambda 8",
            "calls": [
                {"tool":"set_portfolio_param","args":{"param":"cash_reserve","value":0.05}},
                {"tool":"set_portfolio_param","args":{"param":"lambda","value":8}}
            ]
        },
        {
            "user": "run",
            "calls": [
                {"tool":"mean_variance_optimizer","args":"<auto-fill current params>"}
            ]
        },
        {
            "user": "what is lambda?",
            "calls": []
        }
    ]

    exemplar = json.dumps(examples, indent=2)
    user = (
        f"Current parameters: lambda={inv.get('lambda')}, cash_reserve={inv.get('cash_reserve')}.\n"
        f"risk split: equity={risk.get('equity', 0.0)}, bonds={risk.get('bond', 0.0)}.\n"
        f"Mu/Cov file: {inv.get('mu_cov_xlsx_path','')}\n\n"
        f"Examples (for guidance, not for output):\n{exemplar}\n\n"
        f"Latest user message:\n{user_text}\n\n"
        "Now output ONLY the JSON array of tool calls."
    )

    resp = llm.invoke([{"role":"system","content":system},{"role":"user","content":user}])
    content = getattr(resp, "content", "") if resp else ""
    try:
        calls = json.loads(content)
        return calls if isinstance(calls, list) else []
    except Exception:
        return []
