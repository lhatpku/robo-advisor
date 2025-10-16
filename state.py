from typing import TypedDict, List, Dict, Any, Optional

class AgentState(TypedDict, total=False):
    messages: List[Dict[str, Any]]
    q_idx: int
    answers: Dict[str, Dict[str, Any]]   # qid -> MCAnswer as dict
    done: bool
    risk: Optional[Dict[str, float]]
    awaiting_input: bool                 # prevents recursion while waiting
    intent_to_risk: bool
    entry_greeted: bool
    intent_to_portfolio: bool
    intent_to_investment: bool
    intent_to_trading: bool
    portfolio: Optional[Dict[str, Any]]   # {"lambda": float, "cash_reserve": float, "portfolio": dict, "investment": dict}
    trading_requests: Optional[Dict[str, Any]]  # Trading requests and analysis
