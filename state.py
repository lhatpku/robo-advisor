from typing import TypedDict, List, Dict, Any, Optional

class AgentState(TypedDict, total=False):
    messages: List[Dict[str, Any]]
    q_idx: int
    answers: Dict[str, Dict[str, Any]]   # qid -> MCAnswer as dict
    done: bool
    advice: Optional[Dict[str, float]]
    awaiting_input: bool                 # prevents recursion while waiting
    intent_to_advise: bool
    entry_greeted: bool
    intent_to_investment: bool
    investment: Optional[Dict[str, Any]]   # {"lambda": float, "cash_reserve": float, "portfolio": dict}
