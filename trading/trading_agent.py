"""
Refactored Trading Agent

This agent handles trading request generation using LLM structured output
and the SoftObjectiveRebalancer for tax-aware rebalancing.
"""

from __future__ import annotations
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from state import AgentState
from trading.trading_utils import TradingUtils
from trading.trading_scenarios import ALL_SCENARIOS, get_scenario_by_index
from prompts.trading_prompts import (
    TradingIntent, 
    ScenarioSelectionIntent,
    INTENT_CLASSIFICATION_PROMPT, 
    SCENARIO_SELECTION_PROMPT,
    TradingMessages
)


class TradingAgent:
    """
    Trading agent that generates executable trading requests using LLM structured output.
    """
    
    def __init__(self, llm: ChatOpenAI):
        """Initialize the TradingAgent."""
        self.llm = llm
        self.utils = TradingUtils(llm)
        
        # Structured LLM for intent classification
        self._structured_llm = llm.with_structured_output(TradingIntent).bind(temperature=0.0)
        self._scenario_llm = llm.with_structured_output(ScenarioSelectionIntent).bind(temperature=0.0)
        
        # Local parameters with defaults from config
        from trading.config import DEFAULT_REBALANCE_CONFIG
        config = DEFAULT_REBALANCE_CONFIG
        self._tax_weight = config.get('tax_weight', 1.0)
        self._ltcg_rate = config.get('ltcg_rate', 0.15)
        self._integer_shares = False  # default to fractional for flexibility
        
        # Selected scenario
        self._selected_scenario = None
        self._awaiting_scenario_selection = False
    
    def _get_status(self, state: AgentState, agent: str) -> Dict[str, bool]:
        """Get status tracking for a specific agent."""
        return state.get("status_tracking", {}).get(agent, {"done": False, "awaiting_input": False})
    
    def _set_status(self, state: AgentState, agent: str, done: bool = None, awaiting_input: bool = None) -> None:
        """Set status tracking for a specific agent."""
        if "status_tracking" not in state:
            state["status_tracking"] = {}
        if agent not in state["status_tracking"]:
            state["status_tracking"][agent] = {"done": False, "awaiting_input": False}
        
        if done is not None:
            state["status_tracking"][agent]["done"] = done
        if awaiting_input is not None:
            state["status_tracking"][agent]["awaiting_input"] = awaiting_input
    
    def _classify_intent(self, user_input: str) -> TradingIntent:
        """Classify user intent using LLM with structured output."""
        prompt = INTENT_CLASSIFICATION_PROMPT.format(user_input=user_input)
        
        try:
            intent = self._structured_llm.invoke(prompt)
            return intent
        except Exception as e:
            print(f"Error classifying intent: {e}")
            return TradingIntent(action="unknown")
    
    def _classify_scenario_selection(self, user_input: str) -> ScenarioSelectionIntent:
        """Classify scenario selection intent using LLM with structured output."""
        prompt = SCENARIO_SELECTION_PROMPT.format(user_input=user_input)
        
        try:
            intent = self._scenario_llm.invoke(prompt)
            return intent
        except Exception as e:
            print(f"Error classifying scenario selection: {e}")
            return ScenarioSelectionIntent(action="unknown")
    
    def _handle_scenario_selection(self, state: AgentState, user_input: str) -> AgentState:
        """Handle scenario selection from user using LLM structured output."""
        # Classify user intent
        intent = self._classify_scenario_selection(user_input)
        
        if intent.action == "select_scenario":
            if intent.scenario_number and 1 <= intent.scenario_number <= len(ALL_SCENARIOS):
                scenario = get_scenario_by_index(intent.scenario_number - 1)
                
                if scenario:
                    self._selected_scenario = scenario
                    self._awaiting_scenario_selection = False
                    
                    has_trades = bool(state.get("trading_requests"))
                    state["messages"].append({
                        "role": "ai",
                        "content": f"✅ Selected: **{scenario['name']}**\n\n"
                        f"• Account Value: ${scenario['account_value']:,}\n"
                        f"• Current Holdings: {len(scenario['holdings'])} positions\n\n"
                        f"{TradingMessages.intro_message(self._tax_weight, self._ltcg_rate, self._integer_shares, has_trades)}"
                    })
                else:
                    state["messages"].append({
                        "role": "ai",
                        "content": f"Invalid scenario number. Please enter 1-{len(ALL_SCENARIOS)}."
                    })
            else:
                state["messages"].append({
                    "role": "ai",
                    "content": f"Please enter a scenario number between 1 and {len(ALL_SCENARIOS)}."
                })
        
        elif intent.action == "custom_portfolio":
            state["messages"].append({
                "role": "ai",
                "content": "Custom portfolio input not yet implemented. Please select a demo scenario (1-6)."
            })
        
        else:  # unknown
            state["messages"].append({
                "role": "ai",
                "content": f"Please enter a number between 1 and {len(ALL_SCENARIOS)} or 'custom'."
            })
        
        return state
    
    def step(self, state: AgentState) -> AgentState:
        """
        Main step function for trading agent.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated agent state
        """
        # Initialize global state if first time
        status = self._get_status(state, "trading")
        if not status["awaiting_input"] and not status["done"]:
            self._set_status(state, "trading", awaiting_input=True, done=False)
        
        # Get investment portfolio (required for rebalancing)
        investment = state.get("investment", {})
        
        # Show scenario selection if not selected yet
        if not self._selected_scenario:
            if self._awaiting_scenario_selection:
                # Handle scenario selection using LLM
                last_user = state["messages"][-1].get("content", "")
                return self._handle_scenario_selection(state, last_user)
            else:
                # Show scenario selection prompt
                self._awaiting_scenario_selection = True
                self._set_status(state, "trading", awaiting_input=True)
                return self.utils.show_scenario_selection(state)
        
        # Only act on USER turns
        if not state.get("messages") or state["messages"][-1].get("role") != "user":
            return state
        
        last_user = state["messages"][-1].get("content", "")
        
        # Classify user intent
        intent = self._classify_intent(last_user)
        
        # Handle different intents
        if intent.action == "set_tax_weight":
            if intent.tax_weight is not None and intent.tax_weight > 0:
                self._tax_weight = intent.tax_weight
                state["messages"].append({
                    "role": "ai",
                    "content": TradingMessages.tax_weight_set_success(self._tax_weight, self._ltcg_rate, self._integer_shares)
                })
            else:
                state["messages"].append({
                    "role": "ai",
                    "content": TradingMessages.tax_weight_invalid()
                })
            self._set_status(state, "trading", awaiting_input=True)
            return state
        
        elif intent.action == "set_ltcg_rate":
            if intent.ltcg_rate is not None:
                if 0 <= intent.ltcg_rate <= 0.35:
                    self._ltcg_rate = intent.ltcg_rate
                    state["messages"].append({
                        "role": "ai",
                        "content": TradingMessages.ltcg_rate_set_success(self._ltcg_rate, self._tax_weight, self._integer_shares)
                    })
                else:
                    state["messages"].append({
                        "role": "ai",
                        "content": TradingMessages.ltcg_rate_invalid(intent.ltcg_rate)
                    })
            else:
                state["messages"].append({
                    "role": "ai",
                    "content": "Please specify a long-term capital gains rate between 0 and 0.35."
                })
            self._set_status(state, "trading", awaiting_input=True)
            return state
        
        elif intent.action == "set_integer_shares":
            if intent.integer_shares is not None:
                self._integer_shares = intent.integer_shares
                state["messages"].append({
                    "role": "ai",
                    "content": TradingMessages.integer_shares_set_success(self._integer_shares, self._tax_weight, self._ltcg_rate)
                })
            else:
                state["messages"].append({
                    "role": "ai",
                    "content": "Please specify whether to use integer shares (true) or fractional shares (false)."
                })
            self._set_status(state, "trading", awaiting_input=True)
            return state
        
        elif intent.action == "review":
            state["messages"].append({
                "role": "ai",
                "content": TradingMessages.review_configuration(self._tax_weight, self._ltcg_rate, self._integer_shares)
            })
            self._set_status(state, "trading", awaiting_input=True)
            return state
        
        elif intent.action == "run_rebalancing":
            # Execute rebalancing
            return self._execute_rebalancing(state, investment)
        
        elif intent.action == "proceed":
            # User wants to proceed to final review
            if state.get("trading_requests"):
                self._set_status(state, "trading", done=True, awaiting_input=False)
            else:
                has_trades = bool(state.get("trading_requests"))
                state["messages"].append({
                    "role": "ai",
                    "content": TradingMessages.intro_message(self._tax_weight, self._ltcg_rate, self._integer_shares, has_trades)
                })
                self._set_status(state, "trading", awaiting_input=True)
            return state
        
        else:  # unknown
            has_trades = bool(state.get("trading_requests"))
            state["messages"].append({
                "role": "ai",
                "content": TradingMessages.intro_message(self._tax_weight, self._ltcg_rate, self._integer_shares, has_trades)
            })
            self._set_status(state, "trading", awaiting_input=True)
            return state
    
    def _execute_rebalancing(self, state: AgentState, investment: Dict[str, Any]) -> AgentState:
        """Execute portfolio rebalancing using TradingUtils."""
        state = self.utils.execute_rebalancing(
            state,
            investment,
            self._tax_weight,
            self._ltcg_rate,
            self._integer_shares,
            self._selected_scenario
        )
        
        # After execution, ensure awaiting_input is set so user can respond
        if state.get("trading_requests"):
            self._set_status(state, "trading", done=False, awaiting_input=True)
        
        return state
    
    def router(self, state: AgentState) -> str:
        """Route based on trading agent state."""
        status = self._get_status(state, "trading")
        if status["awaiting_input"]:
            return "__end__"
        
        if state.get("trading_requests") and status["done"]:
            return "reviewer_agent"
        
        return "__end__"
