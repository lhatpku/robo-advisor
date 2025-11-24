"""
Refactored Trading Agent

This agent handles trading request generation using LLM structured output
and the SoftObjectiveRebalancer for tax-aware rebalancing.
"""

from __future__ import annotations
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from state import AgentState
from utils.trading.trading_utils import TradingUtils
from utils.trading.trading_scenarios import ALL_SCENARIOS, get_scenario_by_index
from prompts.trading_prompts import (
    TradingIntent, 
    ScenarioSelectionIntent,
    INTENT_CLASSIFICATION_PROMPT, 
    SCENARIO_SELECTION_PROMPT,
    TradingMessages
)
from .base_agent import BaseAgent


class TradingAgent(BaseAgent):
    """
    Trading agent that generates executable trading requests using LLM structured output.
    """
    
    def __init__(self, llm: ChatOpenAI):
        """Initialize the TradingAgent."""
        super().__init__(llm, agent_name="trading")
        self.utils = TradingUtils(llm)
        
        # Structured LLM for intent classification
        self._structured_llm = llm.with_structured_output(TradingIntent).bind(temperature=0.0)
        self._scenario_llm = llm.with_structured_output(ScenarioSelectionIntent).bind(temperature=0.0)
        
        # Local parameters with defaults from config
        from utils.trading.config import DEFAULT_REBALANCE_CONFIG
        config = DEFAULT_REBALANCE_CONFIG
        self._tax_weight = config.get('tax_weight', 1.0)
        self._ltcg_rate = config.get('ltcg_rate', 0.15)
        self._integer_shares = False  # default to fractional for flexibility
        
        # Selected scenario
        self._selected_scenario = None
        self._awaiting_scenario_selection = False
    
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
                    self._add_message(state, "ai", 
                        f"✅ Selected: **{scenario['name']}**\n\n"
                        f"• Account Value: ${scenario['account_value']:,}\n"
                        f"• Current Holdings: {len(scenario['holdings'])} positions\n\n"
                        f"{TradingMessages.intro_message(self._tax_weight, self._ltcg_rate, self._integer_shares, has_trades)}"
                    )
                else:
                    self._add_message(state, "ai", f"Invalid scenario number. Please enter 1-{len(ALL_SCENARIOS)}.")
            else:
                self._add_message(state, "ai", f"Please enter a scenario number between 1 and {len(ALL_SCENARIOS)}.")
        
        elif intent.action == "custom_portfolio":
            self._add_message(state, "ai", "Custom portfolio input not yet implemented. Please select a demo scenario (1-6).")
        
        else:  # unknown
            self._add_message(state, "ai", f"Please enter a number between 1 and {len(ALL_SCENARIOS)} or 'custom'.")
        
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
        status = self._get_status(state)
        if not status["awaiting_input"] and not status["done"]:
            self._set_status(state, awaiting_input=True, done=False)
        
        # Get investment portfolio (required for rebalancing)
        investment = state.get("investment", {})
        
        # Show scenario selection if not selected yet
        if not self._selected_scenario:
            if self._awaiting_scenario_selection:
                # Handle scenario selection using LLM
                last_user = self._get_last_user_message(state)
                if last_user:
                    return self._handle_scenario_selection(state, last_user)
            else:
                # Show scenario selection prompt
                self._awaiting_scenario_selection = True
                self._set_status(state, awaiting_input=True)
                return self.utils.show_scenario_selection(state)
        
        # Only act on USER turns
        if not self._is_user_turn(state):
            return state
        
        last_user = self._get_last_user_message(state)
        if not last_user:
            return state
        
        # Classify user intent
        intent = self._classify_intent(last_user)
        
        # Handle different intents
        if intent.action == "set_tax_weight":
            if intent.tax_weight is not None and intent.tax_weight > 0:
                self._tax_weight = intent.tax_weight
                self._add_message(state, "ai", TradingMessages.tax_weight_set_success(self._tax_weight, self._ltcg_rate, self._integer_shares))
            else:
                self._add_message(state, "ai", TradingMessages.tax_weight_invalid())
            self._set_status(state, awaiting_input=True)
            return state
        
        elif intent.action == "set_ltcg_rate":
            if intent.ltcg_rate is not None:
                if 0 <= intent.ltcg_rate <= 0.35:
                    self._ltcg_rate = intent.ltcg_rate
                    self._add_message(state, "ai", TradingMessages.ltcg_rate_set_success(self._ltcg_rate, self._tax_weight, self._integer_shares))
                else:
                    self._add_message(state, "ai", TradingMessages.ltcg_rate_invalid(intent.ltcg_rate))
            else:
                self._add_message(state, "ai", "Please specify a long-term capital gains rate between 0 and 0.35.")
            self._set_status(state, awaiting_input=True)
            return state
        
        elif intent.action == "set_integer_shares":
            if intent.integer_shares is not None:
                self._integer_shares = intent.integer_shares
                self._add_message(state, "ai", TradingMessages.integer_shares_set_success(self._integer_shares, self._tax_weight, self._ltcg_rate))
            else:
                self._add_message(state, "ai", "Please specify whether to use integer shares (true) or fractional shares (false).")
            self._set_status(state, awaiting_input=True)
            return state
        
        elif intent.action == "review":
            self._add_message(state, "ai", TradingMessages.review_configuration(self._tax_weight, self._ltcg_rate, self._integer_shares))
            self._set_status(state, awaiting_input=True)
            return state
        
        elif intent.action == "run_rebalancing":
            # Execute rebalancing
            return self._execute_rebalancing(state, investment)
        
        elif intent.action == "proceed":
            # User wants to proceed to final review
            if state.get("trading_requests"):
                self._set_status(state, done=True, awaiting_input=False)
            else:
                has_trades = bool(state.get("trading_requests"))
                self._add_message(state, "ai", TradingMessages.intro_message(self._tax_weight, self._ltcg_rate, self._integer_shares, has_trades))
                self._set_status(state, awaiting_input=True)
            return state
        
        else:  # unknown
            has_trades = bool(state.get("trading_requests"))
            self._add_message(state, "ai", TradingMessages.intro_message(self._tax_weight, self._ltcg_rate, self._integer_shares, has_trades))
            self._set_status(state, awaiting_input=True)
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
            self._set_status(state, done=False, awaiting_input=True)
        
        return state
    
    def router(self, state: AgentState) -> str:
        """Route based on trading agent state."""
        status = self._get_status(state)
        if status["awaiting_input"]:
            return "__end__"
        
        if state.get("trading_requests") and status["done"]:
            return "reviewer_agent"
        
        return "__end__"

