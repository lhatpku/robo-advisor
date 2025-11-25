"""
Reviewer utilities for generating summaries and managing reviewer logic
"""

from typing import Dict, Any, List, Optional, Tuple
from state import AgentState
from prompts.reviewer_prompts import REVIEWER_VALIDATION_PROMPTS


class ReviewerUtils:
    """Utility class for reviewer agent operations."""
    
    @staticmethod
    def generate_trading_summary(trading_requests: Dict[str, Any]) -> str:
        """
        Generate a summary of trading requests.
        
        Args:
            trading_requests: Dictionary containing trading request data
            
        Returns:
            Formatted trading summary string
        """
        if not trading_requests or not isinstance(trading_requests, dict):
            return "**Trading Summary:** No trading requests available"
        
        # Extract trading data
        trades = trading_requests.get("trading_requests", [])
        if not trades:
            return "**Trading Summary:** No trades generated"
        
        # Count different types of trades
        # Rebalancer uses "Side" (capital S), check both "Side" and "side" for compatibility
        buy_orders = [trade for trade in trades if trade.get("Side", "").upper() == "BUY" or trade.get("side", "").upper() == "BUY"]
        sell_orders = [trade for trade in trades if trade.get("Side", "").upper() == "SELL" or trade.get("side", "").upper() == "SELL"]
        
        total_trades = len(trades)
        buy_count = len(buy_orders)
        sell_count = len(sell_orders)
        
        # Calculate net cash flow
        net_cash_flow = 0.0
        for trade in trades:
            # Try "Side" first, then fallback to "side"
            side = trade.get("Side", trade.get("side", "")).upper()
            # Try "Price" first, then fallback to "price"
            price = float(trade.get("Price", trade.get("price", 0)))
            # Try "Shares" first, then fallback to "shares"
            shares = float(trade.get("Shares", trade.get("shares", 0)))
            trade_value = price * shares
            
            if side == "BUY":
                net_cash_flow -= trade_value  # Money going out
            elif side == "SELL":
                net_cash_flow += trade_value  # Money coming in
        
        return f"""**Trading Summary:**
**Total Trades:** {total_trades}
**Buy Orders:** {buy_count}
**Sell Orders:** {sell_count}
**Net Cash Flow:** ${net_cash_flow:,.2f}"""

    @staticmethod
    def generate_portfolio_summary(state: AgentState) -> str:
        """
        Generate a comprehensive summary of the entire portfolio process.
        
        Args:
            state: Current agent state containing all phase data
            
        Returns:
            Formatted portfolio summary string
        """
        summary_parts = []
        
        # Risk Assessment Summary
        risk = state.get("risk", {})
        if risk and "equity" in risk and "bond" in risk:
            equity_pct = risk["equity"] * 100
            bond_pct = risk["bond"] * 100
            summary_parts.append(f"**Risk Assessment:** {equity_pct:.1f}% Equity / {bond_pct:.1f}% Bonds")
        
        # Portfolio Construction Summary
        portfolio = state.get("portfolio", {})
        if portfolio and isinstance(portfolio, dict) and len(portfolio) > 0:
            # Filter numeric weights and convert to %
            numeric_items = [
                (asset, weight * 100)
                for asset, weight in portfolio.items()
                if isinstance(weight, (int, float)) and weight > 0
            ]
            if numeric_items:
                sorted_assets = sorted(numeric_items, key=lambda x: x[1], reverse=True)[:5]
                asset_summary = ", ".join([f"{asset}: {weight:.1f}%" for asset, weight in sorted_assets])
                summary_parts.append(f"**Portfolio Construction:** {asset_summary}")
        
        # Investment Selection Summary
        investment = state.get("investment", {})
        if investment and isinstance(investment, dict) and len(investment) > 0:
            # Count different types of investments
            fund_count = len([k for k, v in investment.items() if isinstance(v, dict) and "ticker" in v])
            summary_parts.append(f"**Investment Selection:** {fund_count} funds selected")
        
        # Trading Summary
        trading_requests = state.get("trading_requests", {})
        if trading_requests and isinstance(trading_requests, dict) and len(trading_requests) > 0:
            trading_summary = ReviewerUtils.generate_trading_summary(trading_requests)
            summary_parts.append(trading_summary)
        
        return "\n\n".join(summary_parts)

    @staticmethod
    def generate_final_completion_message(state: AgentState) -> str:
        """
        Generate the final completion message with portfolio summary.
        
        Args:
            state: Current agent state containing all phase data
            
        Returns:
            Complete final message string
        """
        portfolio_summary = ReviewerUtils.generate_portfolio_summary(state)
        
        return f"""🎉 **Portfolio Planning Complete!**

Congratulations! You have successfully completed all phases of the robo-advisor process:

✅ **Risk Assessment** - Your risk tolerance and asset allocation
✅ **Portfolio Construction** - Optimized asset class weights  
✅ **Investment Selection** - Specific funds and ETFs chosen
✅ **Trading Requests** - Ready-to-execute trading orders

---

## 📊 **Your Complete Portfolio Summary**

{portfolio_summary}

---

**What's Next?**

Type **"proceed"** to confirm acknowledgement of the summary."""

    @staticmethod
    def validate_phase_completion(state: AgentState, phase: str) -> Tuple[bool, str]:
        """
        Validate if a specific phase is complete.
        
        Args:
            state: Current agent state
            phase: Phase name to validate
            
        Returns:
            Tuple of (is_complete, feedback_message)
        """
        if phase == "risk":
            risk = state.get("risk")
            if risk and "equity" in risk and "bond" in risk:
                return True, REVIEWER_VALIDATION_PROMPTS["risk"]["complete"]
            return False, REVIEWER_VALIDATION_PROMPTS["risk"]["incomplete"]
        
        elif phase == "portfolio":
            portfolio = state.get("portfolio")
            if portfolio and isinstance(portfolio, dict) and len(portfolio) > 0:
                return True, REVIEWER_VALIDATION_PROMPTS["portfolio"]["complete"]
            return False, REVIEWER_VALIDATION_PROMPTS["portfolio"]["incomplete"]
        
        elif phase == "investment":
            investment = state.get("investment")
            if investment and isinstance(investment, dict) and len(investment) > 0:
                return True, REVIEWER_VALIDATION_PROMPTS["investment"]["complete"]
            return False, REVIEWER_VALIDATION_PROMPTS["investment"]["incomplete"]
        
        elif phase == "trading":
            trading = state.get("trading_requests")
            if trading and isinstance(trading, dict) and "trading_requests" in trading and len(trading.get("trading_requests", [])) > 0:
                return True, REVIEWER_VALIDATION_PROMPTS["trading"]["complete"]
            return False, REVIEWER_VALIDATION_PROMPTS["trading"]["incomplete"]
        
        return False, f"Unknown phase: {phase}"

    @staticmethod
    def get_next_phase(state: AgentState) -> Optional[str]:
        """
        Determine the next phase to proceed to.
        
        Args:
            state: Current agent state
            
        Returns:
            Next phase name or None if all complete
        """
        phases = ["risk", "portfolio", "investment", "trading"]
        
        for phase in phases:
            is_complete, _ = ReviewerUtils.validate_phase_completion(state, phase)
            if not is_complete:
                return phase
        
        return None  # All phases complete

    @staticmethod
    def find_first_incomplete_phase(state: AgentState, validation_results: Dict[str, Tuple[bool, str]]) -> Optional[str]:
        """
        Find the first incomplete phase that has been started.
        
        Args:
            state: Current agent state
            validation_results: Dictionary of phase validation results
            
        Returns:
            First incomplete phase name or None
        """
        incomplete_phases = [phase for phase, (is_complete, _) in validation_results.items() if not is_complete]
        
        if not incomplete_phases:
            return None
        
        # Find the first incomplete phase that has been started
        started_phases = []
        if state.get("risk"):
            started_phases.append("risk")
        if state.get("portfolio"):
            started_phases.append("portfolio")
        if state.get("investment"):
            started_phases.append("investment")
        if state.get("trading_requests"):
            started_phases.append("trading")
        
        # Find incomplete phases that have been started
        incomplete_started_phases = [phase for phase in incomplete_phases if phase in started_phases]
        
        if incomplete_started_phases:
            return incomplete_started_phases[0]
        
        return None

    @staticmethod
    def reset_state(state: AgentState) -> None:
        """
        Reset all state to initial values (for "start over" functionality).
        
        Args:
            state: Agent state to reset
        """
        state["risk"] = None
        state["portfolio"] = None
        state["investment"] = None
        state["trading_requests"] = None
        state["all_phases_complete"] = False
        state["ready_to_proceed"] = None
        state["next_phase"] = "risk"  # Reset to first phase
        state["intent_to_risk"] = False
        state["intent_to_portfolio"] = False
        state["intent_to_investment"] = False
        state["intent_to_trading"] = False
        state["entry_greeted"] = False
        state["summary_shown"] = {
            "risk": False,
            "portfolio": False,
            "investment": False,
            "trading": False
        }
        
        # Reset status tracking for all agents
        state["status_tracking"] = {
            "risk": {"done": False, "awaiting_input": False},
            "portfolio": {"done": False, "awaiting_input": False},
            "investment": {"done": False, "awaiting_input": False},
            "trading": {"done": False, "awaiting_input": False},
            "reviewer": {"done": False, "awaiting_input": False}
        }