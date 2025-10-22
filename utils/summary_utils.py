"""
Summary utilities for generating final portfolio summaries
"""

from typing import Dict, Any, List, Optional
from state import AgentState


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
    buy_orders = [trade for trade in trades if trade.get("side", "").upper() == "BUY"]
    sell_orders = [trade for trade in trades if trade.get("side", "").upper() == "SELL"]
    
    total_trades = len(trades)
    buy_count = len(buy_orders)
    sell_count = len(sell_orders)
    
    # Calculate net cash flow
    net_cash_flow = 0.0
    for trade in trades:
        side = trade.get("side", "").upper()
        price = float(trade.get("price", 0))
        shares = float(trade.get("shares", 0))
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
        # Get top 5 asset classes by weight
        sorted_assets = sorted(portfolio.items(), key=lambda x: x[1], reverse=True)[:5]
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
        trading_summary = generate_trading_summary(trading_requests)
        summary_parts.append(trading_summary)
    
    return "\n\n".join(summary_parts)


def generate_final_completion_message(state: AgentState) -> str:
    """
    Generate the final completion message with portfolio summary.
    
    Args:
        state: Current agent state containing all phase data
        
    Returns:
        Complete final message string
    """
    portfolio_summary = generate_portfolio_summary(state)
    
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
Your personalized investment plan is now ready! You can:
• **Review** any phase by saying "review [phase name]"
• **Start over** with a new portfolio by saying "start over"
• **Exit** the application

Type **"proceed"** to confirm completion and return to the main menu."""
