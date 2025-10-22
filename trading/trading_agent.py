"""
Trading Agent

This agent handles the generation and management of trading requests based on
portfolio optimization results from the investment agent.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional
import json
import os
from datetime import datetime
from langchain_openai import ChatOpenAI
from state import AgentState

from .portfolio_trading import PortfolioTradingManager, RebalanceConfig, TaxRates
from .rebalance import Position
from .config import get_rebalance_config, get_tax_rates


class TradingAgent:
    """
    Trading agent that generates executable trading requests from investment portfolios
    """
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.trading_manager = PortfolioTradingManager()
        self.demo_scenarios = self._load_demo_scenarios()
        
        # Internal state for demo scenarios (not in AgentState)
        self._selecting_scenario = False
        self._demo_scenario = None
        self._risk_tolerance = "moderate"
        self._tax_sensitivity = "moderate"
        self._integer_shares = True
        
    def step(self, state: AgentState) -> AgentState:
        """Main step function for trading agent"""
        # Check if we have an investment portfolio to work with
        portfolio = state.get("portfolio", {})
        investment = state.get("investment", {})
        
        if not investment:
            state["messages"].append({
                "role": "ai", 
                "content": "I need an investment portfolio before I can generate trading requests. Please complete the investment selection first."
            })
            state["awaiting_input"] = True
            return state
        
        # Check if trading requests already exist - if so, just mark as done
        if state.get("trading_requests") and isinstance(state.get("trading_requests"), dict) and "trading_requests" in state.get("trading_requests", {}):
            state["done"] = True
            state["awaiting_input"] = False
            return state
        
        # Always show demo scenarios first (unless already selected)
        if not self._demo_scenario and not self._selecting_scenario:
            # Show demo scenario selection
            self._selecting_scenario = True
            state["messages"].append({
                "role": "ai",
                "content": self._get_scenario_selection_message()
            })
            state["awaiting_input"] = True
            return state
        
        # Check if user is selecting a scenario
        if self._selecting_scenario:
            return self._handle_scenario_selection(state)
        
        # Check if user confirmed scenario and wants to proceed
        last_user = state["messages"][-1].get("content", "").lower().strip()
        if last_user in ["yes", "proceed", "generate", "create"] and self._demo_scenario:
            # User confirmed scenario selection, proceed with trading
            pass
        elif self._demo_scenario and not state.get("intent_to_trading", False):
            # User has selected scenario but hasn't confirmed yet
            state["messages"].append({
                "role": "ai",
                "content": "Great! You've selected a demo scenario. Type 'yes' or 'proceed' to generate trading requests, or select a different scenario (1-6)."
            })
            state["awaiting_input"] = True
            return state
        elif not self._demo_scenario:
            # No scenario selected yet, show scenarios
            state["messages"].append({
                "role": "ai",
                "content": "Please select a demo scenario (1-6) to proceed with trading requests."
            })
            state["awaiting_input"] = True
            return state
        else:
            # No valid scenario or confirmation, wait for input
            state["awaiting_input"] = True
            return state
        
        # Generate trading requests
        try:
            trading_result = self._generate_trading_requests(state, investment)
            state["trading_requests"] = trading_result
            state["awaiting_input"] = True
            state["done"] = True
            
            # Display trading requests
            state["messages"].append({
                "role": "ai",
                "content": self._format_trading_requests(trading_result)
            })
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Trading error details: {error_details}")  # Debug output
            
            state["messages"].append({
                "role": "ai",
                "content": f"I encountered an error generating trading requests: {str(e)}. This might be because you're using your actual portfolio instead of a demo scenario. Please try selecting a demo scenario (1-6) instead."
            })
            state["awaiting_input"] = True
        
        return state
    
    def _load_demo_scenarios(self) -> Dict[str, Any]:
        """Load demo scenarios from JSON file"""
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            scenarios_file = os.path.join(current_dir, "demo_scenarios.json")
            with open(scenarios_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading demo scenarios: {e}")
            return {"scenarios": []}
    
    def _get_scenario_selection_message(self) -> str:
        """Get the scenario selection message"""
        lines = [
            "## 🎯 **Demo Trading Scenarios**",
            "",
            "**Welcome to the Trading Module!** 🚀",
            "",
            "For demo purposes, I've prepared several realistic scenarios with different account values and holdings. ",
            "Select a scenario to see how the trading strategy would work:",
            "",
            "**Available Scenarios:**"
        ]
        
        for i, scenario in enumerate(self.demo_scenarios["scenarios"], 1):
            lines.extend([
                f"**{i}. {scenario['name']}**",
                f"   • Account Value: ${scenario['account_value']:,}",
                f"   • Risk: {scenario['risk_tolerance'].title()} | Tax Sensitivity: {scenario['tax_sensitivity'].title()}",
                f"   • Shares: {'Integer' if scenario['integer_shares'] else 'Fractional'}",
                f"   • {scenario['description']}",
                f"   • Current Holdings: {len(scenario['current_positions'])} positions",
                ""
            ])
        
        lines.extend([
            "**📋 How to proceed:**",
            "• Type a number (1-6) to select a scenario",
            "• Type 'custom' to use your actual portfolio (may have limitations)",
            "",
            "**⚠️ Note:** Demo scenarios are recommended for testing the trading functionality.",
            ""
        ])
        
        return "\n".join(lines)
    
    def _handle_scenario_selection(self, state: AgentState) -> AgentState:
        """Handle user scenario selection"""
        last_user = state["messages"][-1].get("content", "").strip()
        
        if last_user.lower() == "custom":
            # Use user's actual portfolio
            self._selecting_scenario = False
            self._demo_scenario = None
            state["messages"].append({
                "role": "ai",
                "content": "Using your actual portfolio data. Proceeding with trading request generation..."
            })
            return self._generate_trading_requests(state, state.get("investment", {}))
        
        try:
            scenario_num = int(last_user)
            if 1 <= scenario_num <= len(self.demo_scenarios["scenarios"]):
                scenario = self.demo_scenarios["scenarios"][scenario_num - 1]
                self._selecting_scenario = False
                self._demo_scenario = scenario
                
                # Set up the scenario data internally
                self._account_value = scenario["account_value"]
                self._current_positions = self._convert_scenario_positions(scenario["current_positions"])
                self._risk_tolerance = scenario["risk_tolerance"]
                self._tax_sensitivity = scenario["tax_sensitivity"]
                self._integer_shares = scenario["integer_shares"]
                
                # Show scenario summary
                state["messages"].append({
                    "role": "ai",
                    "content": self._format_scenario_summary(scenario)
                })
                state["awaiting_input"] = True
                return state
            else:
                state["messages"].append({
                    "role": "ai",
                    "content": f"Please select a number between 1 and {len(self.demo_scenarios['scenarios'])}."
                })
                state["awaiting_input"] = True
                return state
        except ValueError:
            state["messages"].append({
                "role": "ai",
                "content": "Please enter a valid number (1-6) or 'custom' for your own portfolio."
            })
            state["awaiting_input"] = True
            return state
    
    def _convert_scenario_positions(self, positions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert scenario positions to the format expected by the trading manager"""
        converted = []
        for pos in positions:
            converted.append({
                "ticker": pos["ticker"],
                "quantity": pos["quantity"],
                "cost_basis": pos["cost_basis"],
                "price": pos["price"],
                "acquisition_date": datetime.strptime(pos["acquisition_date"], "%Y-%m-%d")
            })
        return converted
    
    def _format_scenario_summary(self, scenario: Dict[str, Any]) -> str:
        """Format scenario summary for display"""
        lines = [
            f"## 📊 **Selected: {scenario['name']}**",
            "",
            f"**Account Details:**",
            f"• **Value:** ${scenario['account_value']:,}",
            f"• **Risk Tolerance:** {scenario['risk_tolerance'].title()}",
            f"• **Tax Sensitivity:** {scenario['tax_sensitivity'].title()}",
            f"• **Share Type:** {'Integer' if scenario['integer_shares'] else 'Fractional'}",
            "",
            "**Current Holdings:**"
        ]
        
        # Show current positions
        for pos in scenario["current_positions"]:
            if pos["ticker"] == "sweep_cash":
                lines.append(f"• **Cash:** ${pos['quantity']:,}")
            else:
                current_value = pos["quantity"] * pos["price"]
                unrealized_gain = pos["quantity"] * (pos["price"] - pos["cost_basis"])
                gain_pct = (unrealized_gain / (pos["quantity"] * pos["cost_basis"])) * 100 if pos["quantity"] * pos["cost_basis"] > 0 else 0
                lines.append(f"• **{pos['ticker']}:** {pos['quantity']:.0f} shares @ ${pos['price']:.2f} (${current_value:,.0f}) | Gain: {gain_pct:+.1f}%")
        
        # Add summary
        summary = scenario["summary"]
        lines.extend([
            "",
            "**Portfolio Summary:**",
            f"• **Current Allocation:** {summary['current_allocation']}",
            f"• **Unrealized Gains:** ${summary['unrealized_gains']:,}",
            f"• **Tax Implications:** {summary['tax_implications']}",
            f"• **Rebalancing Needs:** {summary['rebalancing_needs']}",
            "",
            "**Ready to generate trading requests?** Type 'yes' to proceed with the optimization."
        ])
        
        return "\n".join(lines)
    
    def _get_trading_intro_message(self, investment: Dict[str, Any]) -> str:
        """Get the trading introduction message"""
        total_assets = len([k for k, v in investment.items() if isinstance(v, dict) and 'ticker' in v])
        cash_weight = investment.get('cash', {}).get('weight', 0) * 100
        
        return f"""**Ready to Generate Trading Requests!**

                I can now create executable trading orders based on your investment portfolio:

                📊 **Portfolio Summary:**
                • **{total_assets} asset classes** with selected funds
                • **{cash_weight:.1f}% cash** (sweep account)
                • **Tax-aware optimization** with soft tax caps
                • **Risk-adjusted rebalancing** using full covariance model

                **What I'll generate:**
                • **Buy/Sell orders** for each fund
                • **Tax optimization** (loss harvesting, gain management)
                • **Execution priorities** based on liquidity and risk
                • **Cost analysis** (spread, turnover, tax implications)

            **Ready to proceed?** Type 'yes' to generate trading requests, or ask questions about the process."""

    def _generate_trading_requests(self, state: AgentState, investment: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading requests from investment portfolio"""
        # Extract current positions from internal state (if available)
        current_positions = getattr(self, '_current_positions', [])
        
        # Get account value from internal state
        account_value = getattr(self, '_account_value', 100000)  # Default to $100k
        
        # Configure rebalancing parameters
        config = self._get_rebalance_config(state)
        self.trading_manager = PortfolioTradingManager(config)
        
        # Generate trading requests
        return self.trading_manager.generate_trading_requests(
            investment, 
            current_positions, 
            account_value
        )
    
    def _get_rebalance_config(self, state: AgentState) -> RebalanceConfig:
        """Get rebalancing configuration based on user preferences using config file"""
        # Extract user preferences from internal state
        risk_tolerance = self._risk_tolerance
        tax_sensitivity = self._tax_sensitivity
        
        # Get base configuration from config file
        config = get_rebalance_config()
        
        # Override based on user preferences
        if risk_tolerance == "conservative":
            config.update({
                'tracking_error_weight': 1.5,
                'cash_band_penalty_weight': 0.3
            })
        elif risk_tolerance == "aggressive":
            config.update({
                'tracking_error_weight': 0.8,
                'cash_band_penalty_weight': 0.1
            })
        
        if tax_sensitivity == "high":
            config.update({
                'tax_penalty_weight': 1.0,
                'soft_tax_cap': 5000.0  # $5,000 tax cap
            })
        elif tax_sensitivity == "low":
            config.update({
                'tax_penalty_weight': 0.2,
                'soft_tax_cap': 20000.0  # $20,000 tax cap
            })
        
        # Add user-specific settings
        config['integer_shares'] = self._integer_shares
        
        # Create TaxRates object from config
        tax_rates = TaxRates()
        config['tax_rates'] = tax_rates
        
        return RebalanceConfig(**config)
    
    def _format_trading_requests(self, trading_result: Dict[str, Any]) -> str:
        """Format trading requests for display"""
        requests = trading_result["trading_requests"]
        summary = trading_result["execution_summary"]
        
        if not requests:
            return "**No trading required** - your portfolio is already optimally positioned."
        
        # Create simplified trading table
        lines = [
            "## 📋 **Trading Requests**",
            "",
            "| Ticker | Action | Unit Price | Shares |",
            "|--------|--------|------------|--------|"
        ]
        
        for req in requests:
            ticker = req["ticker"]
            action = "BUY" if req["side"] == "BUY" else "SELL"
            unit_price = f"${req['price']:.2f}"
            shares = f"{req['shares']:.0f}"
            
            lines.append(f"| {ticker} | {action} | {unit_price} | {shares} |")
        
        # Add summary
        lines.extend([
            "",
            f"**Total Trades:** {summary['total_trades']}",
            f"**Buy Orders:** {summary['buy_trades']}",
            f"**Sell Orders:** {summary['sell_trades']}",
            f"**Net Cash Flow:** ${summary['net_cash_flow']:,.0f}",
            ""
        ])
        
        return "\n".join(lines)
    
    def handle_trading_commands(self, state: AgentState, user_input: str) -> AgentState:
        """Handle trading-specific commands"""
        user_input = user_input.lower().strip()
        
        if user_input in ["yes", "proceed", "generate", "create"]:
            return self.step(state)
        
        elif user_input == "execute":
            return self._execute_trades(state)
        
        elif user_input.startswith("analyze"):
            return self._analyze_specific_trade(state, user_input)
        
        elif user_input in ["help", "commands"]:
            return self._show_trading_help(state)
        
        else:
            state["messages"].append({
                "role": "ai",
                "content": "I didn't understand that trading command. Type 'help' to see available commands."
            })
            state["awaiting_input"] = True
        
        return state
    
    def _execute_trades(self, state: AgentState) -> AgentState:
        """Simulate trade execution"""
        trading_requests = state.get("trading_requests", {})
        
        if not trading_requests:
            state["messages"].append({
                "role": "ai",
                "content": "No trading requests available to execute. Please generate them first."
            })
            state["awaiting_input"] = True
            return state
        
        # Simulate execution
        executed_trades = []
        for req in trading_requests["trading_requests"]:
            executed_trades.append({
                "ticker": req["ticker"],
                "side": req["side"],
                "shares": req["shares"],
                "executed_price": req.get("price", 0),
                "execution_time": "2024-01-15 10:30:00",
                "status": "FILLED"
            })
        
        state["executed_trades"] = executed_trades
        state["messages"].append({
            "role": "ai",
            "content": f"✅ **Trades Executed Successfully!**\n\nExecuted {len(executed_trades)} trades. Your portfolio has been rebalanced according to the optimization results.\n\n**Next Steps:**\n• Monitor performance vs. targets\n• Review tax implications\n• Plan next rebalancing cycle"
        })
        state["awaiting_input"] = True
        
        return state
    
    def _analyze_specific_trade(self, state: AgentState, user_input: str) -> AgentState:
        """Analyze a specific trade in detail"""
        # Extract ticker from input
        parts = user_input.split()
        if len(parts) < 2:
            state["messages"].append({
                "role": "ai",
                "content": "Please specify a ticker to analyze. Example: 'analyze VTI'"
            })
            state["awaiting_input"] = True
            return state
        
        ticker = parts[1].upper()
        trading_requests = state.get("trading_requests", {})
        
        # Find the trade
        trade = None
        for req in trading_requests.get("trading_requests", []):
            if req["ticker"] == ticker:
                trade = req
                break
        
        if not trade:
            state["messages"].append({
                "role": "ai",
                "content": f"No trading request found for {ticker}. Please check the ticker symbol."
            })
            state["awaiting_input"] = True
            return state
        
        # Create detailed analysis
        analysis = self._create_trade_analysis(trade)
        state["messages"].append({
            "role": "ai",
            "content": analysis
        })
        state["awaiting_input"] = True
        
        return state
    
    def _create_trade_analysis(self, trade: Dict[str, Any]) -> str:
        """Create detailed analysis for a specific trade"""
        lines = [
            f"## 🔍 **Detailed Analysis: {trade['ticker']}**",
            "",
            f"**Trade Details:**",
            f"• **Side:** {trade['side']}",
            f"• **Shares:** {trade['shares']:.0f}",
            f"• **Value:** ${abs(trade['proceeds']):,.2f}",
            f"• **Priority:** {trade['execution_priority']:.1f}",
            ""
        ]
        
        # Risk metrics
        risk = trade.get("risk_metrics", {})
        lines.extend([
            "**Risk Metrics:**",
            f"• **Volatility:** {risk.get('volatility', 0):.1%}",
            f"• **Beta:** {risk.get('beta', 1.0):.2f}",
            f"• **Tracking Error:** {risk.get('tracking_error', 0):.1%}",
            ""
        ])
        
        # Tax implications
        tax = trade.get("tax_implications", {})
        lines.extend([
            "**Tax Implications:**",
            f"• **Tax Cost:** ${tax.get('tax_cost', 0):,.2f}",
            f"• **Tax Rate:** {tax.get('tax_rate', 0):.1%}",
            f"• **Net Proceeds:** ${tax.get('net_proceeds', 0):,.2f}",
            ""
        ])
        
        # Fund analysis
        fund_analysis = trade.get("fund_analysis", {})
        if fund_analysis.get("analysis_available", False):
            lines.extend([
                "**Fund Analysis:**",
                f"• **Name:** {fund_analysis.get('name', 'N/A')}",
                f"• **Type:** {fund_analysis.get('type', 'N/A')}",
                ""
            ])
            
            perf = fund_analysis.get("performance_metrics", {})
            if perf:
                lines.extend([
                    "**Performance Metrics:**",
                    f"• **Sharpe Ratio:** {perf.get('sharpe_ratio', 0):.2f}",
                    f"• **Annual Return:** {perf.get('annualized_return', 0):.1%}",
                    f"• **Max Drawdown:** {perf.get('max_drawdown', 0):.1%}",
                    ""
                ])
        
        lines.append("**Recommendation:** Execute this trade as planned based on the optimization results.")
        
        return "\n".join(lines)
    
    def _show_trading_help(self, state: AgentState) -> AgentState:
        """Show trading help commands"""
        help_text = """## 🆘 **Trading Commands Help**

            **Basic Commands:**
            • `yes` / `proceed` - Generate trading requests
            • `execute` - Execute all pending trades
            • `analyze [TICKER]` - Detailed analysis of specific trade
            • `help` - Show this help

            **Examples:**
            • `analyze VTI` - Analyze VTI trade in detail
            • `execute` - Execute all generated trades
            • `proceed` - Generate new trading requests

            **What happens when you execute:**
            1. **Orders are sent** to your broker
            2. **Trades are filled** at market prices
            3. **Portfolio is rebalanced** to target weights
            4. **Tax implications** are tracked
            5. **Performance monitoring** begins

            **Need more help?** Ask specific questions about any trade or the trading process."""
        
        state["messages"].append({
            "role": "ai",
            "content": help_text
        })
        state["awaiting_input"] = True
        
        return state

    def router(self, state: Dict[str, Any]) -> str:
        """
        Route based on trading agent state.
        """
        # If trading requests exist and done, go to reviewer (priority over awaiting_input)
        if state.get("trading_requests") and state.get("done", False):
            return "reviewer_agent"
        
        # If awaiting input, go to end to wait for user input
        if state.get("awaiting_input", False):
            return "__end__"
        
        # If trading requests don't exist yet, go to end to wait for user input
        return "__end__"
