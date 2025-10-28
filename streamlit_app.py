import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional
import json
import os
from dotenv import load_dotenv

# Import the existing app components
from app import build_graph
from state import AgentState
from langchain_openai import ChatOpenAI
# Load environment variables
load_dotenv()

# Install validators BEFORE importing get_guard (for Streamlit Cloud deployment)
if not os.path.exists(".guards_setup_complete"):
    try:
        import subprocess
        import sys
        subprocess.check_call([
            sys.executable, "-m", "guardrails", "hub", "install", 
            "hub://guardrails/unusual_prompt"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # Create marker file to indicate setup is complete
        with open(".guards_setup_complete", "w") as f:
            f.write("complete")
    except Exception:
        pass  # Continue even if installation fails (will try again next time)

# Now import get_guard (after validators are installed)
from guards import get_guard

# Configuration function
def get_config():
    """Get configuration from environment variables"""
    return {
        "api_key": os.getenv("OPENAI_API_KEY"),
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "temperature": float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
    }

# Page configuration
st.set_page_config(
    page_title="Robo-Advisor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498db;
        margin: 0.5rem 0;
    }
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-complete {
        background-color: #27ae60;
    }
    .status-pending {
        background-color: #f39c12;
    }
    .status-not-started {
        background-color: #95a5a6;
    }
    .message-container {
        max-height: 400px;
        overflow-y: auto;
        border: 1px solid #ddd;
        border-radius: 0.5rem;
        padding: 1rem;
        background-color: #f8f9fa;
    }
    .user-message {
        background-color: #e3f2fd;
        color: #000000;
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #2196f3;
    }
    .ai-message {
        background-color: #f1f8e9;
        color: #000000;
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'state' not in st.session_state:
        st.session_state.state = {
            "messages": [],
            "answers": {},
            "risk": None,
            "intent_to_risk": False,
            "entry_greeted": False,
            "intent_to_portfolio": False,
            "intent_to_investment": False,
            "intent_to_trading": False,
            "portfolio": None,
            "investment": None,
            "trading_requests": None,
            "ready_to_proceed": None,
            "all_phases_complete": False,
            "next_phase": "risk",
            "summary_shown": {
                "risk": False,
                "portfolio": False,
                "investment": False,
                "trading": False
            },
            "status_tracking": {
                "risk": {"done": False, "awaiting_input": False},
                "portfolio": {"done": False, "awaiting_input": False},
                "investment": {"done": False, "awaiting_input": False},
                "trading": {"done": False, "awaiting_input": False},
                "reviewer": {"done": False, "awaiting_input": False}
            }
        }
    
    if 'graph' not in st.session_state:
        # Initialize the graph
        config = get_config()
        
        # Check if API key is provided
        if not config["api_key"]:
            st.error("❌ OpenAI API key not found! Please set the OPENAI_API_KEY environment variable.")
            st.stop()
        
        llm = ChatOpenAI(
            api_key=config["api_key"],
            model=config["model"],
            temperature=config["temperature"],
        )
        st.session_state.graph = build_graph(llm)
    
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False

def reset_app():
    """Reset the application to initial state"""
    st.session_state.state = {
        "messages": [],
        "answers": {},
        "risk": None,
        "intent_to_risk": False,
        "entry_greeted": False,
        "intent_to_portfolio": False,
        "intent_to_investment": False,
        "intent_to_trading": False,
        "portfolio": None,
        "investment": None,
        "trading_requests": None,
        "ready_to_proceed": None,
        "all_phases_complete": False,
        "next_phase": "risk",
        "summary_shown": {
            "risk": False,
            "portfolio": False,
            "investment": False,
            "trading": False
        },
        "status_tracking": {
            "risk": {"done": False, "awaiting_input": False},
            "portfolio": {"done": False, "awaiting_input": False},
            "investment": {"done": False, "awaiting_input": False},
            "trading": {"done": False, "awaiting_input": False},
            "reviewer": {"done": False, "awaiting_input": False}
        }
    }
    st.session_state.initialized = False
    st.rerun()

def display_risk_assessment(state: AgentState):
    """Display risk assessment results and questionnaire answers"""
    if not state.get("risk") and not state.get("answers"):
        return
    
    st.markdown('<div class="section-header">🎯 Risk Assessment</div>', unsafe_allow_html=True)
    
    # Display risk allocation if available
    if state.get("risk"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Risk Allocation:**")
            equity = state["risk"].get("equity", 0)
            bond = state["risk"].get("bond", 0)
            
            # Create a simple bar chart for risk allocation
            fig = go.Figure(data=[
                go.Bar(name='', x=['Equity', 'Bonds'], y=[equity, bond], 
                      marker_color=['#2ecc71', '#3498db'])
            ])
            fig.update_layout(
                title="Asset Allocation",
                yaxis_title="Percentage",
                height=300,
                showlegend=False
            )
            st.plotly_chart(fig, config={'displayModeBar': False}, width='stretch')
        
        with col2:
            st.markdown("**Allocation Details:**")
            st.metric("Equity Allocation", f"{equity:.1%}")
            st.metric("Bond Allocation", f"{bond:.1%}")
    
    # Display questionnaire answers if available
    if state.get("answers"):
        st.markdown("**Questionnaire Answers:**")
        
        # Create a collapsible section for answers
        with st.expander("View Questionnaire Responses", expanded=False):
            for qid, answer in state["answers"].items():
                if isinstance(answer, dict) and "selected_label" in answer:
                    st.write(f"**{qid.upper()}:** {answer['selected_label']}")
                    if "raw_user_text" in answer and answer["raw_user_text"]:
                        st.caption(f"User input: \"{answer['raw_user_text']}\"")

def display_portfolio(state: AgentState):
    """Display portfolio allocation with pie chart and table"""
    if not state.get("portfolio"):
        return
    
    st.markdown('<div class="section-header">📊 Portfolio Allocation</div>', unsafe_allow_html=True)
    
    portfolio = state["portfolio"]
    
    # Extract portfolio weights
    if isinstance(portfolio, dict):
        weights = {k: v for k, v in portfolio.items() if isinstance(v, (int, float)) and v > 0}
        
        if weights:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Create pie chart
                labels = list(weights.keys())
                values = list(weights.values())
                
                fig = go.Figure(data=[go.Pie(name='', labels=labels, values=values, hole=0.3)])
                fig.update_layout(
                    title="Portfolio Allocation",
                    height=400,
                    showlegend=True
                )
                st.plotly_chart(fig, config={'displayModeBar': False}, width='stretch')
            
            with col2:
                # Display portfolio table
                st.markdown("**Allocation Details:**")
                df = pd.DataFrame(list(weights.items()), columns=['Asset Class', 'Weight'])
                df['Weight'] = df['Weight'].apply(lambda x: f"{x:.1%}")
                st.dataframe(df, width='stretch', hide_index=True)
                
                # Display portfolio parameters if available
                if "lambda" in portfolio:
                    st.metric("Risk Aversion (λ)", f"{portfolio['lambda']:.2f}")
                if "cash_reserve" in portfolio:
                    st.metric("Cash Reserve", f"{portfolio['cash_reserve']:.1%}")

def display_investment(state: AgentState):
    """Display investment fund selections"""
    investment = state.get("investment")
    if not investment or not isinstance(investment, dict):
        return
    
    st.markdown('<div class="section-header">💼 Investment Selection</div>', unsafe_allow_html=True)
    
    if isinstance(investment, dict):
        # Create a comprehensive table showing asset class, fund details, and weights
        # The investment data structure is: {asset_class: {weight, ticker, analysis, selection_reason, criteria_used}}
        
        # Prepare data for the table
        table_data = []
        
        for asset_class, fund_info in investment.items():
            if isinstance(fund_info, dict) and "ticker" in fund_info:
                table_data.append({
                    "Asset Class": asset_class.replace('_', ' ').title(),
                    "Fund Symbol": fund_info.get('ticker', 'N/A'),
                    "Weight": f"{fund_info.get('weight', 0):.1%}",
                    "Selection Reason": fund_info.get('selection_reason', 'N/A'),
                    "Criteria Used": fund_info.get('criteria_used', 'N/A')
                })
        
        if table_data:
            # Display the main table
            df = pd.DataFrame(table_data)
            st.dataframe(df, width='stretch', hide_index=True)
            
            # Display additional details in expandable sections
            st.markdown("**Detailed Fund Information:**")
            for asset_class, fund_info in investment.items():
                if isinstance(fund_info, dict) and "ticker" in fund_info:
                    with st.expander(f"{asset_class.replace('_', ' ').title()} - {fund_info.get('ticker', 'N/A')}", expanded=False):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Ticker:** {fund_info.get('ticker', 'N/A')}")
                            st.write(f"**Weight:** {fund_info.get('weight', 0):.1%}")
                            st.write(f"**Selection Reason:** {fund_info.get('selection_reason', 'N/A')}")
                        
                        with col2:
                            st.write(f"**Criteria Used:** {fund_info.get('criteria_used', 'N/A')}")
                            if "analysis" in fund_info and fund_info["analysis"]:
                                st.write(f"**Analysis:** {fund_info['analysis']}")

def display_trading_requests(state: AgentState):
    """Display trading requests in table format"""
    if not state.get("trading_requests"):
        return
    
    st.markdown('<div class="section-header">📈 Trading Requests</div>', unsafe_allow_html=True)
    
    trading_requests = state["trading_requests"]
    
    if isinstance(trading_requests, dict) and "trading_requests" in trading_requests:
        requests = trading_requests["trading_requests"]
        
        if isinstance(requests, list) and requests:
            # Convert to DataFrame for better display
            df_data = []
            for req in requests:
                if isinstance(req, dict):
                    df_data.append({
                        "Side": req.get("Side", req.get("side", "N/A")),
                        "Ticker": req.get("Ticker", req.get("ticker", "N/A")),
                        "Shares": req.get("Shares", req.get("shares", "N/A")),
                        "Price": f"${req.get('Price', req.get('price', 0)):.2f}",
                        "Proceeds": f"${req.get('Proceeds', req.get('proceeds', 0)):.2f}",
                        "Realized Gain": f"${req.get('RealizedGain', req.get('realized_gain', 0)):.2f}"
                    })
            
            if df_data:
                df = pd.DataFrame(df_data)
                st.dataframe(df, width='stretch', hide_index=True)
        
        # Display summary information
        if "summary" in trading_requests:
            summary = trading_requests["summary"]
            if isinstance(summary, dict):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if "total_value" in summary:
                        st.metric("Total Value", f"${summary['total_value']:,.2f}")
                
                with col2:
                    if "num_trades" in summary:
                        st.metric("Number of Trades", summary["num_trades"])
                
                with col3:
                    if "estimated_cost" in summary:
                        st.metric("Estimated Cost", f"${summary['estimated_cost']:,.2f}")

def display_status_tracking(state: AgentState):
    """Display the current status of each phase"""
    st.markdown('<div class="section-header">📋 Process Status</div>', unsafe_allow_html=True)
    
    status_tracking = state.get("status_tracking", {})
    
    phases = [
        ("risk", "🎯 Risk Assessment", "Complete risk questionnaire to determine your risk tolerance and investment horizon"),
        ("portfolio", "📊 Portfolio Optimization", "Optimize asset allocation based on your risk profile and preferences"),
        ("investment", "💼 Fund Selection", "Select specific funds and ETFs for each asset class in your portfolio"),
        ("trading", "📈 Trading Requests", "Generate executable trading orders to implement your investment strategy")
    ]
    
    cols = st.columns(len(phases))
    
    for i, (phase, label, description) in enumerate(phases):
        with cols[i]:
            status = status_tracking.get(phase, {"done": False, "awaiting_input": False})
            
            if status["done"]:
                status_class = "status-complete"
                status_text = "Complete"
            elif status["awaiting_input"]:
                status_class = "status-pending"
                status_text = "In Progress"
            else:
                status_class = "status-not-started"
                status_text = "Not Started"
            
            st.markdown(f"""
            <div class="metric-card">
                <div>
                    <span class="status-indicator {status_class}"></span>
                    <strong>{label}</strong>
                </div>
                <div style="margin-top: 0.5rem; color: #666;">
                    {status_text}
                </div>
                <div style="margin-top: 0.5rem; font-size: 0.85rem; color: #888;">
                    {description}
                </div>
            </div>
            """, unsafe_allow_html=True)

def display_messages(state: AgentState):
    """Display message history in a collapsible section"""
    messages = state.get("messages", [])
    
    if not messages:
        return
    
    st.markdown('<div class="section-header">💬 Conversation History</div>', unsafe_allow_html=True)
    
    # Limit to last 15 messages for better performance
    recent_messages = messages[-15:] if len(messages) > 15 else messages
    
    with st.expander(f"View Message History ({len(recent_messages)} of {len(messages)} messages)", expanded=False):
        st.markdown('<div class="message-container">', unsafe_allow_html=True)
        
        for i, message in enumerate(recent_messages):
            role = message.get("role", "")
            content = message.get("content", "")
            
            if role == "user":
                st.markdown(f'<div class="user-message"><strong>You:</strong> {content}</div>', unsafe_allow_html=True)
            elif role == "ai":
                st.markdown(f'<div class="ai-message"><strong>AI:</strong> {content}</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Show message count info
        if len(messages) > 15:
            st.caption(f"Showing last 15 messages. Total conversation: {len(messages)} messages.")

def main():
    """Main Streamlit application"""
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<div class="main-header">🤖 Robo-Advisor</div>', unsafe_allow_html=True)
    
    # Sidebar with reset button
    with st.sidebar:
        st.markdown("### Controls")
        if st.button("🔄 Reset Application", type="primary", width='stretch'):
            reset_app()
        
        st.markdown("### Current Status")
        status_tracking = st.session_state.state.get("status_tracking", {})
        
        # Only count the 4 main phases (exclude reviewer)
        main_phases = ["risk", "portfolio", "investment", "trading"]
        completed_phases = sum(1 for phase in main_phases if status_tracking.get(phase, {}).get("done", False))
        total_phases = len(main_phases)
        
        st.progress(completed_phases / total_phases if total_phases > 0 else 0)
        st.caption(f"Completed: {completed_phases}/{total_phases} phases")
    
    # Initialize the app if not done
    if not st.session_state.initialized:
        st.session_state.state = st.session_state.graph.invoke(st.session_state.state)
        st.session_state.initialized = True
        st.rerun()
    
    # Main content area - side by side layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Chat header
        st.markdown("### 💬 Chat with the Robo-Advisor")
        
        # Display current AI response FIRST (above input box)
        messages = st.session_state.state.get("messages", [])
        if messages:
            last_ai_message = None
            for message in reversed(messages):
                if message.get("role") == "ai":
                    last_ai_message = message
                    break
            
            if last_ai_message:
                st.markdown("### 🤖 AI Assistant")
                st.markdown(last_ai_message["content"])
        
        # Message input BELOW the AI response
        # Use a form to handle input properly
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_input(
                "Type your message here:",
                placeholder="Ask about risk assessment, portfolio optimization, or investment selection...",
                key="user_input"
            )
            
            submitted = st.form_submit_button("Send", type="primary")
            
            if submitted and user_input:
                # Initialize guard if not already done
                if 'guard' not in st.session_state:
                    st.session_state.guard = get_guard()
                
                # Validate user input for prompt injection attempts
                is_safe, error_msg = st.session_state.guard.validate(user_input)
                if not is_safe:
                    # Add AI message to chat explaining the issue
                    st.session_state.state["messages"].append({
                        "role": "ai",
                        "content": f"⚠️ {error_msg}\n\nPlease try rephrasing your message in a different way."
                    })
                    st.rerun()
                    return
                
                # Add user message to state
                st.session_state.state["messages"].append({"role": "user", "content": user_input})
                
                # Process through the graph
                st.session_state.state = st.session_state.graph.invoke(st.session_state.state)
                
                # Rerun to refresh the UI
                st.rerun()
        
        # Display status tracking below the input
        display_status_tracking(st.session_state.state)
        
        # Display messages in collapsible section
        st.markdown("---")
        display_messages(st.session_state.state)
    
    with col2:
        # Display all data sections (charts and tables)
        display_risk_assessment(st.session_state.state)
        st.markdown("---")
        display_portfolio(st.session_state.state)
        st.markdown("---")
        display_investment(st.session_state.state)
        st.markdown("---")
        display_trading_requests(st.session_state.state)

if __name__ == "__main__":
    main()
