import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional
import json
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import logging

# Import the existing app components
from app import build_graph
from state import AgentState
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Import guard and monitoring
from operation.guards import get_guard
from operation.healthcheck import CompositeHealthCheck, OpenAIHealthCheck, YFinanceHealthCheck, HealthStatus
from operation.monitoring.metrics import get_metrics_registry
from operation.logging.logging_config import get_logger, get_correlation_id

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
    page_title="AI Robo-Advisor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern CSS styling
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #1e3a8a;
        --secondary-color: #3b82f6;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --danger-color: #ef4444;
        --bg-color: #f8fafc;
        --card-bg: #ffffff;
        --text-primary: #1f2937;
        --text-secondary: #6b7280;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1.1rem;
    }
    
    /* Status cards - compact */
    .status-card {
        background: white;
        padding: 0.5rem 0.75rem;
        border-radius: 0.5rem;
        border-left: 3px solid;
        margin: 0.25rem 0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s;
    }
    
    .status-card:hover {
        transform: translateY(-1px);
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .status-complete {
        border-left-color: #10b981;
    }
    
    .status-pending {
        border-left-color: #f59e0b;
    }
    
    .status-not-started {
        border-left-color: #9ca3af;
    }
    
    /* Message cards - clean design without white box */
    .ai-message-card {
        background: transparent;
        padding: 0;
        margin: 1.5rem 0;
    }
    
    /* Horizontal line separator */
    .ai-message-card::before {
        content: '';
        display: block;
        width: 100%;
        height: 2px;
        background: linear-gradient(90deg, transparent 0%, #667eea 20%, #764ba2 80%, transparent 100%);
        margin-bottom: 1.25rem;
        border-radius: 1px;
    }
    
    .user-message-card {
        background: transparent;
        padding: 0;
        margin: 1rem 0;
    }
    
    .user-message-card::before {
        content: '';
        display: block;
        width: 100%;
        height: 1px;
        background: linear-gradient(90deg, transparent 0%, #9ca3af 50%, transparent 100%);
        margin-bottom: 1rem;
    }
    
    /* Enhanced message header styling - prominent card design */
    .message-header {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 1rem;
        font-weight: 700;
        font-size: 1.05rem;
        padding: 0.75rem 1.25rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 0.75rem;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.25);
        letter-spacing: 0.3px;
    }
    
    .user-message-header {
        background: linear-gradient(135deg, #e5e7eb 0%, #d1d5db 100%);
        color: #374151;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Markdown content styling within cards - target Streamlit's markdown containers */
    .ai-message-card > div[data-testid="stMarkdownContainer"] {
        margin: 0;
    }
    
    .ai-message-card p {
        margin: 0.5rem 0;
        color: #1f2937;
        line-height: 1.7;
    }
    
    .ai-message-card strong {
        color: #667eea;
        font-weight: 600;
    }
    
    .ai-message-card ul,
    .ai-message-card ol {
        margin: 0.75rem 0;
        padding-left: 1.5rem;
    }
    
    .ai-message-card li {
        margin: 0.4rem 0;
        color: #374151;
    }
    
    .ai-message-card code {
        background: #e0e7ff;
        color: #667eea;
        padding: 0.2rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.9em;
        font-weight: 500;
    }
    
    .user-message-card > div[data-testid="stMarkdownContainer"] {
        margin: 0;
    }
    
    .user-message-card p {
        margin: 0.5rem 0;
        color: #1f2937;
        line-height: 1.7;
    }
    
    .user-message-card strong {
        color: #374151;
        font-weight: 600;
    }
    
    .user-message-card ul,
    .user-message-card ol {
        margin: 0.75rem 0;
        padding-left: 1.5rem;
    }
    
    .user-message-card li {
        margin: 0.4rem 0;
        color: #4b5563;
    }
    
    .user-message-card code {
        background: #e5e7eb;
        color: #1f2937;
        padding: 0.2rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.9em;
    }
    
    /* Health status indicators */
    .health-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .health-healthy {
        background-color: #10b981;
        box-shadow: 0 0 8px rgba(16, 185, 129, 0.5);
    }
    
    .health-degraded {
        background-color: #f59e0b;
        box-shadow: 0 0 8px rgba(245, 158, 11, 0.5);
    }
    
    .health-unhealthy {
        background-color: #ef4444;
        box-shadow: 0 0 8px rgba(239, 68, 68, 0.5);
    }
    
    /* Sidebar styling - compact */
    .sidebar-section {
        background: white;
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    /* Compact sidebar headers */
    .sidebar-section h3 {
        margin-top: 0;
        margin-bottom: 0.5rem;
        font-size: 1rem;
    }
    
    /* Override status card padding for compact sidebar */
    .sidebar-section .status-card {
        padding: 0.5rem 0.75rem;
        margin: 0.25rem 0;
    }
    
    .sidebar-section .status-card strong {
        font-size: 0.9rem;
    }
    
    /* Progress bar enhancement */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    # Initialize logging
    from operation.logging.logging_config import setup_logging, set_correlation_id
    import uuid
    setup_logging(level=os.getenv("LOG_LEVEL", "INFO"))
    
    if 'state' not in st.session_state:
        # Generate correlation ID for this session
        correlation_id = str(uuid.uuid4())
        set_correlation_id(correlation_id)
        
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
            },
            "correlation_id": correlation_id
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
        st.session_state.llm = llm  # Store for health checks
    
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

def render_status_bar(state: AgentState):
    """Render a well-designed status bar in the sidebar"""
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### 📊 Progress Status")
    
    status_tracking = state.get("status_tracking", {})
    phases = [
        ("risk", "🎯", "Risk"),
        ("portfolio", "📊", "Portfolio"),
        ("investment", "💼", "Investment"),
        ("trading", "📈", "Trading")
    ]
    
    # Calculate progress
    completed = sum(1 for phase, _, _ in phases if status_tracking.get(phase, {}).get("done", False))
    total = len(phases)
    progress = completed / total if total > 0 else 0
    
    # Compact progress bar
    st.progress(progress)
    st.caption(f"{completed}/{total} completed")
    
    # Phase status cards - more compact
    for phase, icon, name in phases:
        status = status_tracking.get(phase, {"done": False, "awaiting_input": False})
        
        if status["done"]:
            status_class = "status-complete"
            status_text = "✓"
        elif status["awaiting_input"]:
            status_class = "status-pending"
            status_text = "●"
        else:
            status_class = "status-not-started"
            status_text = "○"
        
        st.markdown(f"""
        <div class="status-card {status_class}">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <span style="font-size: 16px;">{icon}</span>
                <span style="font-size: 0.85rem; font-weight: 500;">{name}</span>
                <span style="font-size: 0.75rem; color: #6b7280;">{status_text}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

@st.cache_data(ttl=30, show_spinner=False)  # Cache for 30 seconds
def _run_health_checks_cached(llm: ChatOpenAI) -> Dict[str, Any]:
    """
    Run health checks with caching. Results are cached for 30 seconds.
    This prevents blocking the UI on every render.
    """
    try:
        checks = [
            OpenAIHealthCheck(llm),
            YFinanceHealthCheck()
        ]
        composite = CompositeHealthCheck(checks)
        results = composite.check_all()
        overall = composite.get_overall_status()
        
        return {
            "results": results,
            "overall": overall,
            "status": "completed"
        }
    except Exception as e:
        return {
            "error": str(e),
            "status": "error"
        }

def render_health_check(llm: ChatOpenAI):
    """
    Render health check status in sidebar with caching.
    
    Health checks are cached for 30 seconds to avoid blocking the UI.
    Set ENABLE_HEALTH_CHECKS=false to disable health checks entirely.
    """
    # Check if health checks are disabled
    enable_health_checks = os.getenv("ENABLE_HEALTH_CHECKS", "true").lower() == "true"
    if not enable_health_checks:
        return  # Skip health checks entirely
    
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### 🏥 System Health")
    
    try:
        # Get cached health check results (will only run if cache expired)
        # Cache TTL is 30 seconds - see @st.cache_data decorator
        health_data = _run_health_checks_cached(llm)
        
        if health_data.get("status") == "completed":
            results = health_data.get("results", {})
            overall = health_data.get("overall")
            show_health_status(results, overall)
        else:
            st.caption(f"Health: {health_data.get('error', 'Unknown error')}")
    except Exception as e:
        st.caption(f"Health: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_health_status(results: Dict, overall: HealthStatus):
    """Helper to display health status"""
    status_colors = {
        HealthStatus.HEALTHY: ("#10b981", "healthy"),
        HealthStatus.DEGRADED: ("#f59e0b", "degraded"),
        HealthStatus.UNHEALTHY: ("#ef4444", "unhealthy")
    }
    color, status_class = status_colors.get(overall, ("#9ca3af", "unknown"))
    
    st.markdown(f"""
    <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
        <span class="health-indicator health-{status_class}"></span>
        <span style="font-size: 0.9rem; font-weight: 500;">{overall.value.title()}</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Individual checks - compact
    with st.expander("Details", expanded=False):
        for name, result in results.items():
            check_color, check_class = status_colors.get(result.status, ("#9ca3af", "unknown"))
            response_time = f" {result.response_time_ms:.0f}ms" if result.response_time_ms else ""
            st.markdown(f"""
            <div style="margin: 0.25rem 0; font-size: 0.85rem;">
                <span class="health-indicator health-{check_class}"></span>
                <strong>{name.title()}:</strong> {result.status.value}{response_time}
            </div>
            """, unsafe_allow_html=True)

def render_monitoring():
    """Render monitoring metrics in sidebar"""
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### 📈 Performance Metrics")
    
    try:
        metrics = get_metrics_registry()
        all_metrics = metrics.get_all_metrics()
        
        # Display key metrics - compact
        if all_metrics:
            # Counters
            counters = {k.replace("counter_", ""): v for k, v in all_metrics.items() if k.startswith("counter_")}
            if counters:
                with st.expander("Counters", expanded=False):
                    for name, value in list(counters.items())[:3]:  # Show top 3
                        st.write(f"**{name.replace('_', ' ').title()}:** {int(value)}")
            
            # Timers
            timers = {k.replace("timer_", ""): v for k, v in all_metrics.items() if k.startswith("timer_")}
            if timers:
                with st.expander("Performance", expanded=False):
                    for name, stats in list(timers.items())[:2]:  # Show top 2
                        if isinstance(stats, dict) and "mean" in stats:
                            st.write(f"**{name.replace('_', ' ').title()}:** {stats['mean']*1000:.0f}ms")
        else:
            st.caption("No metrics yet")
    except Exception as e:
        st.caption(f"Metrics: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_product_overview():
    """Render product overview at the top"""
    st.markdown("""
    <div class="main-header">
        <h1>🤖 AI Robo-Advisor</h1>
        <p>Your intelligent investment planning assistant powered by AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature highlights
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("**🎯 Risk Assessment**")
        st.caption("Personalized risk profiling")
    with col2:
        st.markdown("**📊 Portfolio Optimization**")
        st.caption("Mean-variance optimization")
    with col3:
        st.markdown("**💼 Fund Selection**")
        st.caption("AI-powered fund analysis")
    with col4:
        st.markdown("**📈 Trade Execution**")
        st.caption("Ready-to-execute orders")

# Removed escape_html_for_display - we'll use native markdown rendering instead

def render_chat_messages(state: AgentState):
    """Render chat messages with AI on top"""
    messages = state.get("messages", [])
    
    if not messages:
        st.info("👋 Welcome! Start by saying 'hello' or 'proceed' to begin your investment planning journey.")
        return
    
    # Get last AI message (most recent)
    last_ai = None
    for msg in reversed(messages):
        if msg.get("role") == "ai":
            last_ai = msg
            break
    
    # Display last AI message prominently
    if last_ai:
        content = last_ai.get("content", "")
        # Render in modern card design
        st.markdown('<div class="ai-message-card">', unsafe_allow_html=True)
        st.markdown('<div class="message-header">🤖 AI Assistant</div>', unsafe_allow_html=True)
        st.markdown(content)  # Native markdown rendering
        st.markdown('</div>', unsafe_allow_html=True)

def render_conversation_history(state: AgentState):
    """Render full conversation history in collapsible section"""
    messages = state.get("messages", [])
    
    if not messages:
        return
    
    with st.expander("💬 Conversation History", expanded=False):
        st.markdown('<div style="max-height: 300px; overflow-y: auto; padding: 0.5rem;">', unsafe_allow_html=True)
        
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if role == "user":
                st.markdown('<div class="user-message-card">', unsafe_allow_html=True)
                st.markdown('<div class="message-header user-message-header">👤 You</div>', unsafe_allow_html=True)
                st.markdown(content)
                st.markdown('</div>', unsafe_allow_html=True)
            elif role == "ai":
                st.markdown('<div class="ai-message-card">', unsafe_allow_html=True)
                st.markdown('<div class="message-header">🤖 AI Assistant</div>', unsafe_allow_html=True)
                st.markdown(content)
                st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.caption(f"Total messages: {len(messages)}")

def render_results_panel(state: AgentState):
    """Render results in the right column with better UX"""
    # Use tabs for better organization
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Risk", "📊 Portfolio", "💼 Investment", "📈 Trading"])
    
    with tab1:
        render_risk_results(state)
    
    with tab2:
        render_portfolio_results(state)
    
    with tab3:
        render_investment_results(state)
    
    with tab4:
        render_trading_results(state)

def render_risk_results(state: AgentState):
    """Render risk assessment results"""
    if not state.get("risk") and not state.get("answers"):
        st.info("Risk assessment not started yet. Complete the risk assessment phase to see results here.")
        return
    
    if state.get("risk"):
        col1, col2 = st.columns(2)
        
        with col1:
            equity = state["risk"].get("equity", 0)
            bond = state["risk"].get("bond", 0)
            
            fig = go.Figure(data=[
                go.Bar(name='', x=['Equity', 'Bonds'], y=[equity, bond], 
                      marker_color=['#667eea', '#764ba2'])
            ])
            fig.update_layout(
                title="Asset Allocation",
                yaxis_title="Percentage",
                height=300,
                showlegend=False,
                template="plotly_white"
            )
            st.plotly_chart(fig, width="stretch", config={'displayModeBar': False})
        
        with col2:
            st.markdown("### Allocation Details")
            st.metric("Equity", f"{equity:.1%}")
            st.metric("Bonds", f"{bond:.1%}")
    
    if state.get("answers"):
        with st.expander("📝 Questionnaire Responses", expanded=False):
            for qid, answer in state["answers"].items():
                if isinstance(answer, dict) and "selected_label" in answer:
                    st.write(f"**{qid.upper()}:** {answer['selected_label']}")

def render_portfolio_results(state: AgentState):
    """Render portfolio allocation results"""
    if not state.get("portfolio"):
        st.info("Portfolio optimization not completed yet. Complete the portfolio phase to see results here.")
        return
    
    portfolio = state["portfolio"]
    weights = {k: v for k, v in portfolio.items() if isinstance(v, (int, float)) and v > 0}
    
    if weights:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = go.Figure(data=[go.Pie(
                labels=list(weights.keys()),
                values=list(weights.values()),
                hole=0.4,
                marker_colors=px.colors.qualitative.Set3
            )])
            fig.update_layout(
                title="Portfolio Allocation",
                height=400,
                template="plotly_white"
            )
            st.plotly_chart(fig, width="stretch", config={'displayModeBar': False})
        
        with col2:
            st.markdown("### Details")
            df = pd.DataFrame(list(weights.items()), columns=['Asset Class', 'Weight'])
            df['Weight'] = df['Weight'].apply(lambda x: f"{x:.1%}")
            st.dataframe(df, width="stretch", hide_index=True)
            
            if "lambda" in portfolio:
                st.metric("Risk Aversion (λ)", f"{portfolio['lambda']:.2f}")
            if "cash_reserve" in portfolio:
                st.metric("Cash Reserve", f"{portfolio['cash_reserve']:.1%}")

def render_investment_results(state: AgentState):
    """Render investment fund selections"""
    investment = state.get("investment")
    if not investment or not isinstance(investment, dict):
        st.info("Fund selection not completed yet. Complete the investment phase to see results here.")
        return
    
    table_data = []
    for asset_class, fund_info in investment.items():
        if isinstance(fund_info, dict) and "ticker" in fund_info:
            table_data.append({
                "Asset Class": asset_class.replace('_', ' ').title(),
                "Fund": fund_info.get('ticker', 'N/A'),
                "Weight": f"{fund_info.get('weight', 0):.1%}",
                "Criteria": fund_info.get('criteria_used', 'N/A')
            })
    
    if table_data:
        df = pd.DataFrame(table_data)
        st.dataframe(df, width="stretch", hide_index=True)
        
        # Detailed view
        st.markdown("### Fund Details")
        for asset_class, fund_info in investment.items():
            if isinstance(fund_info, dict) and "ticker" in fund_info:
                with st.expander(f"{asset_class.replace('_', ' ').title()} - {fund_info.get('ticker', 'N/A')}", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Ticker:** {fund_info.get('ticker', 'N/A')}")
                        st.write(f"**Weight:** {fund_info.get('weight', 0):.1%}")
                    with col2:
                        st.write(f"**Reason:** {fund_info.get('selection_reason', 'N/A')}")
                        if "analysis" in fund_info:
                            st.write(f"**Analysis:** {fund_info['analysis'][:200]}...")

def render_trading_results(state: AgentState):
    """Render trading requests"""
    if not state.get("trading_requests"):
        st.info("Trading requests not generated yet. Complete the trading phase to see results here.")
        return
    
    trading_requests = state["trading_requests"]
    
    if isinstance(trading_requests, dict) and "trading_requests" in trading_requests:
        requests = trading_requests["trading_requests"]
        
        if isinstance(requests, list) and requests:
            df_data = []
            for req in requests:
                if isinstance(req, dict):
                    df_data.append({
                        "Side": req.get("Side", req.get("side", "N/A")),
                        "Ticker": req.get("Ticker", req.get("ticker", "N/A")),
                        "Shares": req.get("Shares", req.get("shares", "N/A")),
                        "Price": f"${req.get('Price', req.get('price', 0)):.2f}",
                        "Proceeds": f"${req.get('Proceeds', req.get('proceeds', 0)):.2f}"
                    })
            
            if df_data:
                df = pd.DataFrame(df_data)
                st.dataframe(df, width="stretch", hide_index=True)
        
        if "summary" in trading_requests:
            summary = trading_requests["summary"]
            if isinstance(summary, dict):
                col1, col2, col3 = st.columns(3)
                with col1:
                    if "total_value" in summary:
                        st.metric("Total Value", f"${summary['total_value']:,.2f}")
                with col2:
                    if "num_trades" in summary:
                        st.metric("Trades", summary["num_trades"])
                with col3:
                    if "estimated_cost" in summary:
                        st.metric("Est. Cost", f"${summary['estimated_cost']:,.2f}")

def main():
    """Main Streamlit application"""
    # Initialize session state
    initialize_session_state()
    
    # Sidebar - compact
    with st.sidebar:
        st.markdown("### 🤖 Robo-Advisor")
        
        # Reset button - compact
        if st.button("🔄 Reset", type="primary", width="stretch"):
            reset_app()
        
        st.markdown("---")
        
        # Status bar
        render_status_bar(st.session_state.state)
        
        # Health check
        if 'llm' in st.session_state:
            render_health_check(st.session_state.llm)
        
        # Monitoring
        render_monitoring()
    
    # Main content
    # Product overview
    render_product_overview()
    
    # Initialize the app if not done
    if not st.session_state.initialized:
        st.session_state.state = st.session_state.graph.invoke(st.session_state.state)
        st.session_state.initialized = True
        st.rerun()
    
    # Two column layout
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.markdown("#### 💬 Chat")
        
        # Chat messages (AI on top)
        render_chat_messages(st.session_state.state)
        
        # User input - compact
        # Use a form to handle input properly
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_input(
                "Your message:",
                placeholder="Type your message...",
                key="user_input",
                label_visibility="collapsed"
            )
            
            submitted = st.form_submit_button("Send", type="primary", width="stretch")
        
        # Display input warning below the form if there is one
        if 'input_warning' in st.session_state:
            st.warning(st.session_state.input_warning)
        
        # Validation and processing (outside the form)
        if submitted and user_input:
            # Initialize guard if not already done
            if 'guard' not in st.session_state:
                st.session_state.guard = get_guard()
            
            # Validate user input for prompt injection attempts
            is_safe, error_msg = st.session_state.guard.validate(user_input)
            if not is_safe:
                # Store warning in session state instead of adding AI message
                st.session_state.input_warning = f"⚠️ {error_msg}\n\nPlease try rephrasing your message."
                st.rerun()
                return
            
            # Clear warning if input is safe
            if 'input_warning' in st.session_state:
                del st.session_state.input_warning
            
            # Add user message to state
            st.session_state.state["messages"].append({"role": "user", "content": user_input})
            
            # Process through the graph
            st.session_state.state = st.session_state.graph.invoke(st.session_state.state)
            
            # Rerun to refresh the UI
            st.rerun()
        
        # Collapsible sections
        render_conversation_history(st.session_state.state)
    
    with col2:
        st.markdown("### 📊 Results & Analysis")
        render_results_panel(st.session_state.state)

if __name__ == "__main__":
    main()
