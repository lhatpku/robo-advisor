import os
import inspect
import math
import gradio as gr
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Tuple

from app import build_graph  
from state import AgentState 

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

def _init_state() -> AgentState:
    """Create a fresh AgentState compatible with your app."""
    return {
        "messages": [],
        "q_idx": 0,
        "answers": {},
        "done": False,
        "advice": None,
        "awaiting_input": False,   # used by advice_agent
        "intent_to_advise": False,  # set by entry agent
        "intent_to_investment": False,
        "entry_greeted": False,
        "investment": None             
    }


def _build_graph_instance():
    """Create the LangGraph graph using your build_graph function, handling either signature."""
    try:
        sig = inspect.signature(build_graph)
        if len(sig.parameters) == 0:
            return build_graph()
        # If build_graph expects an LLM, construct a reasonable default
        if ChatOpenAI is None:
            raise RuntimeError("build_graph requires an llm; langchain_openai not available.")
        model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        llm = ChatOpenAI(model=model, temperature=0.1)
        return build_graph(llm)
    except Exception as e:
        raise RuntimeError(f"Failed to build graph: {e}")

def _invoke_graph(graph, state: AgentState) -> AgentState:
    """Single tick invoke of the graph."""
    try:
        new_state = graph.invoke(state)
        return new_state
    except Exception as e:
        # Surface the error in messages (so you see it in the UI)
        state["messages"].append({"role": "ai", "content": f"⚠️ Graph error: {e}"})
        return state


def _last_ai_message(state: AgentState) -> str:
    for m in reversed(state.get("messages", [])):
        if m.get("role") == "ai":
            return str(m.get("content", ""))
    return ""

def _advice_pie(state: AgentState):
    """Return a Matplotlib figure for equity/bond if advice exists; else None."""
    advice = state.get("advice")
    if not advice or "equity" not in advice:
        return None
    eq = float(advice.get("equity", 0.0))
    bd = float(advice.get("bond", max(0.0, 1.0 - eq)))
    if eq <= 0 and bd <= 0:
        return None

    fig, ax = plt.subplots()
    labels = ["Equity", "Bonds"]
    sizes = [max(eq, 0) * 100.0, max(bd, 0) * 100.0]

    # returns (wedges, texts, autotexts) when autopct is used
    res = ax.pie(
        sizes,
        labels=None,              # no on-slice labels
        autopct="%1.1f%%",
        startangle=90
    )
    wedges = res[0]

    ax.axis("equal")
    ax.set_title("Equity / Bond Allocation")

    # Legend to the right to avoid overlap
    legend_labels = [f"{labels[i]} — {sizes[i]:.1f}%" for i in range(len(labels))]
    ax.legend(wedges, legend_labels, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    return fig

def _asset_class_pie(state: AgentState):
    """Return a Matplotlib figure for asset-class portfolio; legend used to avoid label overlap."""
    inv = state.get("investment") or {}
    port: Dict[str, float] = inv.get("portfolio") or {}
    if not port:
        return None

    labels, sizes = [], []
    for k, v in port.items():
        if v and v > 0:
            labels.append(k.replace("_", " ").title())
            sizes.append(float(v) * 100.0)
    if not sizes:
        return None

    fig, ax = plt.subplots()
    # returns (wedges, texts, autotexts) when autopct is used
    res = ax.pie(
        sizes,
        labels=None,            # prevent text on slices (avoids overlap)
        autopct="%1.1f%%",
        startangle=90
    )
    wedges = res[0]

    ax.axis("equal")
    ax.set_title("Asset-Class Portfolio")

    legend_labels = [f"{labels[i]} — {sizes[i]:.1f}%" for i in range(len(labels))]
    # Legend in the lower-right outside the chart area
    ax.legend(
        wedges, legend_labels, loc="lower right",
        bbox_to_anchor=(1.25, -0.02), frameon=False
    )
    fig.tight_layout()
    return fig


def _messages_tail(state: AgentState, n: int = 12) -> str:
    msgs = state.get("messages", [])
    tail = msgs[-n:]
    lines = []
    for m in tail:
        role = m.get("role", "?")
        content = str(m.get("content", "")).strip()
        lines.append(f"{role.upper()}: {content}")
    return "\n\n".join(lines)


# ---------- Gradio Callbacks ----------

def init_session() -> Tuple[List[Tuple[str, str]], AgentState, str, Any, Any, List[List[Any]], str]:
    """
    Initialize the session:
      - Build graph
      - Create a fresh state
      - Invoke once to allow greeting
      - Return (chat_history, state, last_ai, equity/bond fig, asset-class fig, answers table, message tail)
    """
    graph = _build_graph_instance()
    state = _init_state()

    # Initial tick to allow entry agent to greet
    state = _invoke_graph(graph, state)

    chat = []
    last_ai = _last_ai_message(state)
    if last_ai:
        chat.append(("", last_ai))

    eq_fig = _advice_pie(state)
    ac_fig = _asset_class_pie(state)
    tail = _messages_tail(state)

    return chat, state, eq_fig, ac_fig, tail


def on_user_submit(user_text: str, chat_history: List[Tuple[str, str]], state: AgentState):
    """
    Append user text, invoke graph, return updated UI artifacts.
    """
    if not user_text or not isinstance(user_text, str):
        return gr.update(), chat_history, state, gr.update(), gr.update(), gr.update(), gr.update()

    # Grab or rebuild the graph (store it globally on the first call)
    global _GRAPH_SINGLETON
    if "_GRAPH_SINGLETON" not in globals() or _GRAPH_SINGLETON is None:
        _GRAPH_SINGLETON = _build_graph_instance()
    graph = _GRAPH_SINGLETON

    # Append user msg and tick
    state["messages"].append({"role": "user", "content": user_text})
    state = _invoke_graph(graph, state)

    # Find AI response (last)
    ai_text = _last_ai_message(state)
    chat_history = chat_history + [(user_text, ai_text)]

    # Recompute visuals
    eq_fig = _advice_pie(state)
    ac_fig = _asset_class_pie(state)
    tail = _messages_tail(state)

    return "", chat_history, state, eq_fig, ac_fig, tail


def on_reset():
    """Reset conversation entirely."""
    # Clear graph singleton so we rebuild clean
    globals()["_GRAPH_SINGLETON"] = None
    return init_session()

# ---------- Build UI ----------

def launch():
    with gr.Blocks(title="Robo Advisor") as demo:
        gr.Markdown("## Robo Advisor — Chat + Portfolio View")

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=560, type="tuples")
                user_in = gr.Textbox(label="Your message", placeholder="Type here…", autofocus=True)
                with gr.Row():
                    send_btn = gr.Button("Send", variant="primary")
                    reset_btn = gr.Button("Reset")

            with gr.Column(scale=2):
                gr.Markdown("### Allocation")
                eq_plot = gr.Plot(label="Equity/Bond (shows when available)")
                ac_plot = gr.Plot(label="Asset-Class Portfolio (shows when available)")
                gr.Markdown("### State Log")
                state_tail = gr.Textbox(label="Latest Messages (tail)", lines=24, interactive=False)

        # Invisible holders for app state + last AI
        state_store = gr.State({})

        # Wire events
        demo.load(fn=init_session,
                  inputs=[],
                  outputs=[chatbot, state_store, eq_plot, ac_plot, state_tail])

        send_btn.click(fn=on_user_submit,
                       inputs=[user_in, chatbot, state_store],
                       outputs=[user_in, chatbot, state_store, eq_plot, ac_plot, state_tail])

        user_in.submit(fn=on_user_submit,
                       inputs=[user_in, chatbot, state_store],
                       outputs=[user_in, chatbot, state_store, eq_plot, ac_plot, state_tail])

        reset_btn.click(fn=on_reset,
                        inputs=[],
                        outputs=[chatbot, state_store, eq_plot, ac_plot, state_tail])

    demo.queue().launch()


if __name__ == "__main__":
    # Keep a single graph instance per process
    _GRAPH_SINGLETON = None
    launch()
