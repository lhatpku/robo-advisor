# tests/test_entry_learn_more.py
from langchain_groq import ChatGroq
from prompts.entry_prompts import INTENT_CLASSIFICATION_PROMPT
from entry_agent import EntryAgent
from state import AgentState
import os
from dotenv import load_dotenv

load_dotenv()

# Use a mock or lightweight llm if you can't call groq in tests.
# For a quick smoke test, use the same llm configured in your env.
llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=os.getenv("GROQ_API_KEY"), temperature=0.0)

agent = EntryAgent(llm)
state = {
    "messages":[{"role":"ai","content":"welcome"}],
    "summary_shown": {"risk": True, "portfolio": True, "investment": True, "trading": True},
    "answers": {},
    "next_phase": "portfolio",
    "status_tracking": {},
}
# simulate user asking to learn more about portfolio construction
state["messages"].append({"role":"user","content":"what is portfolio management?"})
new_state = agent.step(state)
print("\n--- Full AI Messages ---")
for msg in new_state["messages"]:
    if msg["role"] == "ai":
        print(msg["content"], "\n")



