"""
Base agent class for all agents in the robo-advisor system.
Provides common functionality for status management and helper methods.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from state import AgentState


class BaseAgent(ABC):
    """
    Base class for all agents in the robo-advisor system.
    Provides common functionality for status management and helper methods.
    """
    
    def __init__(self, llm: ChatOpenAI, agent_name: str):
        """
        Initialize the base agent.
        
        Args:
            llm: ChatOpenAI instance for LLM calls
            agent_name: Name of the agent (e.g., "risk", "portfolio") for status tracking
        """
        self.llm = llm
        self.agent_name = agent_name
    
    # ==================== Status Management ====================
    
    def _get_status(self, state: AgentState, agent: str = None) -> Dict[str, bool]:
        """
        Get status tracking for a specific agent.
        
        Args:
            state: Current agent state
            agent: Agent name (defaults to self.agent_name)
            
        Returns:
            Status dictionary with 'done' and 'awaiting_input' keys
        """
        agent = agent or self.agent_name
        return state.get("status_tracking", {}).get(agent, {"done": False, "awaiting_input": False})
    
    def _set_status(
        self, 
        state: AgentState, 
        agent: str = None,
        done: bool = None, 
        awaiting_input: bool = None
    ) -> None:
        """
        Set status tracking for a specific agent.
        
        Args:
            state: Current agent state
            agent: Agent name (defaults to self.agent_name)
            done: Whether the agent has completed its task
            awaiting_input: Whether the agent is waiting for user input
        """
        agent = agent or self.agent_name
        
        if "status_tracking" not in state:
            state["status_tracking"] = {}
        if agent not in state["status_tracking"]:
            state["status_tracking"][agent] = {"done": False, "awaiting_input": False}
        
        if done is not None:
            state["status_tracking"][agent]["done"] = done
        if awaiting_input is not None:
            state["status_tracking"][agent]["awaiting_input"] = awaiting_input
    
    # ==================== Message Helpers ====================
    
    def _add_message(self, state: AgentState, role: str, content: str) -> None:
        """
        Add a message to the state.
        
        Args:
            state: Current agent state
            role: Message role ("user" or "ai")
            content: Message content
        """
        if "messages" not in state:
            state["messages"] = []
        state["messages"].append({"role": role, "content": content})
    
    def _get_last_user_message(self, state: AgentState) -> Optional[str]:
        """
        Get the last user message from state.
        
        Args:
            state: Current agent state
            
        Returns:
            Last user message content or None
        """
        if not state.get("messages"):
            return None
        
        for msg in reversed(state["messages"]):
            if msg.get("role") == "user":
                return msg.get("content", "")
        return None
    
    def _is_user_turn(self, state: AgentState) -> bool:
        """
        Check if the last message is from the user.
        
        Args:
            state: Current agent state
            
        Returns:
            True if last message is from user, False otherwise
        """
        if not state.get("messages"):
            return False
        return state["messages"][-1].get("role") == "user"
    
    # ==================== Abstract Methods ====================
    
    @abstractmethod
    def step(self, state: AgentState) -> AgentState:
        """
        Main step function for the agent.
        Must be implemented by each agent.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated agent state
        """
        pass
    
    @abstractmethod
    def router(self, state: AgentState) -> str:
        """
        Router function that determines the next step.
        Must be implemented by each agent.
        
        Args:
            state: Current agent state
            
        Returns:
            Next node name or "__end__"
        """
        pass

