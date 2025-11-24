"""
Base agent class for all agents in the robo-advisor system.
Provides common functionality for status management, logging, retry, and monitoring.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, TypeVar, Type
from langchain_openai import ChatOpenAI
from state import AgentState
import logging
import uuid

from operation.logging.logging_config import get_logger, set_correlation_id, get_correlation_id
from operation.retry.retry import retry_with_backoff
from operation.retry.retry_config import OPENAI_RETRY_CONFIG
from operation.monitoring.metrics import get_metrics_registry
from operation.monitoring.performance import track_performance, performance_timer

# Type variable for intent models
IntentModel = TypeVar('IntentModel')

# Initialize logging on module import
from operation.logging.logging_config import setup_logging
import os
setup_logging(
    level=os.getenv("LOG_LEVEL", "INFO"),
    log_file=os.getenv("LOG_FILE", None)
)


class BaseAgent(ABC):
    """
    Base class for all agents in the robo-advisor system.
    Provides common functionality for status management, logging, retry, and monitoring.
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
        self._logger = get_logger(f"agents.{agent_name}")
        self._metrics = get_metrics_registry()
        
        # Initialize correlation ID if not set
        if get_correlation_id() is None:
            set_correlation_id()
    
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
    
    def _get_last_ai_message(self, state: AgentState) -> Optional[str]:
        """
        Get the last AI message from state (typically the last question asked).
        
        Args:
            state: Current agent state
            
        Returns:
            Last AI message content or None
        """
        if not state.get("messages"):
            return None
        
        for msg in reversed(state["messages"]):
            if msg.get("role") == "ai":
                return msg.get("content", "")
        return None
    
    def _handle_unknown_intent(self, state: AgentState, fallback_message: Optional[str] = None) -> AgentState:
        """
        Handle unknown intent by repeating the last question with clarification.
        
        Args:
            state: Current agent state
            fallback_message: Optional fallback message if no previous AI message found
            
        Returns:
            Updated agent state with clarification message
        """
        last_ai_message = self._get_last_ai_message(state)
        
        if last_ai_message:
            # Repeat the last question with clarification
            clarification = f"Sorry, I do not understand your intent. {last_ai_message}"
            self._add_message(state, "ai", clarification)
        elif fallback_message:
            # Use provided fallback message
            clarification = f"Sorry, I do not understand your intent. {fallback_message}"
            self._add_message(state, "ai", clarification)
        else:
            # Generic message if no previous message found
            self._add_message(state, "ai", "Sorry, I do not understand your intent. Could you please rephrase your request?")
        
        # Set status to await user input
        self._set_status(state, awaiting_input=True)
        
        return state
    
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
    
    # ==================== Logging ====================
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger instance for this agent"""
        return self._logger
    
    def _get_correlation_id(self, state: AgentState) -> Optional[str]:
        """
        Get correlation ID from state, or generate new one if not present.
        
        Args:
            state: Current agent state
            
        Returns:
            Correlation ID string
        """
        cid = state.get("correlation_id")
        if not cid:
            cid = str(uuid.uuid4())
            state["correlation_id"] = cid
            set_correlation_id(cid)
        else:
            set_correlation_id(cid)
        return cid
    
    # ==================== Retry Mechanism ====================
    
    def _invoke_llm_with_retry(
        self,
        llm_instance,
        prompt: Any,
        operation_name: str = "llm_invoke"
    ) -> Any:
        """
        Invoke LLM with retry logic and exponential backoff.
        
        Args:
            llm_instance: LLM instance to invoke
            prompt: Prompt to send to LLM
            operation_name: Name of operation for logging
            
        Returns:
            LLM response
            
        Raises:
            Exception: If all retry attempts fail
        """
        @retry_with_backoff(
            max_attempts=OPENAI_RETRY_CONFIG.max_attempts,
            initial_delay=OPENAI_RETRY_CONFIG.initial_delay,
            max_delay=OPENAI_RETRY_CONFIG.max_delay,
            multiplier=OPENAI_RETRY_CONFIG.multiplier,
            jitter=OPENAI_RETRY_CONFIG.jitter,
            retryable_exceptions=OPENAI_RETRY_CONFIG.retryable_exceptions,
            strategy=OPENAI_RETRY_CONFIG.strategy
        )
        def _invoke():
            self.logger.debug(f"Invoking LLM: {operation_name}")
            return llm_instance.invoke(prompt)
        
        try:
            result = _invoke()
            self.logger.info(f"LLM operation '{operation_name}' completed successfully")
            self._metrics.counter(f"llm_calls_success_{self.agent_name}").inc()
            return result
        except Exception as e:
            self.logger.error(f"LLM operation '{operation_name}' failed after retries: {e}", exc_info=True)
            self._metrics.counter(f"llm_calls_failed_{self.agent_name}").inc()
            raise
    
    def _classify_intent_with_retry(
        self,
        user_input: str,
        prompt_template: str,
        intent_model: Type[IntentModel],
        structured_llm,
        default_intent: Optional[IntentModel] = None,
        operation_name: str = "classify_intent"
    ) -> IntentModel:
        """
        Classify user intent using LLM with structured output, retry, and logging.
        
        Args:
            user_input: User's input text
            prompt_template: Prompt template string (with {user_input} placeholder)
            intent_model: Pydantic model class for structured output
            structured_llm: Pre-configured structured LLM
            default_intent: Default intent to return on error (optional)
            operation_name: Name of operation for logging
            
        Returns:
            Intent model instance
        """
        prompt = prompt_template.format(user_input=user_input)
        
        self.logger.debug(f"Classifying intent for user input: {user_input[:50]}...")
        
        try:
            intent = self._invoke_llm_with_retry(structured_llm, prompt, operation_name)
            
            # Normalize return
            if isinstance(intent, dict):
                intent = intent_model(**intent)
            elif hasattr(intent, "model_dump"):
                intent = intent_model(**intent.model_dump())
            elif hasattr(intent, "dict"):
                intent = intent_model(**intent.dict())
            
            action = getattr(intent, "action", "unknown")
            self.logger.info(f"Intent classified: {action}")
            
            return intent
            
        except Exception as e:
            self.logger.error(f"Error classifying intent: {e}", exc_info=True)
            
            if default_intent is not None:
                return default_intent
            
            # Try to create a safe default intent
            # First, try with "unknown" action (all models should support it now)
            if hasattr(intent_model, 'action'):
                try:
                    return intent_model(action="unknown")
                except Exception:
                    # "unknown" is not valid (shouldn't happen, but handle gracefully)
                    # Try "proceed" as a safe default
                    try:
                        return intent_model(action="proceed")
                    except Exception:
                        # If "proceed" also fails, try creating with minimal fields
                        try:
                            return intent_model()
                        except Exception:
                            # Last resort: try to extract valid actions from model schema
                            # and use the first one
                            try:
                                schema = intent_model.model_json_schema()
                                if 'properties' in schema and 'action' in schema['properties']:
                                    action_field = schema['properties']['action']
                                    if 'enum' in action_field:
                                        # Use first valid action
                                        first_action = action_field['enum'][0]
                                        return intent_model(action=first_action)
                            except Exception:
                                pass
                            # If all else fails, raise the original error
                            raise RuntimeError(f"Failed to create default intent for {intent_model.__name__}") from e
            
            # No action field, try creating with no args
            try:
                return intent_model()
            except Exception:
                raise RuntimeError(f"Failed to create default intent for {intent_model.__name__}") from e
    
    # ==================== Performance Monitoring ====================
    
    def _track_step_performance(self, func):
        """Decorator to track agent step performance"""
        return track_performance(func)
    
    def _start_step(self, state: AgentState) -> None:
        """
        Initialize step execution: set correlation ID and log step start.
        Call this at the beginning of each agent's step method.
        
        Args:
            state: Current agent state
        """
        # Set correlation ID from state
        cid = self._get_correlation_id(state)
        self.logger.debug(f"Starting {self.agent_name} agent step (correlation_id: {cid})")
        
        # Track step start
        self._metrics.counter(f"agent_step_started_{self.agent_name}").inc()
    
    def _end_step(self, state: AgentState, success: bool = True) -> None:
        """
        Finalize step execution: log step end and track metrics.
        Call this at the end of each agent's step method.
        
        Args:
            state: Current agent state
            success: Whether step completed successfully
        """
        if success:
            self.logger.debug(f"Completed {self.agent_name} agent step")
            self._metrics.counter(f"agent_step_completed_{self.agent_name}").inc()
        else:
            self.logger.warning(f"{self.agent_name} agent step completed with issues")
            self._metrics.counter(f"agent_step_failed_{self.agent_name}").inc()
    
    # ==================== Health Checks ====================
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check for this agent.
        
        Returns:
            Health status dictionary
        """
        health_status = {
            "agent": self.agent_name,
            "status": "healthy",
            "llm_configured": self.llm is not None,
            "logger_configured": self._logger is not None,
            "metrics_enabled": self._metrics is not None
        }
        
        # Test LLM connectivity if possible
        try:
            # Quick test - just check if LLM is configured
            if self.llm:
                health_status["llm_status"] = "configured"
            else:
                health_status["llm_status"] = "not_configured"
                health_status["status"] = "unhealthy"
        except Exception as e:
            health_status["llm_status"] = f"error: {str(e)}"
            health_status["status"] = "unhealthy"
        
        return health_status
    
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

