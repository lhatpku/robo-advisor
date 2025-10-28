"""
Simple input guard to prevent prompt injection and hijacking attempts.
"""

from typing import Optional, Dict, Any, Tuple
from guardrails import Guard
from guardrails.hub import UnusualPrompt


class InputGuard:
    """
    Simple guardrail for user inputs to prevent prompt injection.
    """
    
    def __init__(self, threshold: float = 0.7):
        """
        Initialize the input guard.
        
        Args:
            threshold: Detection threshold for unusual prompts (0-1, higher = more sensitive)
        """
        # Use UnusualPrompt validator to detect prompt injection attempts
        self.guard = Guard().use(
            UnusualPrompt(threshold=threshold, on_fail="exception")
        )
        
    def validate(self, user_input: str) -> Tuple[bool, Optional[str]]:
        """
        Validate user input for prompt injection attempts.
        
        Args:
            user_input: The user's input string to validate
            
        Returns:
            Tuple of (is_valid, error_message)
            - is_valid: True if input is safe, False if suspicious
            - error_message: Error message if validation failed, None otherwise
        """
        if not user_input or not isinstance(user_input, str):
            return False, "Invalid input format"
        
        # Strip whitespace to check if empty
        user_input = user_input.strip()
        if not user_input:
            return False, "Input cannot be empty"
        
        try:
            # Validate using Guardrails
            self.guard.validate(user_input)
            return True, None
            
        except Exception as e:
            # If validation fails, return error
            return False, "Your input appears to contain potentially problematic content. Please rephrase your message."
    
    def is_safe(self, user_input: str) -> bool:
        """
        Simple check if input is safe (returns boolean only).
        
        Args:
            user_input: The user's input string to validate
            
        Returns:
            True if input is safe, False otherwise
        """
        is_valid, _ = self.validate(user_input)
        return is_valid


# Global guard instance
_guard_instance = None

def get_guard() -> InputGuard:
    """
    Get the global guard instance (singleton pattern).
    
    Returns:
        The global InputGuard instance
    """
    global _guard_instance
    if _guard_instance is None:
        _guard_instance = InputGuard()
    return _guard_instance

