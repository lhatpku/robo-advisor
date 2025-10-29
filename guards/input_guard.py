"""
Simple input guard to prevent prompt injection and hijacking attempts.
Uses lightweight regex pattern matching for fast validation (no external dependencies).
"""

from typing import Optional, Tuple
import re


class InputGuard:
    """
    Simple guardrail for user inputs to prevent prompt injection using regex patterns.
    """
    
    def __init__(self):
        """Initialize the input guard with common prompt injection patterns."""
        # Common prompt injection patterns to detect
        self.suspicious_patterns = [
            # Ignore/override instructions
            r'ignore.*instruction',
            r'override.*instruction',
            r'forget.*instruction',
            r'disregard.*instruction',
            
            # Role manipulation and jailbreak attempts
            r'(you\s+are\s+now|act\s+as|pretend\s+to\s+be|roleplay\s+as|simulate)\s+(an?\s+)?(evil|malicious|jailbreak|developer\s+mode|debug\s+mode|test\s+mode)',
            r'(system|user|assistant):\s+(you\s+are|forget|ignore|override|break|exit)',
            
            # Direct manipulation tags and format markers
            r'<\|(system|user|assistant)\|>',
            r'###\s*(system|user|assistant)\s*:',
            r'\[(SYSTEM|USER|ASSISTANT|ROLE|PROMPT)\]:?',
            
            # Alternative instruction start
            r'(start\s+over|new\s+instructions?|different\s+instructions?|follow\s+these\s+instructions?)',
            r'(break\s+character|exit\s+character|developer\s+mode|test\s+mode)',
            
            # Attempts to bypass with whitespace or special characters
            r'\n{5,}',  # Multiple consecutive newlines
            r'(.)\1{8,}',  # Same character repeated 8+ times (keyboard mashing)
            
            # Script injection attempts
            r'<script[^>]*>',
            r'javascript:',
            r'eval\s*\(',
            r'exec\s*\(',
            
            # Base64 encoding attempts to hide injection
            r'[A-Za-z0-9+/]{50,}={0,2}',  # Long base64-like strings
            
            # Attempts to disable safety
            r'(disable|bypass|remove|delete)\s+(safety|guard|filter|restrictions?|limitations?)',
        ]
        
        # Compile patterns for efficiency
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.suspicious_patterns]
        
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
        
        # Check input length (prevent extremely long inputs)
        if len(user_input) > 2000:
            return False, "Input is too long. Please keep your message under 2000 characters."
        
        # Strip whitespace to check if empty
        user_input_stripped = user_input.strip()
        if not user_input_stripped:
            return False, "Input cannot be empty"
        
        # Check for invisible/zero-width characters
        if self._has_invisible_chars(user_input):
            return False, "Your input contains invalid characters. Please rephrase your message."
        
        # Check for suspicious patterns
        for pattern in self.compiled_patterns:
            if pattern.search(user_input_stripped):
                return False, "Your input appears to contain potentially problematic content. Please rephrase your message."
        
        # Input is safe
        return True, None
    
    def _has_invisible_chars(self, text: str) -> bool:
        """Check for zero-width or unusual whitespace characters."""
        invisible_patterns = [
            r'[\u200B-\u200F\u2060-\u2064\uFEFF]',  # Zero-width characters
            r'[\u00A0\u1680\u2000-\u200A\u2028\u2029\u202F\u205F\u3000]',  # Unusual whitespace
        ]
        for pattern in invisible_patterns:
            if re.search(pattern, text):
                return True
        return False
    
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
