"""
Setup script to install Guardrails validators.
Run this automatically on import or manually during deployment.
"""

import subprocess
import sys
import os

def install_validators():
    """Install required Guardrails validators."""
    try:
        # Install UnusualPrompt validator
        subprocess.check_call([
            sys.executable, "-m", "guardrails", "hub", "install", 
            "hub://guardrails/unusual_prompt"
        ])
        print("✓ UnusualPrompt validator installed successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to install validators: {e}")
        return False

if __name__ == "__main__":
    install_validators()

