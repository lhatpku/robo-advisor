"""
OpenAI API health check.
"""

import time
from datetime import datetime
from typing import Dict, Any
from langchain_openai import ChatOpenAI

from operation.healthcheck.health_check import HealthCheck, HealthCheckResult, HealthStatus


class OpenAIHealthCheck(HealthCheck):
    """Health check for OpenAI API"""
    
    def __init__(self, llm: ChatOpenAI):
        """
        Initialize OpenAI health check.
        
        Args:
            llm: ChatOpenAI instance to test
        """
        self.llm = llm
    
    def get_name(self) -> str:
        """Get health check name"""
        return "openai"
    
    def check(self) -> HealthCheckResult:
        """Perform OpenAI API health check"""
        start_time = time.time()
        
        try:
            # Minimal API call to test connectivity
            response = self.llm.invoke("test")
            response_time = (time.time() - start_time) * 1000
            
            # Check response time
            if response_time > 5000:  # > 5 seconds
                return HealthCheckResult(
                    status=HealthStatus.DEGRADED,
                    message="OpenAI API is slow",
                    details={
                        "response_time_ms": response_time,
                        "response_received": True
                    },
                    timestamp=datetime.now(),
                    response_time_ms=response_time
                )
            
            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                message="OpenAI API is available",
                details={
                    "response_time_ms": response_time,
                    "response_received": True
                },
                timestamp=datetime.now(),
                response_time_ms=response_time
            )
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"OpenAI API check failed: {str(e)}",
                details={"error": str(e)},
                timestamp=datetime.now()
            )

