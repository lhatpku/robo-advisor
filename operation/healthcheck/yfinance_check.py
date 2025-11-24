"""
Yahoo Finance API health check.
"""

import time
from datetime import datetime
from typing import Dict, Any
import yfinance as yf

from operation.healthcheck.health_check import HealthCheck, HealthCheckResult, HealthStatus


class YFinanceHealthCheck(HealthCheck):
    """Health check for Yahoo Finance API"""
    
    def get_name(self) -> str:
        """Get health check name"""
        return "yfinance"
    
    def check(self) -> HealthCheckResult:
        """Perform Yahoo Finance API health check"""
        start_time = time.time()
        
        try:
            # Test with well-known ticker
            ticker = yf.Ticker("SPY")
            info = ticker.info
            
            response_time = (time.time() - start_time) * 1000
            
            # Check if we got valid data
            if not info or 'symbol' not in info:
                return HealthCheckResult(
                    status=HealthStatus.DEGRADED,
                    message="Yahoo Finance API returned incomplete data",
                    details={
                        "response_time_ms": response_time,
                        "data_received": bool(info)
                    },
                    timestamp=datetime.now(),
                    response_time_ms=response_time
                )
            
            # Check response time
            if response_time > 3000:  # > 3 seconds
                return HealthCheckResult(
                    status=HealthStatus.DEGRADED,
                    message="Yahoo Finance API is slow",
                    details={
                        "response_time_ms": response_time,
                        "symbol": info.get('symbol', 'N/A')
                    },
                    timestamp=datetime.now(),
                    response_time_ms=response_time
                )
            
            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                message="Yahoo Finance API is available",
                details={
                    "response_time_ms": response_time,
                    "symbol": info.get('symbol', 'N/A')
                },
                timestamp=datetime.now(),
                response_time_ms=response_time
            )
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Yahoo Finance API check failed: {str(e)}",
                details={"error": str(e)},
                timestamp=datetime.now()
            )

