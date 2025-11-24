"""
Health check framework for system components.
"""

from enum import Enum
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from abc import ABC, abstractmethod


class HealthStatus(Enum):
    """Health status enumeration"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheckResult:
    """Result of a health check"""
    status: HealthStatus
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    response_time_ms: Optional[float] = None


class HealthCheck(ABC):
    """Base class for health checks"""
    
    @abstractmethod
    def check(self) -> HealthCheckResult:
        """Perform health check"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get health check name"""
        pass


class CompositeHealthCheck:
    """Composite health checker that aggregates multiple health checks"""
    
    def __init__(self, checks: list[HealthCheck]):
        """
        Initialize composite health check.
        
        Args:
            checks: List of health check instances
        """
        self.checks = checks
    
    def check_all(self) -> Dict[str, HealthCheckResult]:
        """
        Run all health checks.
        
        Returns:
            Dictionary mapping check names to results
        """
        results = {}
        for check in self.checks:
            try:
                results[check.get_name()] = check.check()
            except Exception as e:
                results[check.get_name()] = HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check failed: {str(e)}",
                    details={"error": str(e)},
                    timestamp=datetime.now()
                )
        return results
    
    def get_overall_status(self) -> HealthStatus:
        """
        Get overall system health status.
        
        Returns:
            Overall health status
        """
        results = self.check_all()
        
        if not results:
            return HealthStatus.UNHEALTHY
        
        statuses = [result.status for result in results.values()]
        
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY

