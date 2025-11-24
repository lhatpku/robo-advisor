"""
File system health check.
"""

import os
from datetime import datetime
from typing import Dict, Any
from pathlib import Path

from operation.healthcheck.health_check import HealthCheck, HealthCheckResult, HealthStatus


class FilesystemHealthCheck(HealthCheck):
    """Health check for file system"""
    
    def __init__(self, required_files: list[str] = None):
        """
        Initialize filesystem health check.
        
        Args:
            required_files: List of required file paths
        """
        self.required_files = required_files or []
    
    def get_name(self) -> str:
        """Get health check name"""
        return "filesystem"
    
    def check(self) -> HealthCheckResult:
        """Perform filesystem health check"""
        missing_files = []
        unreadable_files = []
        
        for file_path in self.required_files:
            path = Path(file_path)
            if not path.exists():
                missing_files.append(str(path))
            elif not os.access(path, os.R_OK):
                unreadable_files.append(str(path))
        
        if missing_files or unreadable_files:
            details = {}
            if missing_files:
                details["missing_files"] = missing_files
            if unreadable_files:
                details["unreadable_files"] = unreadable_files
            
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"File system check failed: {len(missing_files)} missing, {len(unreadable_files)} unreadable",
                details=details,
                timestamp=datetime.now()
            )
        
        return HealthCheckResult(
            status=HealthStatus.HEALTHY,
            message="All required files are accessible",
            details={
                "checked_files": len(self.required_files),
                "all_accessible": True
            },
            timestamp=datetime.now()
        )

