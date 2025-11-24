"""
Metrics collection framework for monitoring system performance.
"""

import time
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import threading


@dataclass
class MetricValue:
    """Single metric value with timestamp"""
    value: float
    timestamp: datetime = field(default_factory=datetime.now)


class Counter:
    """Counter metric that can only increase"""
    
    def __init__(self, name: str):
        self.name = name
        self._value = 0
        self._lock = threading.Lock()
    
    def inc(self, value: float = 1.0):
        """Increment counter"""
        with self._lock:
            self._value += value
    
    def get(self) -> float:
        """Get current value"""
        return self._value
    
    def reset(self):
        """Reset counter to zero"""
        with self._lock:
            self._value = 0


class Gauge:
    """Gauge metric that can increase or decrease"""
    
    def __init__(self, name: str):
        self.name = name
        self._value = 0.0
        self._lock = threading.Lock()
    
    def set(self, value: float):
        """Set gauge value"""
        with self._lock:
            self._value = value
    
    def inc(self, value: float = 1.0):
        """Increment gauge"""
        with self._lock:
            self._value += value
    
    def dec(self, value: float = 1.0):
        """Decrement gauge"""
        with self._lock:
            self._value -= value
    
    def get(self) -> float:
        """Get current value"""
        return self._value


class Histogram:
    """Histogram metric for tracking value distributions"""
    
    def __init__(self, name: str):
        self.name = name
        self._values: list[float] = []
        self._lock = threading.Lock()
    
    def observe(self, value: float):
        """Record a value"""
        with self._lock:
            self._values.append(value)
            # Keep only last 1000 values
            if len(self._values) > 1000:
                self._values = self._values[-1000:]
    
    def get_stats(self) -> Dict[str, float]:
        """Get statistics (min, max, mean, count)"""
        with self._lock:
            if not self._values:
                return {"count": 0, "min": 0, "max": 0, "mean": 0}
            
            values = self._values.copy()
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values) if values else 0
        }
    
    def reset(self):
        """Reset histogram"""
        with self._lock:
            self._values.clear()


class Timer:
    """Timer metric for measuring execution time"""
    
    def __init__(self, name: str):
        self.name = name
        self._histogram = Histogram(f"{name}_duration")
        self._counter = Counter(f"{name}_count")
    
    def time(self, func: Callable) -> Callable:
        """Decorator to time a function"""
        import functools
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start
                self._histogram.observe(duration)
                self._counter.inc()
        return wrapper
    
    def record(self, duration: float):
        """Record a duration manually"""
        self._histogram.observe(duration)
        self._counter.inc()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get timer statistics"""
        return {
            "count": self._counter.get(),
            **self._histogram.get_stats()
        }


class MetricsRegistry:
    """Central registry for all metrics"""
    
    def __init__(self):
        self._counters: Dict[str, Counter] = {}
        self._gauges: Dict[str, Gauge] = {}
        self._histograms: Dict[str, Histogram] = {}
        self._timers: Dict[str, Timer] = {}
        self._lock = threading.Lock()
    
    def counter(self, name: str) -> Counter:
        """Get or create a counter"""
        with self._lock:
            if name not in self._counters:
                self._counters[name] = Counter(name)
            return self._counters[name]
    
    def gauge(self, name: str) -> Gauge:
        """Get or create a gauge"""
        with self._lock:
            if name not in self._gauges:
                self._gauges[name] = Gauge(name)
            return self._gauges[name]
    
    def histogram(self, name: str) -> Histogram:
        """Get or create a histogram"""
        with self._lock:
            if name not in self._histograms:
                self._histograms[name] = Histogram(name)
            return self._histograms[name]
    
    def timer(self, name: str) -> Timer:
        """Get or create a timer"""
        with self._lock:
            if name not in self._timers:
                self._timers[name] = Timer(name)
            return self._timers[name]
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics as a dictionary"""
        metrics = {}
        
        for name, counter in self._counters.items():
            metrics[f"counter_{name}"] = counter.get()
        
        for name, gauge in self._gauges.items():
            metrics[f"gauge_{name}"] = gauge.get()
        
        for name, histogram in self._histograms.items():
            metrics[f"histogram_{name}"] = histogram.get_stats()
        
        for name, timer in self._timers.items():
            metrics[f"timer_{name}"] = timer.get_stats()
        
        return metrics
    
    def reset_all(self):
        """Reset all metrics"""
        with self._lock:
            for counter in self._counters.values():
                counter.reset()
            for histogram in self._histograms.values():
                histogram.reset()


# Global metrics registry
_metrics_registry = MetricsRegistry()


def get_metrics_registry() -> MetricsRegistry:
    """Get the global metrics registry"""
    return _metrics_registry

