"""
Performance Monitoring System for Sub-Agents.

Tracks agent utilization, coordination efficiency, and system performance.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import statistics


logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Individual performance metric."""
    metric_name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    agent_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentMetrics:
    """Performance metrics for a single agent."""
    agent_name: str
    invocation_count: int = 0
    total_processing_time: float = 0.0
    total_token_usage: int = 0
    success_count: int = 0
    error_count: int = 0
    average_confidence: float = 0.0
    last_invocation: Optional[datetime] = None
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    token_usage_history: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def record_invocation(
        self,
        processing_time: float,
        token_usage: int,
        success: bool,
        confidence: float = 1.0
    ) -> None:
        """Record a single invocation."""
        self.invocation_count += 1
        self.total_processing_time += processing_time
        self.total_token_usage += token_usage
        self.last_invocation = datetime.now()
        
        self.response_times.append(processing_time)
        self.token_usage_history.append(token_usage)
        
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
        
        # Update average confidence
        self.average_confidence = (
            (self.average_confidence * (self.invocation_count - 1) + confidence) /
            self.invocation_count
        )
    
    def get_average_response_time(self) -> float:
        """Get average response time."""
        if not self.response_times:
            return 0.0
        return statistics.mean(self.response_times)
    
    def get_p95_response_time(self) -> float:
        """Get 95th percentile response time."""
        if not self.response_times:
            return 0.0
        sorted_times = sorted(self.response_times)
        index = int(len(sorted_times) * 0.95)
        return sorted_times[min(index, len(sorted_times) - 1)]
    
    def get_success_rate(self) -> float:
        """Get success rate."""
        total = self.success_count + self.error_count
        if total == 0:
            return 0.0
        return self.success_count / total


@dataclass
class CoordinationMetrics:
    """Metrics for agent coordination."""
    total_coordinations: int = 0
    total_handoff_time: float = 0.0
    parallel_executions: int = 0
    sequential_executions: int = 0
    coordination_patterns: Dict[str, int] = field(default_factory=dict)
    handoff_times: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def record_handoff(self, from_agent: str, to_agent: str, handoff_time: float) -> None:
        """Record a handoff between agents."""
        self.total_coordinations += 1
        self.total_handoff_time += handoff_time
        self.handoff_times.append(handoff_time)
        
        pattern = f"{from_agent}->{to_agent}"
        self.coordination_patterns[pattern] = self.coordination_patterns.get(pattern, 0) + 1
    
    def get_average_handoff_time(self) -> float:
        """Get average handoff time."""
        if not self.handoff_times:
            return 0.0
        return statistics.mean(self.handoff_times)


class MetricsCollector:
    """
    Collects and analyzes performance metrics for the sub-agent system.
    """
    
    def __init__(self):
        """Initialize the metrics collector."""
        self.agent_metrics: Dict[str, AgentMetrics] = {}
        self.coordination_metrics = CoordinationMetrics()
        self.system_metrics: List[PerformanceMetric] = []
        self.metric_history: deque = deque(maxlen=10000)
        self.start_time = datetime.now()
        self._alerts: List[Dict[str, Any]] = []
        
        # Performance thresholds for alerting
        self.thresholds = {
            "response_time_p95": 5.0,  # 5 seconds
            "error_rate": 0.1,          # 10%
            "token_usage_per_request": 5000,
            "handoff_time": 1.0         # 1 second
        }
    
    async def track_metric(self, metric_name: str, value: Any) -> None:
        """
        Track a generic metric.
        
        Args:
            metric_name: Name of the metric
            value: Metric value (can be dict with additional info)
        """
        if isinstance(value, dict):
            metric = PerformanceMetric(
                metric_name=metric_name,
                value=value.get("value", 0),
                agent_name=value.get("agent"),
                metadata=value
            )
        else:
            metric = PerformanceMetric(
                metric_name=metric_name,
                value=float(value)
            )
        
        self.metric_history.append(metric)
        
        # Check for threshold violations
        await self._check_thresholds(metric)
    
    def record_agent_invocation(
        self,
        agent_name: str,
        start_time: float,
        end_time: float,
        token_usage: int,
        success: bool,
        confidence: float = 1.0
    ) -> None:
        """
        Record an agent invocation.
        
        Args:
            agent_name: Name of the agent
            start_time: Start timestamp
            end_time: End timestamp
            token_usage: Tokens used
            success: Whether invocation succeeded
            confidence: Confidence score of result
        """
        if agent_name not in self.agent_metrics:
            self.agent_metrics[agent_name] = AgentMetrics(agent_name)
        
        processing_time = end_time - start_time
        self.agent_metrics[agent_name].record_invocation(
            processing_time,
            token_usage,
            success,
            confidence
        )
        
        logger.debug(
            f"Recorded invocation for {agent_name}: "
            f"{processing_time:.2f}s, {token_usage} tokens"
        )
    
    def record_coordination(
        self,
        from_agent: str,
        to_agent: str,
        handoff_time: float,
        is_parallel: bool = False
    ) -> None:
        """
        Record coordination between agents.
        
        Args:
            from_agent: Source agent
            to_agent: Target agent
            handoff_time: Time for handoff
            is_parallel: Whether execution was parallel
        """
        self.coordination_metrics.record_handoff(from_agent, to_agent, handoff_time)
        
        if is_parallel:
            self.coordination_metrics.parallel_executions += 1
        else:
            self.coordination_metrics.sequential_executions += 1
    
    async def _check_thresholds(self, metric: PerformanceMetric) -> None:
        """Check if metric violates any thresholds."""
        alerts = []
        
        # Check response time
        if metric.metric_name == "response_time" and metric.agent_name:
            agent_metrics = self.agent_metrics.get(metric.agent_name)
            if agent_metrics:
                p95 = agent_metrics.get_p95_response_time()
                if p95 > self.thresholds["response_time_p95"]:
                    alerts.append({
                        "type": "high_response_time",
                        "agent": metric.agent_name,
                        "value": p95,
                        "threshold": self.thresholds["response_time_p95"]
                    })
        
        # Check error rate
        for agent_name, agent_metrics in self.agent_metrics.items():
            error_rate = 1 - agent_metrics.get_success_rate()
            if error_rate > self.thresholds["error_rate"]:
                alerts.append({
                    "type": "high_error_rate",
                    "agent": agent_name,
                    "value": error_rate,
                    "threshold": self.thresholds["error_rate"]
                })
        
        # Store alerts
        for alert in alerts:
            alert["timestamp"] = datetime.now()
            self._alerts.append(alert)
            logger.warning(f"Performance alert: {alert}")
    
    def get_agent_stats(self, agent_name: str) -> Dict[str, Any]:
        """Get statistics for a specific agent."""
        if agent_name not in self.agent_metrics:
            return {}
        
        metrics = self.agent_metrics[agent_name]
        
        return {
            "invocation_count": metrics.invocation_count,
            "total_processing_time": metrics.total_processing_time,
            "average_processing_time": metrics.get_average_response_time(),
            "p95_response_time": metrics.get_p95_response_time(),
            "total_token_usage": metrics.total_token_usage,
            "average_token_usage": (
                metrics.total_token_usage / metrics.invocation_count
                if metrics.invocation_count > 0 else 0
            ),
            "success_rate": metrics.get_success_rate(),
            "average_confidence": metrics.average_confidence,
            "last_invocation": (
                metrics.last_invocation.isoformat()
                if metrics.last_invocation else None
            )
        }
    
    def get_coordination_stats(self) -> Dict[str, Any]:
        """Get coordination statistics."""
        metrics = self.coordination_metrics
        
        return {
            "total_coordinations": metrics.total_coordinations,
            "average_handoff_time": metrics.get_average_handoff_time(),
            "parallel_executions": metrics.parallel_executions,
            "sequential_executions": metrics.sequential_executions,
            "parallel_ratio": (
                metrics.parallel_executions / 
                (metrics.parallel_executions + metrics.sequential_executions)
                if (metrics.parallel_executions + metrics.sequential_executions) > 0
                else 0
            ),
            "top_patterns": sorted(
                metrics.coordination_patterns.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get overall system statistics."""
        total_invocations = sum(m.invocation_count for m in self.agent_metrics.values())
        total_tokens = sum(m.total_token_usage for m in self.agent_metrics.values())
        total_time = sum(m.total_processing_time for m in self.agent_metrics.values())
        
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        return {
            "uptime_seconds": uptime,
            "total_invocations": total_invocations,
            "invocations_per_minute": (total_invocations / uptime) * 60 if uptime > 0 else 0,
            "total_tokens": total_tokens,
            "tokens_per_invocation": total_tokens / total_invocations if total_invocations > 0 else 0,
            "total_processing_time": total_time,
            "agent_count": len(self.agent_metrics),
            "active_agents": len([
                m for m in self.agent_metrics.values()
                if m.last_invocation and 
                (datetime.now() - m.last_invocation) < timedelta(minutes=5)
            ]),
            "recent_alerts": len([
                a for a in self._alerts
                if (datetime.now() - a["timestamp"]) < timedelta(minutes=10)
            ])
        }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        return {
            "system": self.get_system_stats(),
            "agents": {
                name: self.get_agent_stats(name)
                for name in self.agent_metrics.keys()
            },
            "coordination": self.get_coordination_stats(),
            "alerts": self._alerts[-10:] if self._alerts else []
        }
    
    def calculate_efficiency_score(self) -> float:
        """
        Calculate overall system efficiency score (0-100).
        
        Based on:
        - Success rates
        - Response times
        - Token efficiency
        - Parallel execution ratio
        """
        scores = []
        
        # Agent success rates (40% weight)
        if self.agent_metrics:
            success_rates = [m.get_success_rate() for m in self.agent_metrics.values()]
            avg_success = statistics.mean(success_rates) if success_rates else 0
            scores.append(avg_success * 40)
        
        # Response time efficiency (30% weight)
        if self.agent_metrics:
            response_times = [m.get_average_response_time() for m in self.agent_metrics.values()]
            avg_response = statistics.mean(response_times) if response_times else 0
            # Score decreases as response time increases (max 5s for 0 score)
            time_score = max(0, 1 - (avg_response / 5))
            scores.append(time_score * 30)
        
        # Token efficiency (20% weight)
        system_stats = self.get_system_stats()
        tokens_per_inv = system_stats.get("tokens_per_invocation", 0)
        # Score decreases as token usage increases (max 5000 for 0 score)
        token_score = max(0, 1 - (tokens_per_inv / 5000))
        scores.append(token_score * 20)
        
        # Parallel execution (10% weight)
        coord_stats = self.get_coordination_stats()
        parallel_ratio = coord_stats.get("parallel_ratio", 0)
        scores.append(parallel_ratio * 10)
        
        return sum(scores) if scores else 0.0
    
    def reset_metrics(self) -> None:
        """Reset all metrics."""
        self.agent_metrics.clear()
        self.coordination_metrics = CoordinationMetrics()
        self.system_metrics.clear()
        self.metric_history.clear()
        self._alerts.clear()
        self.start_time = datetime.now()
        logger.info("Metrics reset")