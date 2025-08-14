"""
Performance benchmarking suite for sub-agent architecture.

Tests performance metrics including:
- Agent activation latency
- Context isolation overhead
- Parallel execution efficiency
- Memory persistence performance
- Delegation routing speed
"""

import asyncio
import time
import statistics
from pathlib import Path
from typing import List, Dict, Any
import pytest

from src.agents.manager import SubAgentManager
from src.delegation.engine import HybridDelegationEngine


class TestAgentPerformance:
    """Performance benchmarks for sub-agent system."""
    
    @pytest.fixture
    async def manager(self):
        """Create SubAgentManager instance."""
        manager = SubAgentManager()
        await manager.initialize()
        return manager
    
    @pytest.fixture
    async def delegation_engine(self):
        """Create HybridDelegationEngine instance."""
        engine = HybridDelegationEngine()
        await engine.initialize()
        return engine
    
    @pytest.mark.asyncio
    async def test_agent_activation_latency(self, manager):
        """Test agent activation performance."""
        agents = ["prompt-enhancer", "researcher", "reasoner"]
        latencies = []
        
        for agent_name in agents:
            start_time = time.perf_counter()
            await manager.activate_agent(agent_name)
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)  # Convert to ms
        
        avg_latency = statistics.mean(latencies)
        max_latency = max(latencies)
        
        # Performance targets
        assert avg_latency < 50, f"Average activation latency {avg_latency:.2f}ms exceeds 50ms target"
        assert max_latency < 100, f"Max activation latency {max_latency:.2f}ms exceeds 100ms target"
        
        return {
            "avg_latency_ms": avg_latency,
            "max_latency_ms": max_latency,
            "min_latency_ms": min(latencies)
        }
    
    @pytest.mark.asyncio
    async def test_parallel_agent_execution(self, manager):
        """Test parallel execution efficiency."""
        agents = ["prompt-enhancer", "researcher", "reasoner", "evaluator"]
        
        # Sequential execution
        sequential_start = time.perf_counter()
        for agent in agents:
            await manager.activate_agent(agent)
            await manager.execute_agent(agent, {"prompt": "Test prompt"})
        sequential_time = time.perf_counter() - sequential_start
        
        # Reset contexts
        manager.contexts.clear()
        manager.active_agents.clear()
        
        # Parallel execution
        parallel_start = time.perf_counter()
        tasks = [
            manager.execute_agent(agent, {"prompt": "Test prompt"})
            for agent in agents
        ]
        await asyncio.gather(*tasks)
        parallel_time = time.perf_counter() - parallel_start
        
        # Calculate efficiency
        efficiency = (sequential_time / parallel_time) / len(agents) * 100
        
        # Should achieve at least 60% parallel efficiency
        assert efficiency > 60, f"Parallel efficiency {efficiency:.1f}% below 60% target"
        
        return {
            "sequential_time_s": sequential_time,
            "parallel_time_s": parallel_time,
            "efficiency_percent": efficiency,
            "speedup": sequential_time / parallel_time
        }
    
    @pytest.mark.asyncio
    async def test_context_isolation_overhead(self, manager):
        """Test overhead of context isolation."""
        iterations = 100
        
        # Measure context creation overhead
        creation_times = []
        for i in range(iterations):
            start = time.perf_counter()
            await manager.create_context(f"test-agent-{i % 5}")
            end = time.perf_counter()
            creation_times.append((end - start) * 1000)
        
        avg_creation = statistics.mean(creation_times)
        p95_creation = statistics.quantiles(creation_times, n=20)[18]  # 95th percentile
        
        # Context creation should be fast
        assert avg_creation < 10, f"Average context creation {avg_creation:.2f}ms exceeds 10ms"
        assert p95_creation < 20, f"P95 context creation {p95_creation:.2f}ms exceeds 20ms"
        
        return {
            "avg_creation_ms": avg_creation,
            "p95_creation_ms": p95_creation,
            "max_creation_ms": max(creation_times)
        }
    
    @pytest.mark.asyncio
    async def test_memory_persistence_performance(self, manager):
        """Test memory system persistence performance."""
        if not manager.memory_system:
            pytest.skip("Memory system not available")
        
        agents = ["prompt-enhancer", "researcher", "reasoner"]
        persist_times = []
        retrieve_times = []
        
        # Test persistence
        for agent in agents:
            await manager.activate_agent(agent)
            
            start = time.perf_counter()
            await manager.persist_context(agent)
            end = time.perf_counter()
            persist_times.append((end - start) * 1000)
        
        # Test retrieval
        for agent in agents:
            start = time.perf_counter()
            await manager.retrieve_context(agent)
            end = time.perf_counter()
            retrieve_times.append((end - start) * 1000)
        
        avg_persist = statistics.mean(persist_times)
        avg_retrieve = statistics.mean(retrieve_times)
        
        # Memory operations should be reasonably fast
        assert avg_persist < 50, f"Average persist time {avg_persist:.2f}ms exceeds 50ms"
        assert avg_retrieve < 30, f"Average retrieve time {avg_retrieve:.2f}ms exceeds 30ms"
        
        return {
            "avg_persist_ms": avg_persist,
            "avg_retrieve_ms": avg_retrieve,
            "total_memory_ops": len(persist_times) + len(retrieve_times)
        }
    
    @pytest.mark.asyncio
    async def test_delegation_routing_speed(self, delegation_engine):
        """Test delegation engine routing performance."""
        test_prompts = [
            "Write a Python function to sort a list",
            "Research the latest AI trends",
            "Analyze the logical structure of this argument",
            "Evaluate the quality of this code",
            "Create a test plan for this feature"
        ]
        
        routing_times = []
        stage_distributions = {"keyword": 0, "semantic": 0, "fallback": 0}
        
        for prompt in test_prompts * 10:  # Run 50 tests
            start = time.perf_counter()
            result = await delegation_engine.delegate(prompt)
            end = time.perf_counter()
            
            routing_times.append((end - start) * 1000)
            stage_distributions[result.delegation_method] += 1
        
        avg_routing = statistics.mean(routing_times)
        p95_routing = statistics.quantiles(routing_times, n=20)[18]
        
        # Routing should be fast
        assert avg_routing < 100, f"Average routing time {avg_routing:.2f}ms exceeds 100ms"
        assert p95_routing < 200, f"P95 routing time {p95_routing:.2f}ms exceeds 200ms"
        
        # Most should be handled by fast keyword matching
        keyword_percent = stage_distributions["keyword"] / len(routing_times) * 100
        assert keyword_percent > 40, f"Keyword matching {keyword_percent:.1f}% below 40% target"
        
        return {
            "avg_routing_ms": avg_routing,
            "p95_routing_ms": p95_routing,
            "stage_distribution": stage_distributions,
            "keyword_percent": keyword_percent
        }
    
    @pytest.mark.asyncio
    async def test_scalability(self, manager):
        """Test system scalability with many agents."""
        num_agents = 50
        
        # Create many agent contexts
        start = time.perf_counter()
        tasks = [
            manager.create_context(f"agent-{i % 7}")  # Cycle through 7 agent types
            for i in range(num_agents)
        ]
        await asyncio.gather(*tasks)
        total_time = time.perf_counter() - start
        
        # Should handle many agents efficiently
        avg_time_per_agent = (total_time / num_agents) * 1000
        assert avg_time_per_agent < 20, f"Average {avg_time_per_agent:.2f}ms per agent exceeds 20ms"
        
        # Test cleanup
        cleanup_start = time.perf_counter()
        await manager.cleanup_inactive_contexts(max_age_hours=0)
        cleanup_time = time.perf_counter() - cleanup_start
        
        assert cleanup_time < 1.0, f"Cleanup took {cleanup_time:.2f}s, exceeds 1s"
        
        return {
            "num_agents": num_agents,
            "total_creation_time_s": total_time,
            "avg_per_agent_ms": avg_time_per_agent,
            "cleanup_time_s": cleanup_time
        }


class BenchmarkReport:
    """Generate performance benchmark report."""
    
    @staticmethod
    def generate_report(results: Dict[str, Any]) -> str:
        """Generate formatted benchmark report."""
        report = """
========================================
SUB-AGENT PERFORMANCE BENCHMARK REPORT
========================================

1. AGENT ACTIVATION LATENCY
   Average: {activation[avg_latency_ms]:.2f}ms
   Maximum: {activation[max_latency_ms]:.2f}ms
   Target: <50ms avg, <100ms max
   Status: {activation_status}

2. PARALLEL EXECUTION EFFICIENCY
   Sequential Time: {parallel[sequential_time_s]:.2f}s
   Parallel Time: {parallel[parallel_time_s]:.2f}s
   Efficiency: {parallel[efficiency_percent]:.1f}%
   Speedup: {parallel[speedup]:.2f}x
   Target: >60% efficiency
   Status: {parallel_status}

3. CONTEXT ISOLATION OVERHEAD
   Average Creation: {context[avg_creation_ms]:.2f}ms
   P95 Creation: {context[p95_creation_ms]:.2f}ms
   Target: <10ms avg, <20ms P95
   Status: {context_status}

4. MEMORY PERSISTENCE
   Average Persist: {memory[avg_persist_ms]:.2f}ms
   Average Retrieve: {memory[avg_retrieve_ms]:.2f}ms
   Target: <50ms persist, <30ms retrieve
   Status: {memory_status}

5. DELEGATION ROUTING
   Average Routing: {routing[avg_routing_ms]:.2f}ms
   P95 Routing: {routing[p95_routing_ms]:.2f}ms
   Keyword Match: {routing[keyword_percent]:.1f}%
   Target: <100ms avg, >40% keyword
   Status: {routing_status}

6. SCALABILITY
   Agents Tested: {scale[num_agents]}
   Avg per Agent: {scale[avg_per_agent_ms]:.2f}ms
   Cleanup Time: {scale[cleanup_time_s]:.2f}s
   Target: <20ms per agent, <1s cleanup
   Status: {scale_status}

========================================
OVERALL PERFORMANCE: {overall_status}
========================================
"""
        return report


if __name__ == "__main__":
    # Run benchmarks with detailed output
    pytest.main([__file__, "-v", "--tb=short"])