"""
Result Synthesizer for multi-agent coordination.

Combines and synthesizes results from multiple sub-agents into
coherent, unified outputs.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum


logger = logging.getLogger(__name__)


class SynthesisStrategy(Enum):
    """Strategies for synthesizing multi-agent results."""
    MERGE = "merge"          # Combine all results
    CONSENSUS = "consensus"  # Find agreement between agents
    PRIORITY = "priority"    # Use highest priority result
    WEIGHTED = "weighted"    # Weight by confidence scores
    SEQUENTIAL = "sequential"  # Chain results in sequence


@dataclass
class AgentResult:
    """Result from a single agent."""
    agent_name: str
    agent_nickname: str
    content: Any
    confidence: float = 1.0
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def is_valid(self) -> bool:
        """Check if result is valid."""
        return self.content is not None and self.confidence > 0


@dataclass
class SynthesizedResult:
    """Combined result from multiple agents."""
    strategy: SynthesisStrategy
    combined_content: Any
    contributing_agents: List[str]
    overall_confidence: float
    synthesis_time: float
    individual_results: List[AgentResult]
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ResultSynthesizer:
    """
    Synthesizes results from multiple sub-agents.
    
    Handles different synthesis strategies and conflict resolution.
    """
    
    def __init__(self):
        """Initialize the result synthesizer."""
        self.synthesis_history: List[SynthesizedResult] = []
        
    async def synthesize(
        self,
        results: List[AgentResult],
        strategy: SynthesisStrategy = SynthesisStrategy.MERGE
    ) -> SynthesizedResult:
        """
        Synthesize multiple agent results based on strategy.
        
        Args:
            results: List of results from different agents
            strategy: How to combine the results
            
        Returns:
            Synthesized result combining all inputs
        """
        start_time = datetime.now()
        
        # Filter valid results
        valid_results = [r for r in results if r.is_valid()]
        
        if not valid_results:
            logger.warning("No valid results to synthesize")
            return self._create_empty_result(strategy)
        
        # Apply synthesis strategy
        if strategy == SynthesisStrategy.MERGE:
            synthesized = await self._merge_results(valid_results)
        elif strategy == SynthesisStrategy.CONSENSUS:
            synthesized = await self._find_consensus(valid_results)
        elif strategy == SynthesisStrategy.PRIORITY:
            synthesized = await self._priority_selection(valid_results)
        elif strategy == SynthesisStrategy.WEIGHTED:
            synthesized = await self._weighted_combination(valid_results)
        elif strategy == SynthesisStrategy.SEQUENTIAL:
            synthesized = await self._sequential_chain(valid_results)
        else:
            raise ValueError(f"Unknown synthesis strategy: {strategy}")
        
        # Calculate synthesis time
        synthesis_time = (datetime.now() - start_time).total_seconds()
        
        # Create final result
        result = SynthesizedResult(
            strategy=strategy,
            combined_content=synthesized["content"],
            contributing_agents=[r.agent_name for r in valid_results],
            overall_confidence=synthesized["confidence"],
            synthesis_time=synthesis_time,
            individual_results=valid_results,
            conflicts=synthesized.get("conflicts", []),
            metadata=synthesized.get("metadata", {})
        )
        
        # Store in history
        self.synthesis_history.append(result)
        
        logger.info(
            f"Synthesized {len(valid_results)} results using {strategy.value} "
            f"strategy in {synthesis_time:.2f}s"
        )
        
        return result
    
    async def _merge_results(self, results: List[AgentResult]) -> Dict[str, Any]:
        """
        Merge all results into a combined output.
        
        Combines all agent outputs while preserving individual contributions.
        """
        merged_content = {}
        conflicts = []
        
        for result in results:
            if isinstance(result.content, dict):
                for key, value in result.content.items():
                    if key in merged_content:
                        # Conflict detected
                        if merged_content[key] != value:
                            conflicts.append({
                                "field": key,
                                "values": [merged_content[key], value],
                                "agents": [r.agent_nickname for r in results 
                                         if isinstance(r.content, dict) and 
                                         r.content.get(key) == merged_content[key]] + 
                                         [result.agent_nickname]
                            })
                    else:
                        merged_content[key] = value
            else:
                # Non-dict content
                agent_key = f"{result.agent_nickname}_result"
                merged_content[agent_key] = result.content
        
        # Calculate combined confidence
        avg_confidence = sum(r.confidence for r in results) / len(results)
        
        return {
            "content": merged_content,
            "confidence": avg_confidence,
            "conflicts": conflicts,
            "metadata": {"merge_count": len(results)}
        }
    
    async def _find_consensus(self, results: List[AgentResult]) -> Dict[str, Any]:
        """
        Find consensus among agent results.
        
        Identifies areas of agreement and disagreement.
        """
        consensus_items = {}
        disagreements = []
        
        # Group results by content similarity
        content_groups = {}
        for result in results:
            content_str = str(result.content)
            if content_str not in content_groups:
                content_groups[content_str] = []
            content_groups[content_str].append(result)
        
        # Find majority consensus
        max_agreement = max(len(group) for group in content_groups.values())
        consensus_threshold = len(results) / 2
        
        for content_str, group in content_groups.items():
            if len(group) >= consensus_threshold:
                # Consensus found
                consensus_items["consensus"] = group[0].content
                consensus_items["agreement_level"] = len(group) / len(results)
                consensus_items["agreeing_agents"] = [r.agent_nickname for r in group]
            else:
                # Track disagreement
                disagreements.append({
                    "content": group[0].content,
                    "agents": [r.agent_nickname for r in group],
                    "support": len(group) / len(results)
                })
        
        # Calculate confidence based on agreement level
        confidence = max_agreement / len(results)
        
        return {
            "content": consensus_items,
            "confidence": confidence,
            "conflicts": disagreements,
            "metadata": {
                "total_agents": len(results),
                "consensus_threshold": consensus_threshold
            }
        }
    
    async def _priority_selection(self, results: List[AgentResult]) -> Dict[str, Any]:
        """
        Select result based on priority/confidence.
        
        Uses the highest confidence result.
        """
        # Sort by confidence
        sorted_results = sorted(results, key=lambda r: r.confidence, reverse=True)
        best_result = sorted_results[0]
        
        return {
            "content": best_result.content,
            "confidence": best_result.confidence,
            "metadata": {
                "selected_agent": best_result.agent_nickname,
                "alternatives": len(sorted_results) - 1
            }
        }
    
    async def _weighted_combination(self, results: List[AgentResult]) -> Dict[str, Any]:
        """
        Combine results weighted by confidence scores.
        
        Higher confidence results have more influence.
        """
        total_weight = sum(r.confidence for r in results)
        
        weighted_content = {}
        for result in results:
            weight = result.confidence / total_weight
            
            if isinstance(result.content, dict):
                for key, value in result.content.items():
                    if key not in weighted_content:
                        weighted_content[key] = []
                    weighted_content[key].append({
                        "value": value,
                        "weight": weight,
                        "agent": result.agent_nickname
                    })
            else:
                weighted_content[result.agent_nickname] = {
                    "value": result.content,
                    "weight": weight
                }
        
        # Weighted confidence
        weighted_confidence = sum(r.confidence ** 2 for r in results) / total_weight
        
        return {
            "content": weighted_content,
            "confidence": weighted_confidence,
            "metadata": {"weighting_method": "confidence_based"}
        }
    
    async def _sequential_chain(self, results: List[AgentResult]) -> Dict[str, Any]:
        """
        Chain results sequentially.
        
        Each result builds on the previous one.
        """
        chain = []
        accumulated_content = {}
        
        for i, result in enumerate(results):
            step = {
                "step": i + 1,
                "agent": result.agent_nickname,
                "content": result.content,
                "confidence": result.confidence
            }
            chain.append(step)
            
            # Accumulate content
            if isinstance(result.content, dict):
                accumulated_content.update(result.content)
            else:
                accumulated_content[f"step_{i+1}"] = result.content
        
        # Final confidence is product of all confidences
        final_confidence = 1.0
        for result in results:
            final_confidence *= result.confidence
        
        return {
            "content": {
                "chain": chain,
                "final_output": accumulated_content
            },
            "confidence": final_confidence,
            "metadata": {"chain_length": len(chain)}
        }
    
    def _create_empty_result(self, strategy: SynthesisStrategy) -> SynthesizedResult:
        """Create an empty result when no valid inputs."""
        return SynthesizedResult(
            strategy=strategy,
            combined_content=None,
            contributing_agents=[],
            overall_confidence=0.0,
            synthesis_time=0.0,
            individual_results=[]
        )
    
    async def resolve_conflicts(
        self,
        result: SynthesizedResult,
        resolution_strategy: str = "majority"
    ) -> SynthesizedResult:
        """
        Resolve conflicts in synthesized results.
        
        Args:
            result: Result with conflicts
            resolution_strategy: How to resolve conflicts
            
        Returns:
            Result with conflicts resolved
        """
        if not result.conflicts:
            return result
        
        resolved_content = dict(result.combined_content) if isinstance(
            result.combined_content, dict
        ) else result.combined_content
        
        for conflict in result.conflicts:
            if resolution_strategy == "majority":
                # Choose most common value
                values = conflict.get("values", [])
                if values:
                    resolved_content[conflict["field"]] = values[0]
            elif resolution_strategy == "highest_confidence":
                # Choose from highest confidence agent
                # Would need agent confidence mapping
                pass
        
        result.combined_content = resolved_content
        result.metadata["conflicts_resolved"] = len(result.conflicts)
        result.conflicts = []
        
        return result
    
    def get_synthesis_stats(self) -> Dict[str, Any]:
        """Get statistics about synthesis operations."""
        if not self.synthesis_history:
            return {}
        
        total = len(self.synthesis_history)
        avg_confidence = sum(r.overall_confidence for r in self.synthesis_history) / total
        avg_time = sum(r.synthesis_time for r in self.synthesis_history) / total
        strategy_counts = {}
        
        for result in self.synthesis_history:
            strategy = result.strategy.value
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        return {
            "total_syntheses": total,
            "average_confidence": avg_confidence,
            "average_time": avg_time,
            "strategy_usage": strategy_counts,
            "total_conflicts": sum(len(r.conflicts) for r in self.synthesis_history)
        }