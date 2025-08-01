# Memory Consolidation System

## Automated Memory Management

The memory consolidation system ensures efficient memory utilization and intelligent pattern promotion across the 5-layer architecture.

```python
class MemoryConsolidationEngine:
    """
    Manages automated memory consolidation, promotion, and cleanup.
    Runs as a background process optimized for user patterns.
    """
    
    def __init__(self, memory_system: FiveLayerMemorySystem):
        self.memory = memory_system
        self.scheduler = AsyncScheduler()
        self.metrics = ConsolidationMetrics()
        
    async def run_consolidation_cycle(self):
        """Execute complete memory consolidation cycle."""
        start_time = time.time()
        
        # 1. Expire old STM entries (> 2 hours)
        expired_stm = await self.memory.expire_memories(
            memory_type='stm',
            older_than=timedelta(hours=2)
        )
        
        # 2. Promote valuable STM → WM
        stm_promotions = await self.memory.promote_memories(
            from_type='stm',
            to_type='wm',
            criteria={
                'effectiveness_score': {'$gte': 5.0},
                'usage_count': {'$gte': 2}
            }
        )
        
        # 3. Promote proven WM → LTM
        wm_promotions = await self.memory.promote_memories(
            from_type='wm',
            to_type='ltm',
            criteria={
                'effectiveness_score': {'$gte': 8.0},
                'usage_count': {'$gte': 5},
                'negative_feedback': 0
            }
        )
        
        # 4. Process SWARM candidates
        swarm_candidates = await self.memory.identify_swarm_candidates(
            from_type='ltm',
            criteria={
                'effectiveness_score': {'$gte': 9.0},
                'general_applicability': True,
                'privacy_validated': True
            }
        )
        
        # 5. Clean up low-value patterns
        cleaned = await self.memory.cleanup_ineffective(
            criteria={
                'effectiveness_score': {'$lt': 3.0},
                'last_accessed': {'$lt': datetime.now() - timedelta(days=30)}
            }
        )
        
        # 6. Optimize vector indices
        await self.memory.optimize_vector_indices()
        
        # Record metrics
        self.metrics.record_cycle({
            'duration': time.time() - start_time,
            'stm_expired': len(expired_stm),
            'stm_to_wm': len(stm_promotions),
            'wm_to_ltm': len(wm_promotions),
            'swarm_candidates': len(swarm_candidates),
            'cleaned': len(cleaned)
        })
        
    def schedule_consolidation(self):
        """Schedule consolidation based on usage patterns."""
        # Run every hour during active use
        self.scheduler.add_job(
            self.run_consolidation_cycle,
            trigger='interval',
            hours=1,
            id='memory_consolidation'
        )
        
        # Deep cleanup daily at 3 AM
        self.scheduler.add_job(
            self.deep_cleanup,
            trigger='cron',
            hour=3,
            id='deep_cleanup'
        )
```

## Memory Health Monitoring

```python
class MemoryHealthMonitor:
    """Monitor memory system health and performance."""
    
    async def get_health_status(self) -> MemoryHealth:
        return {
            "memory_distribution": {
                "stm": await self.count_memories('stm'),
                "wm": await self.count_memories('wm'),
                "ltm": await self.count_memories('ltm'),
                "swarm": await self.count_memories('swarm')
            },
            "promotion_rates": {
                "stm_to_wm": "15% daily average",
                "wm_to_ltm": "5% weekly average",
                "ltm_to_swarm": "1% monthly average"
            },
            "effectiveness_scores": {
                "average": 7.2,
                "trending": "+0.3 over 30 days"
            },
            "storage_usage": {
                "total": "8.5GB",
                "vectors": "6.2GB",
                "metadata": "2.3GB"
            }
        }
```

---
