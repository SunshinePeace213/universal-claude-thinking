"""
Pattern Registry for Runtime Pattern Updates
Manages dynamic pattern registration and effectiveness tracking
Part of Universal Claude Thinking v2 - Dynamic Pattern Learning Enhancement
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..storage.db import DatabaseConnection

logger = logging.getLogger(__name__)


@dataclass
class RegisteredPattern:
    """A pattern registered in the system"""
    id: str
    pattern: str
    regex: re.Pattern
    agent: str
    confidence: float
    source: str  # 'original', 'learned', 'manual'
    effectiveness: float = 0.0
    hit_count: int = 0
    miss_count: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_used: datetime | None = None
    enabled: bool = True
    
    def update_effectiveness(self, hit: bool) -> None:
        """Update pattern effectiveness based on hit/miss"""
        if hit:
            self.hit_count += 1
        else:
            self.miss_count += 1
        
        total = self.hit_count + self.miss_count
        if total > 0:
            self.effectiveness = self.hit_count / total
        
        self.last_used = datetime.now(UTC)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'id': self.id,
            'pattern': self.pattern,
            'regex_pattern': self.regex.pattern,
            'agent': self.agent,
            'confidence': self.confidence,
            'source': self.source,
            'effectiveness': self.effectiveness,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'created_at': self.created_at.isoformat(),
            'last_used': self.last_used.isoformat() if self.last_used else None,
            'enabled': self.enabled
        }


class PatternRegistry:
    """
    Central registry for all patterns (original and learned).
    Supports runtime updates and effectiveness tracking.
    """
    
    def __init__(
        self, 
        db_connection: DatabaseConnection | None = None,
        config_path: Path | None = None
    ):
        """
        Initialize pattern registry.
        
        Args:
            db_connection: Database for persistence
            config_path: Path to pattern config file
        """
        self.db = db_connection
        self.config_path = config_path or Path("config/patterns.json")
        
        # Pattern storage
        self.patterns: Dict[str, List[RegisteredPattern]] = {
            'PE': [],
            'R1': [],
            'A1': [],
            'E1': [],
            'T1': [],
            'W1': [],
            'I1': []
        }
        
        # Pattern index for fast lookup
        self.pattern_index: Dict[str, RegisteredPattern] = {}
        
        # Load existing patterns
        asyncio.create_task(self._initialize_patterns())
    
    async def _initialize_patterns(self) -> None:
        """Initialize patterns from database and config"""
        # Load original patterns from config
        await self._load_config_patterns()
        
        # Load learned patterns from database
        if self.db:
            await self._load_learned_patterns()
        
        logger.info(f"Initialized {len(self.pattern_index)} patterns")
    
    async def _load_config_patterns(self) -> None:
        """Load original patterns from configuration file"""
        if not self.config_path.exists():
            # Create default config with original patterns
            await self._create_default_config()
        
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            for agent, patterns in config.get('patterns', {}).items():
                for pattern_data in patterns:
                    pattern = RegisteredPattern(
                        id=pattern_data['id'],
                        pattern=pattern_data['pattern'],
                        regex=re.compile(pattern_data['regex'], re.I),
                        agent=agent,
                        confidence=pattern_data.get('confidence', 0.9),
                        source='original',
                        effectiveness=pattern_data.get('effectiveness', 0.0),
                        hit_count=pattern_data.get('hit_count', 0),
                        miss_count=pattern_data.get('miss_count', 0),
                        enabled=pattern_data.get('enabled', True)
                    )
                    
                    if agent in self.patterns:
                        self.patterns[agent].append(pattern)
                        self.pattern_index[pattern.id] = pattern
                    
        except Exception as e:
            logger.error(f"Error loading config patterns: {e}")
    
    async def _create_default_config(self) -> None:
        """Create default configuration with original patterns"""
        default_config = {
            'version': '1.0',
            'patterns': {
                'PE': [
                    {
                        'id': 'pe_001',
                        'pattern': 'enhance prompt',
                        'regex': r'\b(enhance|improve|clarify|refine) (?:my |the )?(?:prompt|request|query)\b',
                        'confidence': 0.95
                    },
                    {
                        'id': 'pe_002',
                        'pattern': 'make clearer',
                        'regex': r'\b(make .+ clearer|help me ask|rephrase|reword)\b',
                        'confidence': 0.90
                    }
                ],
                'R1': [
                    {
                        'id': 'r1_001',
                        'pattern': 'research',
                        'regex': r'\b(research|search|find|lookup|investigate)\b',
                        'confidence': 0.95
                    },
                    {
                        'id': 'r1_002',
                        'pattern': 'gather information',
                        'regex': r'\b(gather|collect|compile) (?:information|data|sources)\b',
                        'confidence': 0.90
                    }
                ],
                'A1': [
                    {
                        'id': 'a1_001',
                        'pattern': 'analyze solve',
                        'regex': r'\b(reason|analyze|solve|think through|figure out)\b',
                        'confidence': 0.90
                    },
                    {
                        'id': 'a1_002',
                        'pattern': 'debug error',
                        'regex': r'\b(debug|troubleshoot|diagnose|investigate).*(error|issue|problem|bug)\b',
                        'confidence': 0.95
                    }
                ]
            }
        }
        
        # Save default config
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
    
    async def _load_learned_patterns(self) -> None:
        """Load learned patterns from database"""
        try:
            async with self.db.connect() as conn:
                cursor = await conn.execute(
                    """
                    SELECT * FROM learned_patterns
                    WHERE enabled = 1
                    ORDER BY effectiveness DESC
                    """
                )
                rows = await cursor.fetchall()
                
                for row in rows:
                    pattern = RegisteredPattern(
                        id=row['id'],
                        pattern=row['pattern'],
                        regex=re.compile(row['regex_pattern'], re.I),
                        agent=row['agent'],
                        confidence=row['confidence'],
                        source='learned',
                        effectiveness=row['effectiveness'],
                        hit_count=row['hit_count'],
                        miss_count=row['miss_count'],
                        created_at=datetime.fromisoformat(row['created_at']),
                        last_used=datetime.fromisoformat(row['last_used']) if row['last_used'] else None,
                        enabled=row['enabled']
                    )
                    
                    if pattern.agent in self.patterns:
                        self.patterns[pattern.agent].append(pattern)
                        self.pattern_index[pattern.id] = pattern
                        
        except Exception as e:
            # Table might not exist yet
            logger.debug(f"Could not load learned patterns: {e}")
    
    async def register_pattern(
        self,
        pattern: str,
        regex_pattern: str,
        agent: str,
        confidence: float = 0.8,
        source: str = 'learned'
    ) -> RegisteredPattern:
        """
        Register a new pattern at runtime.
        
        Args:
            pattern: Human-readable pattern
            regex_pattern: Regular expression pattern
            agent: Target agent
            confidence: Pattern confidence
            source: Pattern source
            
        Returns:
            Registered pattern object
        """
        # Generate ID
        pattern_id = f"{agent.lower()}_{source}_{len(self.pattern_index):04d}"
        
        # Create pattern object
        registered = RegisteredPattern(
            id=pattern_id,
            pattern=pattern,
            regex=re.compile(regex_pattern, re.I),
            agent=agent,
            confidence=confidence,
            source=source
        )
        
        # Add to registry
        if agent in self.patterns:
            self.patterns[agent].append(registered)
            self.pattern_index[pattern_id] = registered
            
            # Persist if database available
            if self.db:
                await self._persist_pattern(registered)
            
            logger.info(f"Registered new pattern: {pattern_id} for agent {agent}")
        else:
            logger.warning(f"Unknown agent {agent} for pattern registration")
        
        return registered
    
    async def _persist_pattern(self, pattern: RegisteredPattern) -> None:
        """Persist pattern to database"""
        try:
            async with self.db.connect() as conn:
                # Create table if not exists
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS learned_patterns (
                        id TEXT PRIMARY KEY,
                        pattern TEXT NOT NULL,
                        regex_pattern TEXT NOT NULL,
                        agent TEXT NOT NULL,
                        confidence REAL,
                        source TEXT,
                        effectiveness REAL DEFAULT 0,
                        hit_count INTEGER DEFAULT 0,
                        miss_count INTEGER DEFAULT 0,
                        created_at TIMESTAMP,
                        last_used TIMESTAMP,
                        enabled BOOLEAN DEFAULT 1
                    )
                """)
                
                # Insert pattern
                await conn.execute(
                    """
                    INSERT OR REPLACE INTO learned_patterns
                    (id, pattern, regex_pattern, agent, confidence, source,
                     effectiveness, hit_count, miss_count, created_at, enabled)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        pattern.id,
                        pattern.pattern,
                        pattern.regex.pattern,
                        pattern.agent,
                        pattern.confidence,
                        pattern.source,
                        pattern.effectiveness,
                        pattern.hit_count,
                        pattern.miss_count,
                        pattern.created_at.isoformat(),
                        pattern.enabled
                    )
                )
                await conn.commit()
                
        except Exception as e:
            logger.error(f"Error persisting pattern: {e}")
    
    def match_patterns(
        self,
        text: str,
        agent: str | None = None
    ) -> List[tuple[RegisteredPattern, List[str]]]:
        """
        Match text against registered patterns.
        
        Args:
            text: Text to match
            agent: Specific agent patterns to check (None for all)
            
        Returns:
            List of (pattern, matches) tuples
        """
        results = []
        
        # Determine which agents to check
        agents_to_check = [agent] if agent else list(self.patterns.keys())
        
        for agent_key in agents_to_check:
            for pattern in self.patterns.get(agent_key, []):
                if not pattern.enabled:
                    continue
                
                matches = pattern.regex.findall(text)
                if matches:
                    results.append((pattern, matches))
                    # Update pattern effectiveness
                    pattern.update_effectiveness(hit=True)
        
        # Sort by confidence
        results.sort(key=lambda x: x[0].confidence, reverse=True)
        
        return results
    
    async def update_pattern_effectiveness(
        self,
        pattern_id: str,
        hit: bool
    ) -> None:
        """
        Update pattern effectiveness based on usage.
        
        Args:
            pattern_id: Pattern identifier
            hit: Whether the pattern was successful
        """
        if pattern_id in self.pattern_index:
            pattern = self.pattern_index[pattern_id]
            pattern.update_effectiveness(hit)
            
            # Persist update if database available
            if self.db:
                await self._update_pattern_stats(pattern)
    
    async def _update_pattern_stats(self, pattern: RegisteredPattern) -> None:
        """Update pattern statistics in database"""
        if not self.db:
            return
        
        try:
            async with self.db.connect() as conn:
                await conn.execute(
                    """
                    UPDATE learned_patterns
                    SET effectiveness = ?, hit_count = ?, miss_count = ?, last_used = ?
                    WHERE id = ?
                    """,
                    (
                        pattern.effectiveness,
                        pattern.hit_count,
                        pattern.miss_count,
                        pattern.last_used.isoformat() if pattern.last_used else None,
                        pattern.id
                    )
                )
                await conn.commit()
                
        except Exception as e:
            logger.error(f"Error updating pattern stats: {e}")
    
    async def disable_ineffective_patterns(
        self,
        effectiveness_threshold: float = 0.3,
        min_attempts: int = 10
    ) -> int:
        """
        Disable patterns that are ineffective.
        
        Args:
            effectiveness_threshold: Minimum effectiveness to keep enabled
            min_attempts: Minimum attempts before evaluating
            
        Returns:
            Number of patterns disabled
        """
        disabled_count = 0
        
        for patterns in self.patterns.values():
            for pattern in patterns:
                total_attempts = pattern.hit_count + pattern.miss_count
                
                if (total_attempts >= min_attempts and 
                    pattern.effectiveness < effectiveness_threshold and
                    pattern.enabled):
                    
                    pattern.enabled = False
                    disabled_count += 1
                    
                    logger.info(
                        f"Disabled ineffective pattern {pattern.id}: "
                        f"effectiveness={pattern.effectiveness:.2f}"
                    )
                    
                    if self.db:
                        await self._disable_pattern_in_db(pattern.id)
        
        return disabled_count
    
    async def _disable_pattern_in_db(self, pattern_id: str) -> None:
        """Disable pattern in database"""
        if not self.db:
            return
        
        try:
            async with self.db.connect() as conn:
                await conn.execute(
                    "UPDATE learned_patterns SET enabled = 0 WHERE id = ?",
                    (pattern_id,)
                )
                await conn.commit()
                
        except Exception as e:
            logger.error(f"Error disabling pattern in database: {e}")
    
    def get_statistics(self) -> dict[str, Any]:
        """Get registry statistics"""
        stats = {
            'total_patterns': len(self.pattern_index),
            'patterns_by_agent': {},
            'patterns_by_source': {
                'original': 0,
                'learned': 0,
                'manual': 0
            },
            'effectiveness': {
                'high': 0,  # > 0.8
                'medium': 0,  # 0.5 - 0.8
                'low': 0,  # < 0.5
                'untested': 0  # no attempts
            },
            'most_effective': [],
            'least_effective': []
        }
        
        # Count patterns by agent and source
        for agent, patterns in self.patterns.items():
            stats['patterns_by_agent'][agent] = len(patterns)
            
            for pattern in patterns:
                stats['patterns_by_source'][pattern.source] += 1
                
                # Categorize by effectiveness
                total_attempts = pattern.hit_count + pattern.miss_count
                if total_attempts == 0:
                    stats['effectiveness']['untested'] += 1
                elif pattern.effectiveness > 0.8:
                    stats['effectiveness']['high'] += 1
                elif pattern.effectiveness >= 0.5:
                    stats['effectiveness']['medium'] += 1
                else:
                    stats['effectiveness']['low'] += 1
        
        # Find most and least effective patterns
        tested_patterns = [
            p for p in self.pattern_index.values() 
            if (p.hit_count + p.miss_count) > 0
        ]
        
        if tested_patterns:
            sorted_patterns = sorted(tested_patterns, key=lambda x: x.effectiveness, reverse=True)
            stats['most_effective'] = [
                {'id': p.id, 'pattern': p.pattern, 'effectiveness': p.effectiveness}
                for p in sorted_patterns[:5]
            ]
            stats['least_effective'] = [
                {'id': p.id, 'pattern': p.pattern, 'effectiveness': p.effectiveness}
                for p in sorted_patterns[-5:]
            ]
        
        return stats