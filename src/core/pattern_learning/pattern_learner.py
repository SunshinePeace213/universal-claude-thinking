"""
Pattern Learning Module
Tracks unmatched patterns and suggests new patterns based on fallback analysis
Part of Universal Claude Thinking v2 - Dynamic Pattern Learning Enhancement
"""

import asyncio
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from ..storage.db import DatabaseConnection

logger = logging.getLogger(__name__)


@dataclass
class PatternSuggestion:
    """Suggested pattern based on learning analysis"""
    pattern: str
    regex_pattern: str
    confidence: float
    frequency: int
    example_prompts: list[str] = field(default_factory=list)
    suggested_agent: str | None = None
    rationale: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'pattern': self.pattern,
            'regex_pattern': self.regex_pattern,
            'confidence': self.confidence,
            'frequency': self.frequency,
            'example_prompts': self.example_prompts[:3],  # Limit examples
            'suggested_agent': self.suggested_agent,
            'rationale': self.rationale,
            'created_at': datetime.now(UTC).isoformat()
        }


@dataclass
class LearningMetrics:
    """Metrics from pattern learning analysis"""
    total_fallbacks: int = 0
    unique_patterns: int = 0
    learning_rate: float = 0.0
    most_common_gaps: list[tuple[str, int]] = field(default_factory=list)
    agent_distribution: dict[str, int] = field(default_factory=dict)
    time_window_hours: int = 24


class PatternLearner:
    """
    Analyzes unmatched patterns to suggest new regex patterns.
    Learns from fallback delegations to improve pattern matching.
    """
    
    def __init__(self, db_connection: DatabaseConnection | None = None):
        """
        Initialize pattern learner.
        
        Args:
            db_connection: Database connection for storing/retrieving learning data
        """
        self.db = db_connection
        self.learned_patterns: list[PatternSuggestion] = []
        self.ngram_frequency: dict[str, int] = defaultdict(int)
        self.word_associations: dict[str, set[str]] = defaultdict(set)
        
    async def analyze_fallbacks(
        self, 
        time_window: timedelta = timedelta(hours=24),
        min_frequency: int = 3
    ) -> list[PatternSuggestion]:
        """
        Analyze recent fallback delegations to suggest new patterns.
        
        Args:
            time_window: Time window to analyze
            min_frequency: Minimum frequency for pattern suggestion
            
        Returns:
            List of suggested patterns
        """
        if not self.db:
            logger.warning("No database connection - cannot analyze fallbacks")
            return []
        
        suggestions = []
        
        try:
            # Fetch recent pattern learning opportunities
            since = datetime.now(UTC) - time_window
            async with self.db.connect() as conn:
                cursor = await conn.execute(
                    """
                    SELECT prompt, delegation_method, metadata 
                    FROM pattern_learning_opportunities
                    WHERE created_at > ? AND delegation_method = 'fallback'
                    ORDER BY created_at DESC
                    """,
                    (since.isoformat(),)
                )
                opportunities = await cursor.fetchall()
            
            if not opportunities:
                logger.info("No fallback delegations found in time window")
                return []
            
            # Extract prompts
            fallback_prompts = [row[0] for row in opportunities]
            
            # Analyze n-grams
            ngrams = self._extract_ngrams(fallback_prompts)
            
            # Find common patterns
            common_patterns = self._find_common_patterns(ngrams, min_frequency)
            
            # Generate regex suggestions
            for pattern, frequency in common_patterns:
                suggestion = self._create_pattern_suggestion(
                    pattern, 
                    frequency,
                    fallback_prompts
                )
                if suggestion:
                    suggestions.append(suggestion)
                    self.learned_patterns.append(suggestion)
            
            logger.info(f"Generated {len(suggestions)} pattern suggestions")
            
        except Exception as e:
            logger.error(f"Error analyzing fallbacks: {e}")
        
        return suggestions
    
    def _extract_ngrams(
        self, 
        prompts: list[str], 
        n_range: tuple[int, int] = (2, 5)
    ) -> Counter:
        """
        Extract n-grams from prompts.
        
        Args:
            prompts: List of prompts to analyze
            n_range: Range of n-gram sizes (min, max)
            
        Returns:
            Counter of n-gram frequencies
        """
        ngrams = Counter()
        
        for prompt in prompts:
            # Tokenize
            words = re.findall(r'\b\w+\b', prompt.lower())
            
            # Extract n-grams
            for n in range(n_range[0], min(n_range[1] + 1, len(words) + 1)):
                for i in range(len(words) - n + 1):
                    ngram = ' '.join(words[i:i+n])
                    ngrams[ngram] += 1
                    self.ngram_frequency[ngram] += 1
            
            # Build word associations
            for i, word in enumerate(words):
                if i > 0:
                    self.word_associations[word].add(words[i-1])
                if i < len(words) - 1:
                    self.word_associations[word].add(words[i+1])
        
        return ngrams
    
    def _find_common_patterns(
        self, 
        ngrams: Counter, 
        min_frequency: int
    ) -> list[tuple[str, int]]:
        """
        Find common patterns from n-grams.
        
        Args:
            ngrams: N-gram frequency counter
            min_frequency: Minimum frequency threshold
            
        Returns:
            List of (pattern, frequency) tuples
        """
        # Filter by minimum frequency
        common = [(pattern, freq) for pattern, freq in ngrams.items() 
                  if freq >= min_frequency]
        
        # Sort by frequency
        common.sort(key=lambda x: x[1], reverse=True)
        
        # Remove overlapping patterns (keep longer ones)
        filtered = []
        for pattern, freq in common:
            # Check if this pattern is a substring of an already selected pattern
            is_substring = any(pattern in selected[0] for selected in filtered)
            if not is_substring:
                filtered.append((pattern, freq))
        
        return filtered[:10]  # Return top 10 patterns
    
    def _create_pattern_suggestion(
        self,
        pattern: str,
        frequency: int,
        example_prompts: list[str]
    ) -> PatternSuggestion | None:
        """
        Create a pattern suggestion from a common pattern.
        
        Args:
            pattern: The common pattern found
            frequency: How often it appeared
            example_prompts: Example prompts containing the pattern
            
        Returns:
            PatternSuggestion or None if pattern is too generic
        """
        # Skip very generic patterns
        generic_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been'}
        if all(word in generic_words for word in pattern.split()):
            return None
        
        # Create regex pattern
        words = pattern.split()
        
        # Build regex with word boundaries and flexible spacing
        regex_parts = []
        for word in words:
            # Escape special regex characters
            escaped = re.escape(word)
            regex_parts.append(f"\\b{escaped}\\b")
        
        regex_pattern = r'\s+'.join(regex_parts)
        
        # Find examples containing this pattern
        examples = []
        for prompt in example_prompts:
            if re.search(regex_pattern, prompt, re.I):
                examples.append(prompt[:100])  # Truncate long prompts
                if len(examples) >= 3:
                    break
        
        # Determine suggested agent based on pattern context
        suggested_agent = self._suggest_agent_for_pattern(pattern)
        
        # Calculate confidence based on frequency and pattern specificity
        confidence = min(0.95, 0.5 + (frequency / 20) + (len(words) / 10))
        
        return PatternSuggestion(
            pattern=pattern,
            regex_pattern=regex_pattern,
            confidence=confidence,
            frequency=frequency,
            example_prompts=examples,
            suggested_agent=suggested_agent,
            rationale=f"Pattern '{pattern}' appeared {frequency} times in fallback delegations"
        )
    
    def _suggest_agent_for_pattern(self, pattern: str) -> str:
        """
        Suggest an agent based on pattern content.
        
        Args:
            pattern: The pattern to analyze
            
        Returns:
            Suggested agent identifier
        """
        pattern_lower = pattern.lower()
        
        # Agent keyword mappings
        agent_keywords = {
            'PE': ['enhance', 'improve', 'clarify', 'refine', 'quality'],
            'R1': ['research', 'search', 'find', 'lookup', 'investigate', 'information'],
            'A1': ['analyze', 'solve', 'debug', 'troubleshoot', 'reason', 'think'],
            'E1': ['evaluate', 'assess', 'review', 'check', 'validate', 'test'],
            'T1': ['execute', 'run', 'automate', 'tool', 'browser', 'api'],
            'W1': ['write', 'create', 'draft', 'compose', 'document', 'content'],
            'I1': ['interact', 'communicate', 'interface', 'explain', 'help']
        }
        
        # Score each agent based on keyword matches
        scores = {}
        for agent, keywords in agent_keywords.items():
            score = sum(1 for keyword in keywords if keyword in pattern_lower)
            if score > 0:
                scores[agent] = score
        
        # Return agent with highest score, or PE as default
        if scores:
            return max(scores, key=scores.get)
        return 'PE'
    
    async def get_learning_metrics(
        self,
        time_window: timedelta = timedelta(hours=24)
    ) -> LearningMetrics:
        """
        Get metrics about pattern learning.
        
        Args:
            time_window: Time window for metrics
            
        Returns:
            LearningMetrics object
        """
        metrics = LearningMetrics(time_window_hours=int(time_window.total_seconds() / 3600))
        
        if not self.db:
            return metrics
        
        try:
            since = datetime.now(UTC) - time_window
            
            async with self.db.connect() as conn:
                # Count total fallbacks
                cursor = await conn.execute(
                    """
                    SELECT COUNT(*) FROM pattern_learning_opportunities
                    WHERE created_at > ? AND delegation_method = 'fallback'
                    """,
                    (since.isoformat(),)
                )
                metrics.total_fallbacks = (await cursor.fetchone())[0]
                
                # Count successful pattern matches
                cursor = await conn.execute(
                    """
                    SELECT COUNT(*) FROM pattern_validations
                    WHERE created_at > ? AND matched = 1
                    """,
                    (since.isoformat(),)
                )
                successful_matches = (await cursor.fetchone())[0]
                
                # Calculate learning rate
                total = metrics.total_fallbacks + successful_matches
                if total > 0:
                    metrics.learning_rate = successful_matches / total
                
                # Get agent distribution
                cursor = await conn.execute(
                    """
                    SELECT agent, COUNT(*) as count 
                    FROM pattern_validations
                    WHERE created_at > ? AND agent IS NOT NULL
                    GROUP BY agent
                    ORDER BY count DESC
                    """,
                    (since.isoformat(),)
                )
                agent_dist = await cursor.fetchall()
                metrics.agent_distribution = dict(agent_dist)
                
                # Count unique patterns (approximation using prompt hashes)
                metrics.unique_patterns = len(self.learned_patterns)
                
        except Exception as e:
            logger.error(f"Error getting learning metrics: {e}")
        
        return metrics
    
    async def export_suggestions(self) -> dict[str, Any]:
        """
        Export learned pattern suggestions for review.
        
        Returns:
            Dictionary of suggestions organized by agent
        """
        export = {
            'metadata': {
                'generated_at': datetime.now(UTC).isoformat(),
                'total_suggestions': len(self.learned_patterns),
                'metrics': (await self.get_learning_metrics()).dict() if self.db else {}
            },
            'suggestions_by_agent': defaultdict(list)
        }
        
        for suggestion in self.learned_patterns:
            agent = suggestion.suggested_agent or 'UNASSIGNED'
            export['suggestions_by_agent'][agent].append(suggestion.to_dict())
        
        return dict(export)