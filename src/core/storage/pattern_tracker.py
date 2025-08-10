"""
Pattern Effectiveness Tracking System
Tracks and analyzes the effectiveness of classification patterns
Part of Story 1.2: Request Classification Engine with Delegation Integration
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path

import aiosqlite

logger = logging.getLogger(__name__)


@dataclass
class PatternStats:
    """Statistics for a single pattern"""
    pattern: str
    task_type: str
    match_count: int = 0
    correct_predictions: int = 0
    false_positives: int = 0
    avg_confidence: float = 0.0
    accuracy: float = 0.0
    last_matched: Optional[datetime] = None
    
    def update_accuracy(self) -> None:
        """Calculate accuracy percentage"""
        if self.match_count > 0:
            self.accuracy = (self.correct_predictions / self.match_count) * 100


@dataclass
class PatternEffectivenessReport:
    """Comprehensive pattern effectiveness report"""
    timestamp: datetime = field(default_factory=datetime.now)
    total_patterns: int = 0
    active_patterns: int = 0
    inactive_patterns: int = 0
    high_performing: List[PatternStats] = field(default_factory=list)
    low_performing: List[PatternStats] = field(default_factory=list)
    never_matched: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class PatternTracker:
    """
    Tracks and analyzes pattern effectiveness for continuous improvement.
    Provides insights into which patterns are working and which need adjustment.
    """
    
    def __init__(self, db_path: str = "data/delegation_metrics.db") -> None:
        """
        Initialize pattern tracker with database connection.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self._ensure_data_directory()
    
    def _ensure_data_directory(self) -> None:
        """Ensure data directory exists"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
    
    async def initialize(self) -> None:
        """Initialize pattern tracking tables"""
        async with aiosqlite.connect(self.db_path) as db:
            await self._create_tables(db)
            await db.commit()
    
    async def _create_tables(self, db) -> None:
        """Create pattern tracking tables if they don't exist"""
        
        # Pattern effectiveness table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS pattern_effectiveness (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern TEXT NOT NULL,
                task_type VARCHAR(20) NOT NULL,
                matched_text TEXT,
                predicted_type VARCHAR(20),
                actual_type VARCHAR(20),
                was_correct BOOLEAN,
                confidence DECIMAL(3,2),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(pattern, task_type, matched_text, created_at)
            )
        """)
        
        # Pattern statistics table (aggregated view)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS pattern_statistics (
                pattern TEXT PRIMARY KEY,
                task_type VARCHAR(20),
                total_matches INTEGER DEFAULT 0,
                correct_predictions INTEGER DEFAULT 0,
                false_positives INTEGER DEFAULT 0,
                avg_confidence DECIMAL(3,2),
                accuracy_percentage DECIMAL(5,2),
                last_matched TIMESTAMP,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Pattern recommendations table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS pattern_recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern TEXT NOT NULL,
                recommendation_type VARCHAR(50),
                recommendation TEXT,
                priority VARCHAR(20),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                addressed BOOLEAN DEFAULT 0
            )
        """)
        
        # Create indexes for performance
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_pattern_effectiveness_pattern 
            ON pattern_effectiveness(pattern, task_type)
        """)
        
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_pattern_effectiveness_created 
            ON pattern_effectiveness(created_at)
        """)
    
    async def track_pattern_match(
        self,
        pattern: str,
        task_type: str,
        matched_text: str,
        predicted_type: str,
        confidence: float,
        actual_type: Optional[str] = None
    ) -> None:
        """
        Track a pattern match event.
        
        Args:
            pattern: The regex pattern that matched
            task_type: The task type the pattern is associated with
            matched_text: The text that matched the pattern
            predicted_type: The predicted classification type
            confidence: Confidence score of the match
            actual_type: Actual type (if known) for accuracy tracking
        """
        was_correct = None
        if actual_type:
            was_correct = predicted_type == actual_type
        
        async with aiosqlite.connect(self.db_path) as db:
            await self._create_tables(db)
            
            # Insert pattern match event
            try:
                await db.execute("""
                    INSERT INTO pattern_effectiveness (
                        pattern, task_type, matched_text, predicted_type,
                        actual_type, was_correct, confidence
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    pattern[:200],  # Truncate long patterns
                    task_type,
                    matched_text[:500],  # Truncate long text
                    predicted_type,
                    actual_type,
                    was_correct,
                    confidence
                ))
                
                # Update aggregated statistics
                await self._update_pattern_statistics(db, pattern, task_type)
                
                await db.commit()
                
            except Exception as e:
                logger.error(f"Error tracking pattern match: {e}")
    
    async def _update_pattern_statistics(
        self,
        db,
        pattern: str,
        task_type: str
    ) -> None:
        """Update aggregated pattern statistics"""
        
        # Calculate statistics
        cursor = await db.execute("""
            SELECT 
                COUNT(*) as total_matches,
                SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as correct,
                SUM(CASE WHEN was_correct = 0 THEN 1 ELSE 0 END) as false_positives,
                AVG(confidence) as avg_confidence,
                MAX(created_at) as last_matched
            FROM pattern_effectiveness
            WHERE pattern = ? AND task_type = ?
        """, (pattern, task_type))
        
        row = await cursor.fetchone()
        
        if row and row[0] > 0:
            total_matches = row[0]
            correct = row[1] or 0
            false_positives = row[2] or 0
            avg_confidence = row[3] or 0
            last_matched = row[4]
            
            accuracy = (correct / total_matches * 100) if total_matches > 0 else 0
            
            # Update or insert statistics
            await db.execute("""
                INSERT OR REPLACE INTO pattern_statistics (
                    pattern, task_type, total_matches, correct_predictions,
                    false_positives, avg_confidence, accuracy_percentage,
                    last_matched, last_updated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                pattern, task_type, total_matches, correct,
                false_positives, avg_confidence, accuracy,
                last_matched
            ))
    
    async def get_pattern_effectiveness_report(
        self,
        days_back: int = 7,
        min_matches: int = 5
    ) -> PatternEffectivenessReport:
        """
        Generate a comprehensive pattern effectiveness report.
        
        Args:
            days_back: Number of days to analyze
            min_matches: Minimum matches for pattern to be considered
            
        Returns:
            PatternEffectivenessReport with analysis and recommendations
        """
        report = PatternEffectivenessReport()
        since_date = datetime.now() - timedelta(days=days_back)
        
        async with aiosqlite.connect(self.db_path) as db:
            await self._create_tables(db)
            
            # Get all pattern statistics
            cursor = await db.execute("""
                SELECT 
                    pattern, task_type, total_matches, correct_predictions,
                    false_positives, avg_confidence, accuracy_percentage,
                    last_matched
                FROM pattern_statistics
                WHERE last_matched >= ?
                ORDER BY accuracy_percentage DESC, total_matches DESC
            """, (since_date,))
            
            all_patterns = []
            async for row in cursor:
                stats = PatternStats(
                    pattern=row[0],
                    task_type=row[1],
                    match_count=row[2],
                    correct_predictions=row[3],
                    false_positives=row[4],
                    avg_confidence=row[5],
                    accuracy=row[6],
                    last_matched=datetime.fromisoformat(row[7]) if row[7] else None
                )
                all_patterns.append(stats)
            
            report.total_patterns = len(all_patterns)
            
            # Categorize patterns
            for pattern in all_patterns:
                if pattern.match_count >= min_matches:
                    report.active_patterns += 1
                    
                    if pattern.accuracy >= 90:
                        report.high_performing.append(pattern)
                    elif pattern.accuracy < 60:
                        report.low_performing.append(pattern)
                else:
                    report.inactive_patterns += 1
            
            # Generate recommendations
            report.recommendations = await self._generate_recommendations(
                report.high_performing,
                report.low_performing,
                all_patterns
            )
            
            # Find never-matched patterns (requires pattern library integration)
            # This would need to be implemented with the PatternLibrary class
            
        return report
    
    async def _generate_recommendations(
        self,
        high_performing: List[PatternStats],
        low_performing: List[PatternStats],
        all_patterns: List[PatternStats]
    ) -> List[str]:
        """Generate actionable recommendations based on pattern performance"""
        recommendations = []
        
        # Recommend removing consistently poor performers
        for pattern in low_performing:
            if pattern.match_count > 20 and pattern.accuracy < 50:
                recommendations.append(
                    f"Consider removing pattern '{pattern.pattern[:50]}...' "
                    f"for {pattern.task_type} (accuracy: {pattern.accuracy:.1f}%)"
                )
        
        # Recommend investigating high false positive patterns
        high_fp_patterns = [p for p in all_patterns 
                           if p.false_positives > p.correct_predictions]
        for pattern in high_fp_patterns[:3]:
            recommendations.append(
                f"Pattern '{pattern.pattern[:50]}...' has high false positives "
                f"({pattern.false_positives} vs {pattern.correct_predictions} correct)"
            )
        
        # Recommend patterns that might need refinement
        medium_performers = [p for p in all_patterns 
                            if 60 <= p.accuracy < 80 and p.match_count >= 10]
        if medium_performers:
            recommendations.append(
                f"Consider refining {len(medium_performers)} patterns "
                f"with 60-80% accuracy for better performance"
            )
        
        # Positive feedback for high performers
        if len(high_performing) > 5:
            recommendations.append(
                f"Excellent: {len(high_performing)} patterns showing >90% accuracy"
            )
        
        return recommendations
    
    async def get_pattern_trends(
        self,
        pattern: str,
        days_back: int = 30
    ) -> Dict[str, Any]:
        """
        Get performance trends for a specific pattern over time.
        
        Args:
            pattern: The pattern to analyze
            days_back: Number of days to analyze
            
        Returns:
            Dictionary with trend data
        """
        since_date = datetime.now() - timedelta(days=days_back)
        
        async with aiosqlite.connect(self.db_path) as db:
            await self._create_tables(db)
            
            cursor = await db.execute("""
                SELECT 
                    DATE(created_at) as date,
                    COUNT(*) as daily_matches,
                    SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as correct,
                    AVG(confidence) as avg_confidence
                FROM pattern_effectiveness
                WHERE pattern = ? AND created_at >= ?
                GROUP BY DATE(created_at)
                ORDER BY date
            """, (pattern, since_date))
            
            trends = {
                "pattern": pattern,
                "daily_data": [],
                "total_matches": 0,
                "overall_accuracy": 0
            }
            
            total_correct = 0
            async for row in cursor:
                daily_accuracy = (row[2] / row[1] * 100) if row[1] > 0 else 0
                trends["daily_data"].append({
                    "date": row[0],
                    "matches": row[1],
                    "accuracy": daily_accuracy,
                    "avg_confidence": row[3]
                })
                trends["total_matches"] += row[1]
                total_correct += row[2]
            
            if trends["total_matches"] > 0:
                trends["overall_accuracy"] = total_correct / trends["total_matches"] * 100
            
            return trends
    
    async def export_pattern_report(
        self,
        output_path: str = "pattern_effectiveness_report.json"
    ) -> None:
        """
        Export pattern effectiveness report to JSON file.
        
        Args:
            output_path: Path to save the report
        """
        report = await self.get_pattern_effectiveness_report()
        
        # Convert to dictionary for JSON serialization
        report_dict = {
            "timestamp": report.timestamp.isoformat(),
            "total_patterns": report.total_patterns,
            "active_patterns": report.active_patterns,
            "inactive_patterns": report.inactive_patterns,
            "high_performing": [asdict(p) for p in report.high_performing],
            "low_performing": [asdict(p) for p in report.low_performing],
            "recommendations": report.recommendations
        }
        
        # Handle datetime serialization
        def json_serial(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Type {type(obj)} not serializable")
        
        with open(output_path, 'w') as f:
            json.dump(report_dict, f, indent=2, default=json_serial)
        
        logger.info(f"Pattern effectiveness report exported to {output_path}")