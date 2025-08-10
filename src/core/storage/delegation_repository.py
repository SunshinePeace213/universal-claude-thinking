"""
Delegation Metrics Repository
Async database operations for storing delegation and classification metrics
Part of Story 1.2: Request Classification Engine with Delegation Integration
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import aiosqlite

logger = logging.getLogger(__name__)


class DelegationRepository:
    """
    Repository for delegation metrics and classification history.
    Uses async SQLite for non-blocking database operations.
    """

    def __init__(self, db_path: str = "data/delegation_metrics.db"):
        """
        Initialize repository with database path.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._ensure_data_directory()

    def _ensure_data_directory(self):
        """Ensure data directory exists"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    async def initialize(self):
        """Initialize database and run migrations"""
        async with aiosqlite.connect(self.db_path) as db:
            # Create tables directly for in-memory databases and tests
            await self._create_tables(db)
            await db.commit()

    async def _create_tables(self, db):
        """Create required tables if they don't exist"""
        # Create delegation_metrics table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS delegation_metrics (
                id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
                request_id TEXT NOT NULL,
                classification_type VARCHAR(20),
                confidence_score DECIMAL(3,2),
                delegation_method VARCHAR(20),
                selected_agent VARCHAR(50),
                stage1_latency_ms INTEGER,
                stage2_latency_ms INTEGER,
                stage3_latency_ms INTEGER,
                total_latency_ms INTEGER,
                success BOOLEAN DEFAULT 1,
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create classification_history table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS classification_history (
                id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
                prompt_hash TEXT,
                prompt_text TEXT,
                predicted_type VARCHAR(20),
                actual_type VARCHAR(20),
                confidence DECIMAL(3,2),
                correct BOOLEAN,
                patterns_matched TEXT,
                processing_time_ms INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create confidence_factors table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS confidence_factors (
                id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
                request_id TEXT NOT NULL,
                classification_score DECIMAL(3,2),
                keyword_match_score DECIMAL(3,2),
                semantic_similarity_score DECIMAL(3,2),
                context_quality_score DECIMAL(3,2),
                input_clarity_score DECIMAL(3,2),
                overall_confidence DECIMAL(3,2),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create agent_performance table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS agent_performance (
                id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
                agent_name VARCHAR(50) NOT NULL,
                task_type VARCHAR(20),
                total_delegations INTEGER DEFAULT 0,
                successful_completions INTEGER DEFAULT 0,
                avg_confidence DECIMAL(3,2),
                avg_processing_time_ms INTEGER,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(agent_name, task_type)
            )
        """)

    async def save_delegation_metrics(self,
                                     request_id: str,
                                     classification_type: str,
                                     confidence_score: float,
                                     delegation_method: str,
                                     selected_agent: str,
                                     stage_latencies: dict[str, float],
                                     total_latency: float,
                                     success: bool = True,
                                     error_message: str | None = None) -> str:
        """
        Save delegation metrics to database.
        
        Returns:
            ID of the saved record
        """
        async with aiosqlite.connect(self.db_path) as db:
            # Ensure tables exist (needed for in-memory databases)
            await self._create_tables(db)

            cursor = await db.execute("""
                INSERT INTO delegation_metrics (
                    request_id, classification_type, confidence_score,
                    delegation_method, selected_agent,
                    stage1_latency_ms, stage2_latency_ms, stage3_latency_ms,
                    total_latency_ms, success, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                request_id, classification_type, confidence_score,
                delegation_method, selected_agent,
                stage_latencies.get('keyword', 0),
                stage_latencies.get('semantic', 0),
                stage_latencies.get('fallback', 0),
                total_latency, success, error_message
            ))

            await db.commit()
            return str(cursor.lastrowid)

    async def save_classification_history(self,
                                        prompt_text: str,
                                        predicted_type: str,
                                        confidence: float,
                                        patterns_matched: list[str],
                                        processing_time_ms: float,
                                        actual_type: str | None = None) -> str:
        """
        Save classification history for accuracy tracking.
        
        Returns:
            ID of the saved record
        """
        import hashlib
        prompt_hash = hashlib.sha256(prompt_text.encode()).hexdigest()[:16]

        correct = None
        if actual_type:
            correct = predicted_type == actual_type

        async with aiosqlite.connect(self.db_path) as db:
            # Ensure tables exist (needed for in-memory databases)
            await self._create_tables(db)

            cursor = await db.execute("""
                INSERT INTO classification_history (
                    prompt_hash, prompt_text, predicted_type,
                    actual_type, confidence, correct,
                    patterns_matched, processing_time_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                prompt_hash, prompt_text, predicted_type,
                actual_type, confidence, correct,
                json.dumps(patterns_matched), processing_time_ms
            ))

            await db.commit()
            return str(cursor.lastrowid)

    async def save_confidence_factors(self,
                                     request_id: str,
                                     factors: dict[str, float],
                                     overall_confidence: float):
        """Save confidence factor breakdown"""
        async with aiosqlite.connect(self.db_path) as db:
            # Ensure tables exist (needed for in-memory databases)
            await self._create_tables(db)

            await db.execute("""
                INSERT INTO confidence_factors (
                    request_id, classification_score, keyword_match_score,
                    semantic_similarity_score, context_quality_score,
                    input_clarity_score, overall_confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                request_id,
                factors.get('classification', 0),
                factors.get('keyword_match', 0),
                factors.get('semantic_similarity', 0),
                factors.get('context_quality', 0),
                factors.get('input_clarity', 0),
                overall_confidence
            ))
            await db.commit()

    async def get_classification_accuracy(self,
                                         task_type: str | None = None,
                                         days_back: int = 7) -> dict[str, Any]:
        """
        Get classification accuracy metrics.
        
        Args:
            task_type: Optional specific task type to filter
            days_back: Number of days to look back
            
        Returns:
            Dictionary with accuracy metrics
        """
        since_date = datetime.now() - timedelta(days=days_back)

        async with aiosqlite.connect(self.db_path) as db:
            # Ensure tables exist (needed for in-memory databases)
            await self._create_tables(db)

            if task_type:
                cursor = await db.execute("""
                    SELECT 
                        COUNT(*) as total,
                        SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) as correct,
                        AVG(confidence) as avg_confidence,
                        AVG(processing_time_ms) as avg_time
                    FROM classification_history
                    WHERE predicted_type = ? AND created_at >= ?
                """, (task_type, since_date))
            else:
                cursor = await db.execute("""
                    SELECT 
                        COUNT(*) as total,
                        SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) as correct,
                        AVG(confidence) as avg_confidence,
                        AVG(processing_time_ms) as avg_time
                    FROM classification_history
                    WHERE created_at >= ?
                """, (since_date,))

            row = await cursor.fetchone()

            if row and row[0] > 0:
                return {
                    'total_predictions': row[0],
                    'correct_predictions': row[1] or 0,
                    'accuracy_percentage': (row[1] or 0) / row[0] * 100,
                    'avg_confidence': row[2] or 0,
                    'avg_processing_time_ms': row[3] or 0
                }
            else:
                return {
                    'total_predictions': 0,
                    'correct_predictions': 0,
                    'accuracy_percentage': 0,
                    'avg_confidence': 0,
                    'avg_processing_time_ms': 0
                }

    async def get_delegation_distribution(self, days_back: int = 7) -> list[dict[str, Any]]:
        """Get distribution of delegation methods"""
        since_date = datetime.now() - timedelta(days=days_back)

        async with aiosqlite.connect(self.db_path) as db:
            # Ensure tables exist (needed for in-memory databases)
            await self._create_tables(db)

            cursor = await db.execute("""
                SELECT 
                    delegation_method,
                    COUNT(*) as count,
                    AVG(confidence_score) as avg_confidence,
                    AVG(total_latency_ms) as avg_latency
                FROM delegation_metrics
                WHERE created_at >= ?
                GROUP BY delegation_method
            """, (since_date,))

            rows = await cursor.fetchall()

            return [
                {
                    'method': row[0],
                    'count': row[1],
                    'avg_confidence': row[2],
                    'avg_latency_ms': row[3]
                }
                for row in rows
            ]

    async def get_agent_performance(self) -> list[dict[str, Any]]:
        """Get agent performance summary"""
        async with aiosqlite.connect(self.db_path) as db:
            # Ensure tables exist (needed for in-memory databases)
            await self._create_tables(db)

            cursor = await db.execute("""
                SELECT * FROM v_agent_performance_summary
            """)

            columns = [desc[0] for desc in cursor.description]
            rows = await cursor.fetchall()

            return [
                dict(zip(columns, row, strict=False))
                for row in rows
            ]

    async def get_hourly_metrics(self, hours_back: int = 24) -> list[dict[str, Any]]:
        """Get hourly delegation metrics"""
        since_date = datetime.now() - timedelta(hours=hours_back)

        async with aiosqlite.connect(self.db_path) as db:
            # Ensure tables exist (needed for in-memory databases)
            await self._create_tables(db)

            cursor = await db.execute("""
                SELECT * FROM v_hourly_metrics
                WHERE hour >= ?
                ORDER BY hour DESC
            """, (since_date.strftime('%Y-%m-%d %H:00:00'),))

            columns = [desc[0] for desc in cursor.description]
            rows = await cursor.fetchall()

            return [
                dict(zip(columns, row, strict=False))
                for row in rows
            ]

    async def update_classification_actual_type(self,
                                               classification_id: str,
                                               actual_type: str):
        """Update classification with actual type for accuracy tracking"""
        async with aiosqlite.connect(self.db_path) as db:
            # Ensure tables exist (needed for in-memory databases)
            await self._create_tables(db)

            # Get the predicted type
            cursor = await db.execute(
                "SELECT predicted_type FROM classification_history WHERE id = ?",
                (classification_id,)
            )
            row = await cursor.fetchone()

            if row:
                predicted_type = row[0]
                correct = predicted_type == actual_type

                await db.execute("""
                    UPDATE classification_history
                    SET actual_type = ?, correct = ?
                    WHERE id = ?
                """, (actual_type, correct, classification_id))

                await db.commit()
