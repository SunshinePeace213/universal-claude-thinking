"""Repository for atomic analysis database operations."""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from .db import DatabaseConnection, JSONEncoder, generate_uuid


class AtomicAnalysisRepository:
    """Repository for atomic analysis CRUD operations."""

    def __init__(self, db: DatabaseConnection) -> None:
        """Initialize repository with database connection.

        Args:
            db: Database connection instance
        """
        self.db = db
        self._encoder = JSONEncoder()

    async def save_analysis(
        self,
        prompt_hash: str,
        structure: dict[str, str | None],
        quality_score: float,
        gaps: list[str],
        suggestions: list[str],
    ) -> str:
        """Save atomic analysis to database.

        Args:
            prompt_hash: Hash of the analyzed prompt
            structure: Extracted prompt structure
            quality_score: Quality score (1-10)
            gaps: Detected gaps in prompt
            suggestions: Enhancement suggestions

        Returns:
            ID of saved analysis
        """
        analysis_id = generate_uuid()

        async with self.db.connect() as conn:
            # Check if analysis already exists
            cursor = await conn.execute(
                "SELECT id, usage_count FROM atomic_analyses WHERE prompt_hash = ?",
                (prompt_hash,),
            )
            existing = await cursor.fetchone()

            if existing:
                # Update usage count instead of creating new
                await conn.execute(
                    "UPDATE atomic_analyses SET usage_count = usage_count + 1 WHERE id = ?",
                    (existing[0],),
                )
                await conn.commit()
                return existing[0]

            # Insert new analysis
            await conn.execute(
                """
                INSERT INTO atomic_analyses 
                (id, prompt_hash, structure, quality_score, gaps, suggestions, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    analysis_id,
                    prompt_hash,
                    self._encoder.encode(structure),
                    quality_score,
                    self._encoder.encode(gaps),
                    self._encoder.encode(suggestions),
                    datetime.now(timezone.utc),
                ),
            )
            await conn.commit()

        return analysis_id

    async def get_by_hash(self, prompt_hash: str) -> Dict[str, Any] | None:
        """Retrieve analysis by prompt hash.

        Args:
            prompt_hash: Hash of the prompt

        Returns:
            Analysis data or None if not found
        """
        async with self.db.connect() as conn:
            cursor = await conn.execute(
                """
                SELECT id, prompt_hash, structure, quality_score, gaps, 
                       suggestions, created_at, usage_count
                FROM atomic_analyses
                WHERE prompt_hash = ?
                """,
                (prompt_hash,),
            )
            row = await cursor.fetchone()

            if not row:
                return None

            return {
                "id": row[0],
                "prompt_hash": row[1],
                "structure": self._encoder.decode(row[2]),
                "quality_score": row[3],
                "gaps": self._encoder.decode(row[4]),
                "suggestions": self._encoder.decode(row[5]),
                "created_at": row[6],
                "usage_count": row[7],
            }

    async def increment_usage(self, prompt_hash: str) -> bool:
        """Increment usage count for an analysis.

        Args:
            prompt_hash: Hash of the prompt

        Returns:
            True if updated, False if not found
        """
        async with self.db.connect() as conn:
            cursor = await conn.execute(
                "UPDATE atomic_analyses SET usage_count = usage_count + 1 WHERE prompt_hash = ?",
                (prompt_hash,),
            )
            await conn.commit()
            return cursor.rowcount > 0

    async def get_recent(
        self, limit: int = 10, min_score: float | None = None
    ) -> List[Dict[str, Any]]:
        """Get recent analyses.

        Args:
            limit: Maximum number of results
            min_score: Minimum quality score filter

        Returns:
            List of recent analyses
        """
        query = """
            SELECT id, prompt_hash, structure, quality_score, gaps, 
                   suggestions, created_at, usage_count
            FROM atomic_analyses
        """
        params: list[Any] = []

        if min_score is not None:
            query += " WHERE quality_score >= ?"
            params.append(min_score)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        async with self.db.connect() as conn:
            cursor = await conn.execute(query, params)
            rows = await cursor.fetchall()

            return [
                {
                    "id": row[0],
                    "prompt_hash": row[1],
                    "structure": self._encoder.decode(row[2]),
                    "quality_score": row[3],
                    "gaps": self._encoder.decode(row[4]),
                    "suggestions": self._encoder.decode(row[5]),
                    "created_at": row[6],
                    "usage_count": row[7],
                }
                for row in rows
            ]

    async def cleanup_old(self, days: int = 30) -> int:
        """Remove old analyses that haven't been used recently.

        Args:
            days: Age threshold in days

        Returns:
            Number of deleted records
        """
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)

        async with self.db.connect() as conn:
            # Delete old records with low usage
            cursor = await conn.execute(
                """
                DELETE FROM atomic_analyses
                WHERE created_at < ? AND usage_count < 5
                """,
                (cutoff_date,),
            )
            await conn.commit()
            return cursor.rowcount

    async def get_statistics(self) -> Dict[str, Any]:
        """Get repository statistics.

        Returns:
            Dictionary with statistics
        """
        async with self.db.connect() as conn:
            # Total count
            cursor = await conn.execute("SELECT COUNT(*) FROM atomic_analyses")
            total_count = (await cursor.fetchone())[0]

            # Average score
            cursor = await conn.execute(
                "SELECT AVG(quality_score) FROM atomic_analyses"
            )
            avg_score = (await cursor.fetchone())[0] or 0

            # High quality count
            cursor = await conn.execute(
                "SELECT COUNT(*) FROM atomic_analyses WHERE quality_score >= 7.0"
            )
            high_quality_count = (await cursor.fetchone())[0]

            # Most used
            cursor = await conn.execute(
                """
                SELECT prompt_hash, usage_count 
                FROM atomic_analyses 
                ORDER BY usage_count DESC 
                LIMIT 1
                """
            )
            most_used = await cursor.fetchone()

            return {
                "total_analyses": total_count,
                "average_score": round(avg_score, 2) if avg_score else 0,
                "high_quality_count": high_quality_count,
                "high_quality_percentage": (
                    round((high_quality_count / total_count) * 100, 1)
                    if total_count > 0
                    else 0
                ),
                "most_used_hash": most_used[0] if most_used else None,
                "most_used_count": most_used[1] if most_used else 0,
            }