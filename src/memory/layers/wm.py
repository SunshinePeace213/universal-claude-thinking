"""
Working Memory (WM) implementation.

Provides a 7-day TTL memory layer with SQLite persistence for
recent patterns and frequently accessed information.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta

import aiosqlite
import numpy as np

from ..storage.sqlite_storage import SQLiteStorage
from .base import MemoryItem, MemoryLayer, MemoryType

logger = logging.getLogger(__name__)


class WorkingMemory(MemoryLayer):
    """
    Working Memory layer with 7-day TTL.
    
    Uses SQLite for persistent storage of recent patterns and
    frequently accessed memories with automatic promotion from STM.
    """

    def __init__(
        self,
        db_path: str = "data/memories/working_memory.db",
        ttl_days: float = 7.0,
        max_size: int = 10000,
    ):
        """
        Initialize Working Memory layer.
        
        Args:
            db_path: Path to SQLite database
            ttl_days: Time-to-live in days (default: 7)
            max_size: Maximum number of memories to store
        """
        super().__init__(
            layer_type=MemoryType.WM,
            ttl=timedelta(days=ttl_days),
            max_size=max_size
        )

        self.db_path = db_path
        self.ttl_days = ttl_days
        self.memory_type = "wm"  # String version for compatibility
        self._connection: aiosqlite.Connection | None = None
        self.storage = SQLiteStorage(db_path)  # Use SQLiteStorage abstraction

    async def initialize(self) -> None:
        """Initialize the WM layer and database."""
        if self._initialized:
            return

        logger.info(f"Initializing WM with {self.ttl_days}d TTL at {self.db_path}")

        # Create database connection
        self._connection = await aiosqlite.connect(self.db_path)

        # Create table if not exists
        await self._create_schema()

        # Start background cleanup task
        asyncio.create_task(self._cleanup_task())

        self._initialized = True

    async def _create_schema(self) -> None:
        """Create database schema for working memory."""
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS working_memory (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding BLOB,
                metadata TEXT,
                effectiveness_score REAL DEFAULT 5.0,
                usage_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                promoted_from TEXT,
                promoted_at TIMESTAMP,
                promotion_reason TEXT
            )
        """)

        # Create indices
        await self._connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_wm_user 
            ON working_memory(user_id)
        """)

        await self._connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_wm_expires 
            ON working_memory(expires_at)
        """)

        await self._connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_wm_effectiveness 
            ON working_memory(effectiveness_score DESC)
        """)

        await self._connection.commit()

    async def store(self, memory: MemoryItem) -> str:
        """
        Store a memory in WM.
        
        Args:
            memory: Memory item to store
            
        Returns:
            ID of stored memory
        """
        # Set TTL if not already set
        if memory.expires_at is None:
            memory.set_ttl(ttl_days=self.ttl_days)

        # Set memory type
        memory.memory_type = "wm"

        # Serialize embedding
        embedding_blob = None
        if memory.embedding is not None:
            embedding_blob = memory.embedding.tobytes()

        # Serialize metadata
        metadata_json = json.dumps(memory.metadata) if memory.metadata else None

        # Store in database
        await self._connection.execute("""
            INSERT OR REPLACE INTO working_memory (
                id, user_id, content, embedding, metadata,
                effectiveness_score, usage_count,
                created_at, last_accessed, expires_at,
                promoted_from, promoted_at, promotion_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            memory.id,
            memory.user_id,
            json.dumps(memory.content),
            embedding_blob,
            metadata_json,
            memory.effectiveness_score,
            memory.usage_count,
            memory.created_at,
            memory.last_accessed,
            memory.expires_at,
            memory.promoted_from,
            memory.promoted_at,
            memory.promotion_reason
        ))

        await self._connection.commit()

        logger.debug(f"Stored memory {memory.id} in WM")
        return memory.id

    async def retrieve(
        self,
        memory_id: str | None = None,
        user_id: str | None = None,
        limit: int = 10
    ) -> MemoryItem | list[MemoryItem] | None:
        """
        Retrieve memory or memories from WM.
        
        Args:
            memory_id: Specific memory ID to retrieve
            user_id: User ID to filter memories
            limit: Maximum number of memories to retrieve
            
        Returns:
            Single memory, list of memories, or None
        """
        if memory_id:
            # Retrieve specific memory
            cursor = await self._connection.execute("""
                SELECT * FROM working_memory WHERE id = ?
            """, (memory_id,))

            row = await cursor.fetchone()
            if row:
                memory = await self._row_to_memory(row)

                # Update access time
                await self._connection.execute("""
                    UPDATE working_memory 
                    SET last_accessed = CURRENT_TIMESTAMP,
                        usage_count = usage_count + 1
                    WHERE id = ?
                """, (memory_id,))
                await self._connection.commit()

                return memory
            return None

        # Retrieve memories by user_id
        query = """
            SELECT * FROM working_memory 
            WHERE expires_at > CURRENT_TIMESTAMP
        """
        params = []

        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)

        query += " ORDER BY last_accessed DESC LIMIT ?"
        params.append(limit)

        cursor = await self._connection.execute(query, params)
        rows = await cursor.fetchall()

        memories = []
        for row in rows:
            memory = await self._row_to_memory(row)
            memories.append(memory)

        return memories if memories else []

    async def _row_to_memory(self, row: tuple) -> MemoryItem:
        """Convert database row to MemoryItem."""
        # Parse embedding
        embedding = None
        if row[3]:  # embedding column
            embedding = np.frombuffer(row[3], dtype=np.float32)

        # Parse metadata
        metadata = json.loads(row[4]) if row[4] else {}

        # Parse content
        content = json.loads(row[2]) if isinstance(row[2], str) else row[2]

        return MemoryItem(
            id=row[0],
            user_id=row[1],
            memory_type="wm",
            content=content,
            embedding=embedding,
            metadata=metadata,
            effectiveness_score=row[5],
            usage_count=row[6],
            created_at=row[7],
            last_accessed=row[8],
            expires_at=row[9],
            promoted_from=row[10],
            promoted_at=row[11],
            promotion_reason=row[12]
        )

    async def delete(self, memory_id: str) -> bool:
        """
        Delete a memory from WM.
        
        Args:
            memory_id: ID of memory to delete
            
        Returns:
            True if deleted, False if not found
        """
        cursor = await self._connection.execute("""
            DELETE FROM working_memory WHERE id = ?
        """, (memory_id,))

        await self._connection.commit()

        deleted = cursor.rowcount > 0
        if deleted:
            logger.debug(f"Deleted memory {memory_id} from WM")
        return deleted

    async def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        threshold: float = 0.85
    ) -> list[MemoryItem]:
        """
        Search for similar memories using embedding similarity.
        
        Note: This is a simplified implementation. In production,
        you'd want to use vector indexing for efficiency.
        
        Args:
            query_embedding: Query vector for similarity search
            k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of similar memories
        """
        # Retrieve all non-expired memories with embeddings
        cursor = await self._connection.execute("""
            SELECT * FROM working_memory 
            WHERE expires_at > CURRENT_TIMESTAMP 
            AND embedding IS NOT NULL
        """)

        rows = await cursor.fetchall()

        if not rows:
            return []

        # Normalize query embedding
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_embedding = query_embedding / query_norm

        # Calculate similarities
        similarities = []
        for row in rows:
            memory = await self._row_to_memory(row)

            if memory.embedding is None:
                continue

            # Normalize stored embedding
            norm = np.linalg.norm(memory.embedding)
            if norm > 0:
                embedding = memory.embedding / norm

                # Cosine similarity
                similarity = float(np.dot(query_embedding, embedding))

                if similarity >= threshold:
                    similarities.append((memory, similarity))

        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [mem for mem, _ in similarities[:k]]

    async def cleanup_expired(self) -> int:
        """
        Remove expired memories from WM.
        
        Returns:
            Number of memories removed
        """
        current_time = datetime.now().isoformat()

        cursor = await self._connection.execute("""
            DELETE FROM working_memory 
            WHERE expires_at IS NOT NULL AND expires_at <= ?
        """, (current_time,))

        await self._connection.commit()

        deleted = cursor.rowcount
        if deleted > 0:
            logger.info(f"Cleaned up {deleted} expired memories from WM")

        return deleted if deleted > 0 else 0

    async def _cleanup_task(self) -> None:
        """Background task to periodically clean up expired memories."""
        while self._initialized:
            try:
                await asyncio.sleep(3600)  # Check every hour
                await self.cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in WM cleanup task: {e}")

    async def get_candidates_for_promotion(
        self,
        min_effectiveness: float = 8.0,
        min_usage: int = 5
    ) -> list[MemoryItem]:
        """
        Get memories that are candidates for promotion to LTM.
        
        Args:
            min_effectiveness: Minimum effectiveness score for promotion
            min_usage: Minimum usage count for promotion
            
        Returns:
            List of memories eligible for promotion
        """
        cursor = await self._connection.execute("""
            SELECT * FROM working_memory 
            WHERE effectiveness_score > ? 
            AND usage_count > ?
            AND expires_at > CURRENT_TIMESTAMP
            ORDER BY effectiveness_score DESC
        """, (min_effectiveness, min_usage))

        rows = await cursor.fetchall()

        candidates = []
        for row in rows:
            memory = await self._row_to_memory(row)
            candidates.append(memory)

        return candidates

    async def close(self) -> None:
        """Clean up WM resources."""
        self._initialized = False
        if self._connection:
            await self._connection.close()
        logger.info("WM closed")
