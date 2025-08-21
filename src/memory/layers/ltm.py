"""
Long-Term Memory (LTM) implementation.

Provides permanent storage for valuable patterns and user preferences
with no expiration and high effectiveness requirements.
"""

import json
import logging
from typing import Any

import aiosqlite
import numpy as np

from ..storage.sqlite_storage import SQLiteStorage
from .base import MemoryItem, MemoryLayer, MemoryType

logger = logging.getLogger(__name__)


class LongTermMemory(MemoryLayer):
    """
    Long-Term Memory layer with permanent storage.
    
    Stores valuable patterns and user preferences permanently,
    with promotion from WM based on high effectiveness and usage.
    """

    def __init__(
        self,
        db_path: str = "data/memories/long_term_memory.db",
        max_size: int = 100000,
    ):
        """
        Initialize Long-Term Memory layer.
        
        Args:
            db_path: Path to SQLite database
            max_size: Maximum number of memories to store
        """
        super().__init__(
            layer_type=MemoryType.LTM,
            ttl=None,  # No expiration for LTM
            max_size=max_size
        )

        self.db_path = db_path
        self.memory_type = "ltm"  # String version for compatibility
        self.ttl_days = None  # No TTL for LTM
        self._connection: aiosqlite.Connection | None = None
        self.storage = SQLiteStorage(db_path)  # Use SQLiteStorage abstraction

    async def initialize(self) -> None:
        """Initialize the LTM layer and database."""
        if self._initialized:
            return

        logger.info(f"Initializing LTM with permanent storage at {self.db_path}")

        # Create database connection
        self._connection = await aiosqlite.connect(self.db_path)

        # Create table if not exists
        await self._create_schema()

        self._initialized = True

    async def _create_schema(self) -> None:
        """Create database schema for long-term memory."""
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS long_term_memory (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding BLOB,
                metadata TEXT,
                effectiveness_score REAL DEFAULT 8.0,
                usage_count INTEGER DEFAULT 5,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                promoted_from TEXT,
                promoted_at TIMESTAMP,
                promotion_reason TEXT,
                category TEXT,
                tags TEXT
            )
        """)

        # Create indices for efficient querying
        await self._connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_ltm_user 
            ON long_term_memory(user_id)
        """)

        await self._connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_ltm_effectiveness 
            ON long_term_memory(effectiveness_score DESC)
        """)

        await self._connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_ltm_usage 
            ON long_term_memory(usage_count DESC)
        """)

        await self._connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_ltm_category 
            ON long_term_memory(category)
        """)

        await self._connection.commit()

    async def store(self, memory: MemoryItem) -> str:
        """
        Store a memory in LTM.
        
        Args:
            memory: Memory item to store
            
        Returns:
            ID of stored memory
        """
        # No expiration for LTM
        memory.expires_at = None
        memory.memory_type = "ltm"

        # Ensure minimum requirements are met
        if memory.effectiveness_score < 8.0:
            logger.warning(f"Memory {memory.id} has low effectiveness score {memory.effectiveness_score} for LTM")
        if memory.usage_count < 5:
            logger.warning(f"Memory {memory.id} has low usage count {memory.usage_count} for LTM")

        # Serialize embedding
        embedding_blob = None
        if memory.embedding is not None:
            embedding_blob = memory.embedding.tobytes()

        # Serialize metadata and extract category/tags
        metadata_json = json.dumps(memory.metadata) if memory.metadata else None
        category = memory.metadata.get("category", "general") if memory.metadata else "general"
        tags = json.dumps(memory.metadata.get("tags", [])) if memory.metadata else "[]"

        # Store in database
        await self._connection.execute("""
            INSERT OR REPLACE INTO long_term_memory (
                id, user_id, content, embedding, metadata,
                effectiveness_score, usage_count,
                created_at, last_accessed,
                promoted_from, promoted_at, promotion_reason,
                category, tags
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            memory.promoted_from,
            memory.promoted_at,
            memory.promotion_reason,
            category,
            tags
        ))

        await self._connection.commit()

        logger.info(f"Stored permanent memory {memory.id} in LTM")
        return memory.id

    async def retrieve(
        self,
        memory_id: str | None = None,
        user_id: str | None = None,
        limit: int = 10,
        category: str | None = None
    ) -> MemoryItem | list[MemoryItem] | None:
        """
        Retrieve memory or memories from LTM.
        
        Args:
            memory_id: Specific memory ID to retrieve
            user_id: User ID to filter memories
            limit: Maximum number of memories to retrieve
            category: Category to filter by
            
        Returns:
            Single memory, list of memories, or None
        """
        if memory_id:
            # Retrieve specific memory
            cursor = await self._connection.execute("""
                SELECT * FROM long_term_memory WHERE id = ?
            """, (memory_id,))

            row = await cursor.fetchone()
            if row:
                memory = await self._row_to_memory(row)

                # Update access time and usage
                await self._connection.execute("""
                    UPDATE long_term_memory 
                    SET last_accessed = CURRENT_TIMESTAMP,
                        usage_count = usage_count + 1
                    WHERE id = ?
                """, (memory_id,))
                await self._connection.commit()

                return memory
            return None

        # Build query for multiple memories
        query = "SELECT * FROM long_term_memory WHERE 1=1"
        params = []

        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)

        if category:
            query += " AND category = ?"
            params.append(category)

        query += " ORDER BY effectiveness_score DESC, usage_count DESC LIMIT ?"
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

        # Add category and tags to metadata
        if row[12]:  # category column
            metadata["category"] = row[12]
        if row[13]:  # tags column
            metadata["tags"] = json.loads(row[13])

        # Parse content
        content = json.loads(row[2]) if isinstance(row[2], str) else row[2]

        return MemoryItem(
            id=row[0],
            user_id=row[1],
            memory_type="ltm",
            content=content,
            embedding=embedding,
            metadata=metadata,
            effectiveness_score=row[5],
            usage_count=row[6],
            created_at=row[7],
            last_accessed=row[8],
            expires_at=None,  # No expiration for LTM
            promoted_from=row[9],
            promoted_at=row[10],
            promotion_reason=row[11]
        )

    async def delete(self, memory_id: str) -> bool:
        """
        Delete a memory from LTM.
        
        Note: Should be used sparingly as LTM contains valuable patterns.
        
        Args:
            memory_id: ID of memory to delete
            
        Returns:
            True if deleted, False if not found
        """
        # Log warning as LTM deletions should be rare
        logger.warning(f"Deleting permanent memory {memory_id} from LTM")

        cursor = await self._connection.execute("""
            DELETE FROM long_term_memory WHERE id = ?
        """, (memory_id,))

        await self._connection.commit()

        deleted = cursor.rowcount > 0
        if deleted:
            logger.info(f"Deleted memory {memory_id} from LTM")
        return deleted

    async def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        threshold: float = 0.85
    ) -> list[MemoryItem]:
        """
        Search for similar memories using embedding similarity.
        
        Args:
            query_embedding: Query vector for similarity search
            k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of similar memories
        """
        # Retrieve all memories with embeddings
        cursor = await self._connection.execute("""
            SELECT * FROM long_term_memory 
            WHERE embedding IS NOT NULL
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
        LTM has no expiration, so this always returns 0.
        
        Returns:
            Always 0 (no memories expire in LTM)
        """
        return 0

    async def get_statistics_by_category(self) -> dict[str, dict[str, Any]]:
        """
        Get statistics grouped by category.
        
        Returns:
            Dictionary of statistics per category
        """
        cursor = await self._connection.execute("""
            SELECT 
                category,
                COUNT(*) as count,
                AVG(effectiveness_score) as avg_effectiveness,
                AVG(usage_count) as avg_usage,
                MAX(usage_count) as max_usage
            FROM long_term_memory
            GROUP BY category
        """)

        rows = await cursor.fetchall()

        stats = {}
        for row in rows:
            stats[row[0]] = {
                "count": row[1],
                "avg_effectiveness": row[2],
                "avg_usage": row[3],
                "max_usage": row[4]
            }

        return stats

    async def close(self) -> None:
        """Clean up LTM resources."""
        self._initialized = False
        if self._connection:
            await self._connection.close()
        logger.info("LTM closed")
