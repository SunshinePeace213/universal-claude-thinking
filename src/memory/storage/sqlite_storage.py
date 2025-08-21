"""
SQLite storage backend for memory system.

Implements persistent storage using SQLite with connection pooling,
vector storage support, and efficient batch operations.
"""

import asyncio
import json
import uuid
from collections import namedtuple
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import aiosqlite
import numpy as np

from src.memory.layers.base import MemoryItem
from src.memory.storage.base import StorageBackend

# Named tuple for search results
SearchResult = namedtuple('SearchResult', ['memory', 'similarity'])


class SQLiteStorage(StorageBackend):
    """
    SQLite storage backend implementation.
    
    Provides persistent storage with connection pooling, vector storage,
    and efficient batch operations for the memory system.
    """

    def __init__(self, db_path: str, pool_size: int = 10):
        """
        Initialize SQLite storage.
        
        Args:
            db_path: Path to SQLite database file
            pool_size: Number of connections in the pool
        """
        self.db_path = db_path
        self.pool_size = pool_size
        self._pool: list[aiosqlite.Connection] = []
        self._pool_lock = asyncio.Lock()
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize database and connection pool."""
        # Create database directory if needed
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        # Create connection pool
        for _ in range(self.pool_size):
            conn = await aiosqlite.connect(self.db_path)
            await conn.execute("PRAGMA journal_mode=WAL")  # Better concurrency
            await conn.execute("PRAGMA synchronous=NORMAL")  # Performance
            self._pool.append(conn)

        # Create schema
        await self._create_schema()
        self._initialized = True

    async def _create_schema(self) -> None:
        """Create database schema."""
        async with self._get_connection() as conn:
            # Main memories table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    memory_type TEXT CHECK (memory_type IN ('stm', 'wm', 'ltm', 'swarm')),
                    content TEXT NOT NULL,
                    embedding BLOB,
                    metadata TEXT,
                    effectiveness_score REAL DEFAULT 5.0,
                    usage_count INTEGER DEFAULT 0,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    promoted_from TEXT,
                    promoted_at TIMESTAMP,
                    promotion_reason TEXT
                )
            """)

            # Create indexes
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_memory 
                ON memories(user_id, memory_type)
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_effectiveness 
                ON memories(effectiveness_score DESC)
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_expiration 
                ON memories(expires_at)
            """)

            # Memory promotions tracking table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_promotions (
                    id TEXT PRIMARY KEY,
                    memory_id TEXT REFERENCES memories(id),
                    from_type TEXT,
                    to_type TEXT,
                    promotion_score REAL,
                    promoted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    reason TEXT
                )
            """)

            # Fallback vector storage table (if sqlite-vec not available)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_vectors (
                    id TEXT PRIMARY KEY,
                    embedding BLOB NOT NULL,
                    FOREIGN KEY (id) REFERENCES memories(id) ON DELETE CASCADE
                )
            """)

            await conn.commit()

    @asynccontextmanager
    async def _get_connection(self):
        """Get a connection from the pool."""
        # Wait for available connection
        conn = None
        while conn is None:
            async with self._pool_lock:
                if self._pool:
                    conn = self._pool.pop()
            if conn is None:
                # Wait a bit before retrying
                await asyncio.sleep(0.01)

        try:
            yield conn
        finally:
            async with self._pool_lock:
                self._pool.append(conn)

    async def store(self, memory: MemoryItem) -> None:
        """Store a memory item."""
        async with self._get_connection() as conn:
            # Serialize content and metadata to JSON
            content_json = json.dumps(memory.content)
            metadata_json = json.dumps(memory.metadata) if memory.metadata else None

            # Serialize embedding
            embedding_blob = None
            if memory.embedding is not None:
                embedding_blob = memory.embedding.tobytes()

            # Insert memory
            await conn.execute("""
                INSERT OR REPLACE INTO memories (
                    id, user_id, memory_type, content, embedding, metadata,
                    effectiveness_score, usage_count, last_accessed, expires_at,
                    created_at, promoted_from, promoted_at, promotion_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                memory.id,
                memory.user_id,
                memory.memory_type,
                content_json,
                embedding_blob,
                metadata_json,
                memory.effectiveness_score,
                memory.usage_count,
                memory.last_accessed.isoformat() if isinstance(memory.last_accessed, datetime) else memory.last_accessed,
                memory.expires_at.isoformat() if memory.expires_at and isinstance(memory.expires_at, datetime) else memory.expires_at,
                memory.created_at.isoformat() if isinstance(memory.created_at, datetime) else memory.created_at,
                memory.promoted_from,
                memory.promoted_at.isoformat() if memory.promoted_at and isinstance(memory.promoted_at, datetime) else memory.promoted_at,
                memory.promotion_reason
            ))

            # Store embedding in vectors table if present
            if memory.embedding is not None:
                await conn.execute("""
                    INSERT OR REPLACE INTO memory_vectors (id, embedding)
                    VALUES (?, ?)
                """, (memory.id, embedding_blob))

            await conn.commit()

    async def retrieve(self, memory_id: str) -> MemoryItem | None:
        """Retrieve a memory item by ID."""
        async with self._get_connection() as conn:
            cursor = await conn.execute("""
                SELECT id, user_id, memory_type, content, embedding, metadata,
                       effectiveness_score, usage_count, last_accessed, expires_at,
                       created_at, promoted_from, promoted_at, promotion_reason
                FROM memories
                WHERE id = ?
            """, (memory_id,))

            row = await cursor.fetchone()
            if not row:
                return None

            return self._row_to_memory(row)

    async def update(self, memory: MemoryItem) -> None:
        """Update an existing memory item."""
        # For update, we'll use the same logic as store
        await self.store(memory)

    async def delete(self, memory_id: str) -> None:
        """Delete a memory item."""
        async with self._get_connection() as conn:
            await conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            await conn.execute("DELETE FROM memory_vectors WHERE id = ?", (memory_id,))
            await conn.commit()

    async def list_by_user(
        self,
        user_id: str,
        memory_type: str | None = None,
        limit: int = 100
    ) -> list[MemoryItem]:
        """List memories for a specific user."""
        async with self._get_connection() as conn:
            if memory_type:
                cursor = await conn.execute("""
                    SELECT id, user_id, memory_type, content, embedding, metadata,
                           effectiveness_score, usage_count, last_accessed, expires_at,
                           created_at, promoted_from, promoted_at, promotion_reason
                    FROM memories
                    WHERE user_id = ? AND memory_type = ?
                    ORDER BY last_accessed DESC
                    LIMIT ?
                """, (user_id, memory_type, limit))
            else:
                cursor = await conn.execute("""
                    SELECT id, user_id, memory_type, content, embedding, metadata,
                           effectiveness_score, usage_count, last_accessed, expires_at,
                           created_at, promoted_from, promoted_at, promotion_reason
                    FROM memories
                    WHERE user_id = ?
                    ORDER BY last_accessed DESC
                    LIMIT ?
                """, (user_id, limit))

            rows = await cursor.fetchall()
            return [self._row_to_memory(row) for row in rows]

    async def search_by_embedding(
        self,
        embedding: np.ndarray,
        k: int = 10,
        min_similarity: float = 0.5,
        user_id: str | None = None,
        memory_type: str | None = None
    ) -> list[SearchResult]:
        """Search for similar memories using embedding similarity."""
        async with self._get_connection() as conn:
            # Build query based on filters
            query = """
                SELECT m.id, m.user_id, m.memory_type, m.content, m.embedding, m.metadata,
                       m.effectiveness_score, m.usage_count, m.last_accessed, m.expires_at,
                       m.created_at, m.promoted_from, m.promoted_at, m.promotion_reason,
                       v.embedding as vec_embedding
                FROM memories m
                LEFT JOIN memory_vectors v ON m.id = v.id
                WHERE v.embedding IS NOT NULL
            """

            params = []
            if user_id:
                query += " AND m.user_id = ?"
                params.append(user_id)
            if memory_type:
                query += " AND m.memory_type = ?"
                params.append(memory_type)

            cursor = await conn.execute(query, params)
            rows = await cursor.fetchall()

            # Calculate similarities
            results = []
            query_norm = embedding / np.linalg.norm(embedding)

            for row in rows:
                memory = self._row_to_memory(row[:-1])  # Exclude vec_embedding

                # Reconstruct embedding from blob
                if row[-1]:  # vec_embedding
                    stored_embedding = np.frombuffer(row[-1], dtype=np.float32)
                    stored_norm = stored_embedding / np.linalg.norm(stored_embedding)
                    similarity = float(np.dot(query_norm, stored_norm))

                    if similarity >= min_similarity:
                        results.append(SearchResult(memory, similarity))

            # Sort by similarity and return top k
            results.sort(key=lambda x: x.similarity, reverse=True)
            return results[:k]

    def _row_to_memory(self, row) -> MemoryItem:
        """Convert database row to MemoryItem."""
        # Parse JSON fields
        content = json.loads(row[3]) if row[3] else {}
        metadata = json.loads(row[5]) if row[5] else {}

        # Reconstruct embedding if present
        embedding = None
        if row[4]:
            embedding = np.frombuffer(row[4], dtype=np.float32)

        # Parse datetime fields
        last_accessed = datetime.fromisoformat(row[8]) if isinstance(row[8], str) else row[8]
        expires_at = datetime.fromisoformat(row[9]) if row[9] and isinstance(row[9], str) else row[9]
        created_at = datetime.fromisoformat(row[10]) if isinstance(row[10], str) else row[10]
        promoted_at = datetime.fromisoformat(row[12]) if row[12] and isinstance(row[12], str) else row[12]

        return MemoryItem(
            id=row[0],
            user_id=row[1],
            memory_type=row[2],
            content=content,
            embedding=embedding,
            metadata=metadata,
            effectiveness_score=row[6],
            usage_count=row[7],
            last_accessed=last_accessed,
            expires_at=expires_at,
            created_at=created_at,
            promoted_from=row[11],
            promoted_at=promoted_at,
            promotion_reason=row[13]
        )

    async def batch_store(self, memories: list[MemoryItem]) -> None:
        """Store multiple memory items in batch."""
        async with self._get_connection() as conn:
            for memory in memories:
                content_json = json.dumps(memory.content)
                metadata_json = json.dumps(memory.metadata) if memory.metadata else None
                embedding_blob = memory.embedding.tobytes() if memory.embedding is not None else None

                await conn.execute("""
                    INSERT OR REPLACE INTO memories (
                        id, user_id, memory_type, content, embedding, metadata,
                        effectiveness_score, usage_count, last_accessed, expires_at,
                        created_at, promoted_from, promoted_at, promotion_reason
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    memory.id,
                    memory.user_id,
                    memory.memory_type,
                    content_json,
                    embedding_blob,
                    metadata_json,
                    memory.effectiveness_score,
                    memory.usage_count,
                    memory.last_accessed.isoformat() if isinstance(memory.last_accessed, datetime) else memory.last_accessed,
                    memory.expires_at.isoformat() if memory.expires_at and isinstance(memory.expires_at, datetime) else memory.expires_at,
                    memory.created_at.isoformat() if isinstance(memory.created_at, datetime) else memory.created_at,
                    memory.promoted_from,
                    memory.promoted_at.isoformat() if memory.promoted_at and isinstance(memory.promoted_at, datetime) else memory.promoted_at,
                    memory.promotion_reason
                ))

                if memory.embedding is not None:
                    await conn.execute("""
                        INSERT OR REPLACE INTO memory_vectors (id, embedding)
                        VALUES (?, ?)
                    """, (memory.id, embedding_blob))

            await conn.commit()

    async def record_promotion(
        self,
        memory_id: str,
        from_type: str,
        to_type: str,
        promotion_score: float,
        reason: str
    ) -> None:
        """
        Record a memory promotion event.
        
        Args:
            memory_id: ID of the promoted memory
            from_type: Source memory type
            to_type: Target memory type
            promotion_score: Score at time of promotion
            reason: Reason for promotion
        """
        async with self._get_connection() as conn:
            promotion_id = str(uuid.uuid4())
            await conn.execute("""
                INSERT INTO memory_promotions (
                    id, memory_id, from_type, to_type, 
                    promotion_score, promoted_at, reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                promotion_id,
                memory_id,
                from_type,
                to_type,
                promotion_score,
                datetime.now().isoformat(),
                reason
            ))
            await conn.commit()

    async def get_promotion_history(self, memory_id: str) -> list[dict[str, Any]]:
        """
        Get promotion history for a memory.
        
        Args:
            memory_id: ID of the memory
            
        Returns:
            List of promotion events
        """
        async with self._get_connection() as conn:
            cursor = await conn.execute("""
                SELECT from_type, to_type, promotion_score, 
                       promoted_at, reason
                FROM memory_promotions
                WHERE memory_id = ?
                ORDER BY promoted_at ASC
            """, (memory_id,))

            rows = await cursor.fetchall()
            history = []
            for row in rows:
                history.append({
                    'from_type': row[0],
                    'to_type': row[1],
                    'promotion_score': row[2],
                    'promoted_at': row[3],
                    'reason': row[4]
                })
            return history

    async def cleanup_expired(self) -> int:
        """Remove expired memories from storage."""
        async with self._get_connection() as conn:
            # Get current time
            current_time = datetime.now().isoformat()

            # Count expired memories
            cursor = await conn.execute("""
                SELECT COUNT(*) FROM memories 
                WHERE expires_at IS NOT NULL AND expires_at < ?
            """, (current_time,))
            count = (await cursor.fetchone())[0]

            # Delete expired memories
            await conn.execute("""
                DELETE FROM memories 
                WHERE expires_at IS NOT NULL AND expires_at < ?
            """, (current_time,))

            # Also delete from vectors table
            await conn.execute("""
                DELETE FROM memory_vectors 
                WHERE id NOT IN (SELECT id FROM memories)
            """)

            await conn.commit()
            return count

    async def get_memories_by_type(self, memory_type: str) -> list[MemoryItem]:
        """
        Get all memories of a specific type.
        
        Args:
            memory_type: Type of memories to retrieve ("stm", "wm", "ltm", "swarm")
            
        Returns:
            List of memory items of the specified type
        """
        async with self._get_connection() as conn:
            cursor = await conn.execute("""
                SELECT id, user_id, memory_type, content, embedding, metadata,
                       effectiveness_score, usage_count, last_accessed, expires_at,
                       created_at, promoted_from, promoted_at, promotion_reason
                FROM memories
                WHERE memory_type = ?
                ORDER BY created_at DESC
            """, (memory_type,))

            rows = await cursor.fetchall()
            memories = []

            for row in rows:
                memory = self._row_to_memory(row)
                memories.append(memory)

            return memories

    async def update_memory_type(self, memory_id: str, from_type: str, to_type: str) -> None:
        """
        Update the type of a memory (for promotion).
        
        Args:
            memory_id: ID of the memory to update
            from_type: Current memory type (for verification)
            to_type: New memory type
        """
        async with self._get_connection() as conn:
            # Update memory type and promotion metadata
            await conn.execute("""
                UPDATE memories 
                SET memory_type = ?,
                    promoted_from = ?,
                    promoted_at = ?
                WHERE id = ? AND memory_type = ?
            """, (
                to_type,
                from_type,
                datetime.now().isoformat(),
                memory_id,
                from_type
            ))
            await conn.commit()

    async def get_memory(self, memory_id: str) -> MemoryItem | None:
        """
        Get a single memory by ID (alias for retrieve).
        
        Args:
            memory_id: ID of the memory to retrieve
            
        Returns:
            Memory item if found, None otherwise
        """
        return await self.retrieve(memory_id)

    async def update_expiry(self, memory_id: str, expires_at: datetime | None) -> None:
        """
        Update the expiry time of a memory.
        
        Args:
            memory_id: ID of the memory to update
            expires_at: New expiry time (None for no expiry)
        """
        async with self._get_connection() as conn:
            expiry_str = expires_at.isoformat() if expires_at else None
            await conn.execute("""
                UPDATE memories 
                SET expires_at = ?
                WHERE id = ?
            """, (expiry_str, memory_id))
            await conn.commit()

    async def get_memories_batch(self, memory_ids: list[str]) -> list[MemoryItem]:
        """
        Get multiple memories by their IDs.
        
        Args:
            memory_ids: List of memory IDs to retrieve
            
        Returns:
            List of memory items found
        """
        if not memory_ids:
            return []

        async with self._get_connection() as conn:
            # Create placeholders for SQL IN clause
            placeholders = ','.join('?' * len(memory_ids))

            cursor = await conn.execute(f"""
                SELECT id, user_id, memory_type, content, embedding, metadata,
                       effectiveness_score, usage_count, last_accessed, expires_at,
                       created_at, promoted_from, promoted_at, promotion_reason
                FROM memories
                WHERE id IN ({placeholders})
            """, memory_ids)

            rows = await cursor.fetchall()
            memories = []

            for row in rows:
                memory = self._row_to_memory(row)
                memories.append(memory)

            return memories

    async def close(self) -> None:
        """Close all connections in the pool."""
        async with self._pool_lock:
            for conn in self._pool:
                await conn.close()
            self._pool.clear()
        self._initialized = False
