"""
SQLite-vec Vector Storage Backend.

Implements efficient vector storage and similarity search using sqlite-vec
for 4096-dimensional embeddings with cosine similarity matching.
"""

import asyncio
import json
import logging
import sqlite3
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import aiosqlite
import numpy as np
import sqlite_vec
from sqlite_vec import serialize_float32

logger = logging.getLogger(__name__)


@dataclass
class VectorSearchResult:
    """Result from vector similarity search."""
    
    id: int
    distance: float
    similarity: float  # Cosine similarity (1 - distance)
    metadata: Optional[Dict[str, Any]] = None
    embedding: Optional[np.ndarray] = None
    content: Optional[str] = None  # Content of the matched example


class VectorStore:
    """
    SQLite-vec based vector storage for 4096-dimensional embeddings.
    
    Provides CRUD operations and cosine similarity search with efficient
    indexing for <100ms search latency.
    """
    
    def __init__(
        self,
        db_path: Union[str, Path] = ":memory:",
        dimension: int = 4096,
        similarity_threshold: float = 0.85,
        connection_pool_size: int = 5,
    ) -> None:
        """
        Initialize the vector store.
        
        Args:
            db_path: Path to SQLite database or ':memory:' for in-memory
            dimension: Vector dimension (4096 for Qwen3 embeddings)
            similarity_threshold: Minimum cosine similarity for matches
            connection_pool_size: Number of concurrent connections
        """
        self.db_path = str(db_path)
        self.dimension = dimension
        self.similarity_threshold = similarity_threshold
        self.pool_size = connection_pool_size
        
        self._connection_pool: List[aiosqlite.Connection] = []
        self._initialized = False
        self._vec0_available = False  # Will be set during initialization
        
    async def initialize(self) -> None:
        """Initialize the database and create tables."""
        if self._initialized:
            return
            
        logger.info(f"Initializing vector store at {self.db_path}")
        
        # Create connection pool
        for _ in range(self.pool_size):
            conn = await aiosqlite.connect(self.db_path)
            
            # Try to load sqlite-vec extension if supported
            try:
                # Check if extension loading is available
                if hasattr(conn._conn, 'enable_load_extension'):
                    await conn.enable_load_extension(True)
                    conn._conn.load_extension(sqlite_vec.loadable_path())
                    await conn.enable_load_extension(False)
                else:
                    # Fallback: Load extension directly if possible
                    sqlite_vec.load(conn._conn)
            except (AttributeError, Exception) as e:
                logger.warning(f"Could not load sqlite-vec extension: {e}")
                # Continue without extension - tests will use mock data
            
            self._connection_pool.append(conn)
            
        # Initialize schema with first connection
        async with self._get_connection() as conn:
            await self._create_schema(conn)
            
        self._initialized = True
        logger.info(f"Vector store initialized with {self.dimension}D vectors")
        
    async def _create_schema(self, conn: aiosqlite.Connection) -> None:
        """Create database schema with vec0 virtual tables."""
        # Try to create virtual table for vector storage
        try:
            await conn.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_vectors USING vec0(
                    vector_id INTEGER PRIMARY KEY,
                    embedding FLOAT[{self.dimension}]
                )
            """)
            self._vec0_available = True
        except sqlite3.OperationalError as e:
            if "no such module: vec0" in str(e):
                # Fallback to regular table for testing without vec0
                logger.warning("vec0 module not available, using fallback table schema")
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS memory_vectors (
                        vector_id INTEGER PRIMARY KEY,
                        embedding BLOB
                    )
                """)
                self._vec0_available = False
            else:
                raise
        
        # Create metadata table for additional information
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS vector_metadata (
                vector_id INTEGER PRIMARY KEY,
                content TEXT,
                metadata JSON,
                usage_count INTEGER DEFAULT 0,
                effectiveness_score REAL DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (vector_id) REFERENCES memory_vectors(vector_id)
            )
        """)
        
        # Create index for metadata queries
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_effectiveness 
            ON vector_metadata(effectiveness_score DESC)
        """)
        
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_usage 
            ON vector_metadata(usage_count DESC)
        """)
        
        await conn.commit()
        
    @asynccontextmanager
    async def _get_connection(self):
        """Get a connection from the pool."""
        if not self._connection_pool:
            await self.initialize()
            
        # Wait if pool is temporarily empty due to concurrent access
        retry_count = 0
        while not self._connection_pool and retry_count < 100:
            await asyncio.sleep(0.01)  # Wait 10ms
            retry_count += 1
            
        if not self._connection_pool:
            raise RuntimeError("Connection pool exhausted")
            
        conn = self._connection_pool.pop(0)
        try:
            yield conn
        finally:
            self._connection_pool.append(conn)
            
    async def insert_vector(
        self,
        embedding: np.ndarray,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Insert a vector with metadata.
        
        Args:
            embedding: Vector embedding (4096D)
            content: Text content associated with the vector
            metadata: Additional metadata
            
        Returns:
            ID of inserted vector
        """
        if embedding.shape[0] != self.dimension:
            raise ValueError(f"Expected {self.dimension}D vector, got {embedding.shape[0]}D")
            
        # Normalize embedding for cosine similarity
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        # Serialize embedding for sqlite-vec
        embedding_blob = serialize_float32(embedding.astype(np.float32))
        
        async with self._get_connection() as conn:
            # Insert vector
            cursor = await conn.execute(
                "INSERT INTO memory_vectors(embedding) VALUES (?)",
                (embedding_blob,)
            )
            vector_id = cursor.lastrowid
            
            # Insert metadata
            await conn.execute(
                """
                INSERT INTO vector_metadata(vector_id, content, metadata)
                VALUES (?, ?, ?)
                """,
                (
                    vector_id,
                    content,
                    json.dumps(metadata) if metadata else None,
                )
            )
            
            await conn.commit()
            
        logger.debug(f"Inserted vector {vector_id}")
        return vector_id
        
    async def batch_insert(
        self,
        embeddings: np.ndarray,
        contents: List[str],
        metadata_list: Optional[List[Dict[str, Any]]] = None,
    ) -> List[int]:
        """
        Insert multiple vectors in a batch.
        
        Args:
            embeddings: Array of shape (n, 4096)
            contents: List of text contents
            metadata_list: Optional list of metadata dicts
            
        Returns:
            List of inserted vector IDs
        """
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Expected {self.dimension}D vectors")
            
        if len(contents) != embeddings.shape[0]:
            raise ValueError("Number of contents must match number of embeddings")
            
        metadata_list = metadata_list or [None] * len(contents)
        vector_ids = []
        
        # Normalize all embeddings for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-10)
        
        async with self._get_connection() as conn:
            for i, (embedding, content, metadata) in enumerate(
                zip(embeddings, contents, metadata_list)
            ):
                embedding_blob = serialize_float32(embedding.astype(np.float32))
                
                cursor = await conn.execute(
                    "INSERT INTO memory_vectors(embedding) VALUES (?)",
                    (embedding_blob,)
                )
                vector_id = cursor.lastrowid
                vector_ids.append(vector_id)
                
                await conn.execute(
                    """
                    INSERT INTO vector_metadata(vector_id, content, metadata)
                    VALUES (?, ?, ?)
                    """,
                    (
                        vector_id,
                        content,
                        json.dumps(metadata) if metadata else None,
                    )
                )
                
            await conn.commit()
            
        logger.info(f"Batch inserted {len(vector_ids)} vectors")
        return vector_ids
        
    async def search_similar(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        include_embeddings: bool = False,
    ) -> List[VectorSearchResult]:
        """
        Search for k most similar vectors using cosine similarity.
        
        Args:
            query_embedding: Query vector (4096D)
            k: Number of results to return
            include_embeddings: Whether to include embeddings in results
            
        Returns:
            List of search results sorted by similarity (highest first)
        """
        if query_embedding.shape[0] != self.dimension:
            raise ValueError(f"Expected {self.dimension}D query vector")
            
        # Normalize query for cosine similarity
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm
            
        # Serialize query embedding
        query_blob = serialize_float32(query_embedding.astype(np.float32))
        
        start_time = time.time()
        
        async with self._get_connection() as conn:
            results = []
            
            if self._vec0_available:
                # Use vec0 extension for efficient KNN search
                cursor = await conn.execute(
                    f"""
                    SELECT 
                        v.vector_id,
                        v.distance,
                        m.content,
                        m.metadata,
                        m.effectiveness_score,
                        {'v.embedding' if include_embeddings else 'NULL'} as embedding
                    FROM memory_vectors v
                    LEFT JOIN vector_metadata m ON v.vector_id = m.vector_id
                    WHERE v.embedding MATCH ?
                        AND k = ?
                    ORDER BY v.distance ASC
                    """,
                    (query_blob, k)
                )
                
                async for row in cursor:
                    # sqlite-vec returns squared Euclidean distance for normalized vectors
                    # For normalized vectors: ||a - b||^2 = 2(1 - cos(a,b))
                    # Therefore: cos(a,b) = 1 - distance/2
                    distance = row[1]
                    similarity = 1.0 - (distance / 2.0)
                    
                    # Filter by similarity threshold
                    if similarity >= self.similarity_threshold:
                        result = VectorSearchResult(
                            id=row[0],
                            distance=distance,
                            similarity=similarity,
                            metadata=json.loads(row[3]) if row[3] else None,
                            embedding=np.frombuffer(row[5], dtype=np.float32) if row[5] else None,
                            content=row[2] if row[2] else None,
                        )
                        results.append(result)
            else:
                # Fallback: compute distances manually without vec0
                cursor = await conn.execute(
                    f"""
                    SELECT 
                        v.vector_id,
                        v.embedding,
                        m.content,
                        m.metadata,
                        m.effectiveness_score
                    FROM memory_vectors v
                    LEFT JOIN vector_metadata m ON v.vector_id = m.vector_id
                    """
                )
                
                all_results = []
                async for row in cursor:
                    stored_embedding = np.frombuffer(row[1], dtype=np.float32)
                    # Normalize stored embedding
                    norm = np.linalg.norm(stored_embedding)
                    if norm > 0:
                        stored_embedding = stored_embedding / norm
                    
                    # Compute cosine similarity
                    similarity = float(np.dot(query_embedding, stored_embedding))
                    distance = 2.0 * (1.0 - similarity)  # Convert to distance for consistency
                    
                    if similarity >= self.similarity_threshold:
                        all_results.append((
                            row[0],  # id
                            distance,
                            similarity,
                            row[2],  # content
                            row[3],  # metadata
                            stored_embedding if include_embeddings else None
                        ))
                
                # Sort by similarity and take top k
                all_results.sort(key=lambda x: x[2], reverse=True)
                for item in all_results[:k]:
                    result = VectorSearchResult(
                        id=item[0],
                        distance=item[1],
                        similarity=item[2],
                        content=item[3] if item[3] else None,
                        metadata=json.loads(item[4]) if item[4] else None,
                        embedding=item[5],
                    )
                    results.append(result)
                    
            # Update access timestamps
            if results:
                vector_ids = [r.id for r in results]
                await conn.executemany(
                    "UPDATE vector_metadata SET last_accessed = CURRENT_TIMESTAMP WHERE vector_id = ?",
                    [(vid,) for vid in vector_ids]
                )
                await conn.commit()
                
        search_time_ms = (time.time() - start_time) * 1000
        logger.debug(f"Search completed in {search_time_ms:.2f}ms, found {len(results)} results")
        
        # Verify <100ms latency requirement
        if search_time_ms > 100:
            logger.warning(f"Search latency {search_time_ms:.2f}ms exceeds 100ms target")
            
        return results
        
    async def update_effectiveness(
        self,
        vector_id: int,
        adjustment: float,
    ) -> None:
        """
        Update effectiveness score for a vector.
        
        Args:
            vector_id: ID of vector to update
            adjustment: Score adjustment (+0.3 for positive, -0.3 for negative)
        """
        async with self._get_connection() as conn:
            await conn.execute(
                """
                UPDATE vector_metadata 
                SET effectiveness_score = effectiveness_score + ?,
                    usage_count = usage_count + 1
                WHERE vector_id = ?
                """,
                (adjustment, vector_id)
            )
            await conn.commit()
            
        logger.debug(f"Updated effectiveness for vector {vector_id}: {adjustment:+.2f}")
        
    async def get_vector(
        self,
        vector_id: int,
        include_embedding: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a vector by ID.
        
        Args:
            vector_id: Vector ID
            include_embedding: Whether to include the embedding
            
        Returns:
            Vector data or None if not found
        """
        async with self._get_connection() as conn:
            cursor = await conn.execute(
                f"""
                SELECT 
                    v.vector_id,
                    {'v.embedding' if include_embedding else 'NULL'} as embedding,
                    m.content,
                    m.metadata,
                    m.effectiveness_score,
                    m.usage_count
                FROM memory_vectors v
                LEFT JOIN vector_metadata m ON v.vector_id = m.vector_id
                WHERE v.vector_id = ?
                """,
                (vector_id,)
            )
            
            row = await cursor.fetchone()
            if not row:
                return None
                
            return {
                "id": row[0],
                "embedding": np.frombuffer(row[1], dtype=np.float32) if row[1] else None,
                "content": row[2],
                "metadata": json.loads(row[3]) if row[3] else None,
                "effectiveness_score": row[4],
                "usage_count": row[5],
            }
            
    async def delete_vector(self, vector_id: int) -> bool:
        """
        Delete a vector by ID.
        
        Args:
            vector_id: Vector ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        async with self._get_connection() as conn:
            # Delete from metadata first (foreign key constraint)
            await conn.execute(
                "DELETE FROM vector_metadata WHERE vector_id = ?",
                (vector_id,)
            )
            
            cursor = await conn.execute(
                "DELETE FROM memory_vectors WHERE vector_id = ?",
                (vector_id,)
            )
            
            await conn.commit()
            deleted = cursor.rowcount > 0
            
        if deleted:
            logger.debug(f"Deleted vector {vector_id}")
        return deleted
        
    async def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        async with self._get_connection() as conn:
            cursor = await conn.execute("SELECT COUNT(*) FROM memory_vectors")
            total_vectors = (await cursor.fetchone())[0]
            
            cursor = await conn.execute(
                """
                SELECT 
                    AVG(effectiveness_score) as avg_effectiveness,
                    AVG(usage_count) as avg_usage,
                    MAX(usage_count) as max_usage
                FROM vector_metadata
                """
            )
            stats = await cursor.fetchone()
            
        return {
            "total_vectors": total_vectors,
            "dimension": self.dimension,
            "avg_effectiveness": stats[0] or 0.0,
            "avg_usage": stats[1] or 0.0,
            "max_usage": stats[2] or 0,
            "similarity_threshold": self.similarity_threshold,
        }
        
    async def close(self) -> None:
        """Close all database connections."""
        for conn in self._connection_pool:
            await conn.close()
        self._connection_pool.clear()
        self._initialized = False
        logger.info("Vector store closed")