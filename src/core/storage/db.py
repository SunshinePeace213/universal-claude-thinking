"""Database connection and management for atomic analyses."""

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import aiosqlite


class DatabaseConnection:
    """Manages SQLite database connections with async support."""

    def __init__(self, db_path: str | None = None) -> None:
        """Initialize database connection.

        Args:
            db_path: Path to SQLite database file. Defaults to data/thinking_v2.db
        """
        if db_path is None:
            # Create data directory if it doesn't exist
            data_dir = Path("data")
            data_dir.mkdir(exist_ok=True)
            db_path = str(data_dir / "thinking_v2.db")

        self.db_path = db_path
        self._connection: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def connect(self) -> AsyncIterator[aiosqlite.Connection]:
        """Get database connection with context manager.

        Yields:
            Active database connection
        """
        async with self._lock:
            if self._connection is None:
                self._connection = await aiosqlite.connect(self.db_path)
                # Enable JSON1 extension for JSONB-like functionality
                await self._connection.execute("PRAGMA journal_mode=WAL")
                await self._connection.execute("PRAGMA foreign_keys=ON")

            yield self._connection

    async def initialize_schema(self) -> None:
        """Create database tables if they don't exist."""
        async with self.connect() as db:
            # Create atomic_analyses table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS atomic_analyses (
                    id TEXT PRIMARY KEY,
                    prompt_hash TEXT UNIQUE NOT NULL,
                    structure TEXT NOT NULL,
                    quality_score REAL CHECK (quality_score >= 1 AND quality_score <= 10),
                    gaps TEXT,
                    suggestions TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    usage_count INTEGER DEFAULT 0
                )
            """)

            # Create indexes for performance
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_prompt_hash 
                ON atomic_analyses(prompt_hash)
            """)

            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at 
                ON atomic_analyses(created_at DESC)
            """)

            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_quality_score 
                ON atomic_analyses(quality_score DESC)
            """)

            await db.commit()

    async def close(self) -> None:
        """Close database connection."""
        if self._connection:
            await self._connection.close()
            self._connection = None


class JSONEncoder:
    """Helper class for JSON serialization of database values."""

    @staticmethod
    def encode(obj: Any) -> str:
        """Encode Python object to JSON string.

        Args:
            obj: Object to encode

        Returns:
            JSON string representation
        """
        if obj is None:
            return json.dumps(None)

        # Handle datetime objects
        def json_serial(obj: Any) -> str:
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Type {type(obj)} not serializable")

        return json.dumps(obj, default=json_serial)

    @staticmethod
    def decode(json_str: str | None) -> Any:
        """Decode JSON string to Python object.

        Args:
            json_str: JSON string to decode

        Returns:
            Decoded Python object
        """
        if json_str is None:
            return None
        return json.loads(json_str)


def generate_uuid() -> str:
    """Generate a new UUID string.

    Returns:
        UUID as string
    """
    return str(uuid.uuid4())
