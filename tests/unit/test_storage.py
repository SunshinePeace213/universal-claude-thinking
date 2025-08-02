"""Unit tests for storage module."""

import asyncio
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.core.storage.atomic_repository import AtomicAnalysisRepository
from src.core.storage.db import DatabaseConnection, JSONEncoder, generate_uuid


@pytest.fixture
async def temp_db():
    """Create temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name
    
    db = DatabaseConnection(db_path)
    await db.initialize_schema()
    
    yield db
    
    await db.close()
    Path(db_path).unlink()  # Clean up


@pytest.fixture
async def repository(temp_db):
    """Create repository with temporary database."""
    return AtomicAnalysisRepository(temp_db)


class TestDatabaseConnection:
    """Tests for DatabaseConnection class."""
    
    async def test_initialize_schema(self, temp_db):
        """Test schema initialization creates tables and indexes."""
        async with temp_db.connect() as conn:
            # Check table exists
            cursor = await conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='atomic_analyses'"
            )
            table = await cursor.fetchone()
            assert table is not None
            assert table[0] == "atomic_analyses"
            
            # Check indexes exist
            cursor = await conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='atomic_analyses'"
            )
            indexes = await cursor.fetchall()
            index_names = [idx[0] for idx in indexes]
            
            assert "idx_prompt_hash" in index_names
            assert "idx_created_at" in index_names
            assert "idx_quality_score" in index_names
    
    async def test_connection_reuse(self, temp_db):
        """Test connection is reused properly."""
        async with temp_db.connect() as conn1:
            async with temp_db.connect() as conn2:
                # Should be the same connection
                assert conn1 is conn2


class TestJSONEncoder:
    """Tests for JSONEncoder helper class."""
    
    def test_encode_basic_types(self):
        """Test encoding basic Python types."""
        encoder = JSONEncoder()
        
        assert encoder.encode(None) == "null"
        assert encoder.encode({"key": "value"}) == '{"key": "value"}'
        assert encoder.encode([1, 2, 3]) == "[1, 2, 3]"
        assert encoder.encode("string") == '"string"'
    
    def test_encode_datetime(self):
        """Test encoding datetime objects."""
        encoder = JSONEncoder()
        dt = datetime(2024, 1, 1, 12, 0, 0)
        result = encoder.encode(dt)
        assert "2024-01-01T12:00:00" in result
    
    def test_decode(self):
        """Test decoding JSON strings."""
        encoder = JSONEncoder()
        
        assert encoder.decode("null") is None
        assert encoder.decode('{"key": "value"}') == {"key": "value"}
        assert encoder.decode("[1, 2, 3]") == [1, 2, 3]
        assert encoder.decode(None) is None


class TestAtomicAnalysisRepository:
    """Tests for AtomicAnalysisRepository class."""
    
    async def test_save_and_retrieve(self, repository):
        """Test saving and retrieving analysis."""
        # Save analysis
        analysis_id = await repository.save_analysis(
            prompt_hash="test_hash_123",
            structure={"task": "Test task", "constraints": None, "output_format": None},
            quality_score=7.5,
            gaps=["constraints", "output_format"],
            suggestions=["Add constraints", "Specify output format"]
        )
        
        assert analysis_id is not None
        
        # Retrieve by hash
        result = await repository.get_by_hash("test_hash_123")
        assert result is not None
        assert result["prompt_hash"] == "test_hash_123"
        assert result["structure"]["task"] == "Test task"
        assert result["quality_score"] == 7.5
        assert result["gaps"] == ["constraints", "output_format"]
        assert result["usage_count"] == 0
    
    async def test_duplicate_hash_increments_usage(self, repository):
        """Test that duplicate hash increments usage count."""
        # First save
        id1 = await repository.save_analysis(
            prompt_hash="duplicate_hash",
            structure={"task": "Task 1"},
            quality_score=5.0,
            gaps=[],
            suggestions=[]
        )
        
        # Second save with same hash
        id2 = await repository.save_analysis(
            prompt_hash="duplicate_hash",
            structure={"task": "Task 2"},  # Different data
            quality_score=6.0,
            gaps=["task"],
            suggestions=["Improve task"]
        )
        
        # Should return same ID
        assert id1 == id2
        
        # Check usage count incremented
        result = await repository.get_by_hash("duplicate_hash")
        assert result["usage_count"] == 1
        assert result["structure"]["task"] == "Task 1"  # Original data preserved
    
    async def test_increment_usage(self, repository):
        """Test incrementing usage count."""
        # Save analysis
        await repository.save_analysis(
            prompt_hash="usage_test",
            structure={"task": "Test"},
            quality_score=8.0,
            gaps=[],
            suggestions=[]
        )
        
        # Increment usage
        success = await repository.increment_usage("usage_test")
        assert success is True
        
        # Check count
        result = await repository.get_by_hash("usage_test")
        assert result["usage_count"] == 1
        
        # Increment again
        await repository.increment_usage("usage_test")
        result = await repository.get_by_hash("usage_test")
        assert result["usage_count"] == 2
        
        # Try non-existent hash
        success = await repository.increment_usage("non_existent")
        assert success is False
    
    async def test_get_recent(self, repository):
        """Test getting recent analyses."""
        # Save multiple analyses
        for i in range(5):
            await repository.save_analysis(
                prompt_hash=f"recent_{i}",
                structure={"task": f"Task {i}"},
                quality_score=5.0 + i,
                gaps=[],
                suggestions=[]
            )
            await asyncio.sleep(0.01)  # Ensure different timestamps
        
        # Get recent
        recent = await repository.get_recent(limit=3)
        assert len(recent) == 3
        assert recent[0]["prompt_hash"] == "recent_4"  # Most recent first
        assert recent[1]["prompt_hash"] == "recent_3"
        assert recent[2]["prompt_hash"] == "recent_2"
        
        # Get with quality filter
        high_quality = await repository.get_recent(limit=10, min_score=8.0)
        assert len(high_quality) == 2  # Only recent_3 and recent_4
        assert all(r["quality_score"] >= 8.0 for r in high_quality)
    
    async def test_cleanup_old(self, repository):
        """Test cleanup of old analyses."""
        # Save analyses with different ages
        now = datetime.utcnow()
        
        # Recent analysis
        await repository.save_analysis(
            prompt_hash="recent",
            structure={"task": "Recent"},
            quality_score=7.0,
            gaps=[],
            suggestions=[]
        )
        
        # Old analysis with low usage (should be deleted)
        async with repository.db.connect() as conn:
            old_date = now - timedelta(days=40)
            await conn.execute(
                """
                INSERT INTO atomic_analyses 
                (id, prompt_hash, structure, quality_score, gaps, suggestions, created_at, usage_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    generate_uuid(),
                    "old_low_usage",
                    '{"task": "Old"}',
                    5.0,
                    '[]',
                    '[]',
                    old_date,
                    2  # Low usage
                )
            )
            
            # Old analysis with high usage (should be kept)
            await conn.execute(
                """
                INSERT INTO atomic_analyses 
                (id, prompt_hash, structure, quality_score, gaps, suggestions, created_at, usage_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    generate_uuid(),
                    "old_high_usage",
                    '{"task": "Old but used"}',
                    8.0,
                    '[]',
                    '[]',
                    old_date,
                    10  # High usage
                )
            )
            await conn.commit()
        
        # Run cleanup
        deleted = await repository.cleanup_old(days=30)
        assert deleted == 1  # Only old_low_usage deleted
        
        # Verify results
        assert await repository.get_by_hash("recent") is not None
        assert await repository.get_by_hash("old_low_usage") is None
        assert await repository.get_by_hash("old_high_usage") is not None
    
    async def test_get_statistics(self, repository):
        """Test getting repository statistics."""
        # Empty repository
        stats = await repository.get_statistics()
        assert stats["total_analyses"] == 0
        assert stats["average_score"] == 0
        assert stats["high_quality_count"] == 0
        assert stats["high_quality_percentage"] == 0
        
        # Add analyses
        await repository.save_analysis("hash1", {}, 5.0, [], [])
        await repository.save_analysis("hash2", {}, 8.0, [], [])
        await repository.save_analysis("hash3", {}, 9.0, [], [])
        
        # Increment usage for hash3
        await repository.increment_usage("hash3")
        await repository.increment_usage("hash3")
        
        # Get stats
        stats = await repository.get_statistics()
        assert stats["total_analyses"] == 3
        assert stats["average_score"] == 7.33  # (5+8+9)/3
        assert stats["high_quality_count"] == 2  # 8.0 and 9.0
        assert stats["high_quality_percentage"] == 66.7
        assert stats["most_used_hash"] == "hash3"
        assert stats["most_used_count"] == 2


def test_generate_uuid():
    """Test UUID generation."""
    uuid1 = generate_uuid()
    uuid2 = generate_uuid()
    
    assert uuid1 != uuid2
    assert len(uuid1) == 36  # Standard UUID format
    assert uuid1.count("-") == 4