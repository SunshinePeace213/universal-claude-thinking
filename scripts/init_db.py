#!/usr/bin/env python3
"""Initialize the atomic analyses database."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.storage.db import DatabaseConnection


async def main() -> None:
    """Initialize database with schema."""
    print("🔧 Initializing atomic analyses database...")
    
    # Create database connection
    db = DatabaseConnection()
    
    try:
        # Initialize schema
        await db.initialize_schema()
        print("✅ Database schema created successfully")
        
        # Verify tables exist
        async with db.connect() as conn:
            cursor = await conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='atomic_analyses'"
            )
            table = await cursor.fetchone()
            
            if table:
                print("✅ Verified: atomic_analyses table exists")
                
                # Check indexes
                cursor = await conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='atomic_analyses'"
                )
                indexes = await cursor.fetchall()
                
                print(f"✅ Created {len(indexes)} indexes for performance")
                for idx in indexes:
                    print(f"   - {idx[0]}")
            else:
                print("❌ Error: atomic_analyses table not found")
                sys.exit(1)
                
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        sys.exit(1)
    finally:
        await db.close()
        
    print(f"\n📍 Database location: {db.db_path}")
    print("🎉 Database initialization complete!")


if __name__ == "__main__":
    asyncio.run(main())