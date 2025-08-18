"""
Initialize Vector Store for Story 1.4 - Molecular Context Assembly.

This script initializes the vector store database and RAG pipeline components
for the molecular context assembly system. It creates the necessary tables
and verifies the system is ready for production use.

Usage:
    python -m src.init_vector_store [--db-path PATH] [--dimension DIM]
    
Example:
    python -m src.init_vector_store --db-path data/production.db
    python -m src.init_vector_store --dimension 4096 --model-path embedding/Qwen3-Embedding-8B
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.molecular.vector_store import VectorStore
from src.rag.pipeline import RAGPipeline, PipelineConfig, PipelineMode
from src.rag.embedder import Qwen8BEmbedder

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def initialize_vector_store(
    db_path: Path = Path("data/vector_store.db"),
    dimension: int = 4096,
    model_path: Optional[Path] = None,
    skip_model: bool = False
) -> bool:
    """
    Initialize the vector store and optionally the RAG pipeline.
    
    Args:
        db_path: Path to the SQLite database file
        dimension: Dimension of embeddings (default: 4096 for Qwen3)
        model_path: Path to the Qwen3-Embedding-8B model
        skip_model: Skip model initialization (useful for testing)
        
    Returns:
        True if initialization successful
    """
    
    logger.info("=== Vector Store Initialization ===")
    logger.info(f"Database path: {db_path}")
    logger.info(f"Embedding dimension: {dimension}")
    
    # Ensure directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize vector store
    vector_store = VectorStore(
        db_path=db_path,
        dimension=dimension,
        similarity_threshold=0.85,
        connection_pool_size=4
    )
    
    try:
        # Initialize vector store
        await vector_store.initialize()
        logger.info("✓ Vector store initialized")
        
        # Verify tables created
        async with vector_store._get_connection() as conn:
            cursor = await conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = await cursor.fetchall()
            table_names = [t[0] for t in tables]
            
            logger.info(f"✓ Database tables created: {table_names}")
            
            # Verify expected tables exist
            expected_tables = ["memory_vectors", "vector_metadata"]
            for table in expected_tables:
                if table not in table_names:
                    logger.warning(f"  Warning: Expected table '{table}' not found")
        
        # Optionally initialize full pipeline with model
        if not skip_model:
            if model_path is None:
                model_path = Path("embedding/Qwen3-Embedding-8B")
            
            if not model_path.exists():
                logger.warning(f"Model not found at {model_path}")
                logger.info("Skipping model initialization")
            else:
                logger.info(f"Initializing model from {model_path}")
                
                embedder = Qwen8BEmbedder(
                    model_path=model_path,
                    batch_size=32
                )
                
                config = PipelineConfig(
                    mode=PipelineMode.HYBRID,
                    max_examples=10,
                    similarity_threshold=0.85,
                    chunk_size=1024,
                    overlap_ratio=0.15,
                    target_latency_ms=800
                )
                
                pipeline = RAGPipeline(
                    vector_store=vector_store,
                    embedder=embedder,
                    config=config
                )
                
                # Initialize pipeline (includes embedder initialization)
                await pipeline.initialize()
                logger.info("✓ RAG pipeline initialized")
                
                # Test with a sample embedding
                logger.info("Testing pipeline...")
                test_text = "This is a test initialization."
                result = await pipeline.process(test_text)
                
                if result:
                    logger.info("✓ Pipeline test successful")
                    logger.info(f"  - Context created: {len(result.context.format())} chars")
                    logger.info(f"  - Latency: {result.total_latency_ms:.2f}ms")
                
                await pipeline.close()
        
        # Add initial statistics
        logger.info("\n=== Initialization Complete ===")
        logger.info(f"Database location: {db_path.absolute()}")
        logger.info(f"Vector dimensions: {dimension}")
        logger.info(f"Similarity threshold: 0.85")
        logger.info(f"Connection pool size: 4")
        
        if not skip_model and model_path and model_path.exists():
            logger.info(f"Model loaded: {model_path.name}")
            logger.info(f"Batch size: 32 (Mac M3 optimized)")
        
        return True
        
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        return False
        
    finally:
        await vector_store.close()
        logger.info("✓ Resources cleaned up")


def main():
    """Main entry point for command-line usage."""
    
    parser = argparse.ArgumentParser(
        description="Initialize Vector Store for Story 1.4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Initialize with defaults
  python -m src.init_vector_store
  
  # Custom database path
  python -m src.init_vector_store --db-path data/production.db
  
  # Initialize without model (database only)
  python -m src.init_vector_store --skip-model
  
  # Custom model path
  python -m src.init_vector_store --model-path /path/to/model
        """
    )
    
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("data/vector_store.db"),
        help="Path to the SQLite database file (default: data/vector_store.db)"
    )
    
    parser.add_argument(
        "--dimension",
        type=int,
        default=4096,
        help="Dimension of embeddings (default: 4096 for Qwen3)"
    )
    
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to Qwen3-Embedding-8B model (default: embedding/Qwen3-Embedding-8B)"
    )
    
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Skip model initialization, only create database"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run initialization
    success = asyncio.run(initialize_vector_store(
        db_path=args.db_path,
        dimension=args.dimension,
        model_path=args.model_path,
        skip_model=args.skip_model
    ))
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()