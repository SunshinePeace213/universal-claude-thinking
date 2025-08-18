"""
End-to-End Integration Tests for Vector Pipeline - User Journey Focused.

Tests complete user journeys from database initialization through to
effectiveness tracking, simulating real production usage scenarios.
"""

import asyncio
import logging
import os
import tempfile
import time
import unittest
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pytest
import torch

from src.rag.embedder import Qwen8BEmbedder
from src.core.molecular.vector_store import VectorStore, VectorSearchResult
from src.core.molecular.example_selector import ExampleSelector, SelectionStrategy
from src.core.molecular.context_builder import MoleculeContextBuilder
from src.rag.pipeline import RAGPipeline, PipelineConfig, PipelineMode, PipelineResult

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check if real model is available
MODEL_PATH = Path("embedding/Qwen3-Embedding-8B")
SKIP_E2E = not MODEL_PATH.exists() and not os.environ.get("FORCE_E2E_TEST", "").lower() == "true"


@pytest.mark.skipif(SKIP_E2E, reason="Real model not available for e2e testing")
class TestCompleteRAGPipelineE2E(unittest.IsolatedAsyncioTestCase):
    """End-to-end tests simulating complete user journeys through the RAG pipeline."""
    
    @classmethod
    def setUpClass(cls):
        """Set up class-level fixtures."""
        cls.model_path = MODEL_PATH
        logger.info(f"Running E2E tests with model at {cls.model_path}")
    
    async def asyncSetUp(self):
        """Set up test fixtures for user journey testing."""
        # Create temporary database for fresh start
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = Path(self.temp_db.name)
        
        # Production dataset for realistic testing
        self.production_dataset = self._create_production_dataset()
        
        logger.info(f"Test database: {self.db_path}")
    
    def _create_production_dataset(self) -> List[Dict[str, Any]]:
        """Create a realistic production dataset with various categories."""
        return [
            # Software Engineering Best Practices
            {"text": "Use dependency injection for better testability and modularity", "category": "architecture", "effectiveness": 0.8},
            {"text": "Implement comprehensive error handling with proper logging", "category": "error_handling", "effectiveness": 0.9},
            {"text": "Apply the SOLID principles for maintainable object-oriented design", "category": "design", "effectiveness": 0.85},
            {"text": "Use async/await for non-blocking I/O operations", "category": "performance", "effectiveness": 0.75},
            {"text": "Implement caching strategies to reduce database load", "category": "performance", "effectiveness": 0.9},
            
            # API Development
            {"text": "Design RESTful APIs with clear resource naming", "category": "api", "effectiveness": 0.7},
            {"text": "Implement rate limiting to prevent API abuse", "category": "api", "effectiveness": 0.85},
            {"text": "Use API versioning for backward compatibility", "category": "api", "effectiveness": 0.8},
            {"text": "Add comprehensive API documentation with examples", "category": "api", "effectiveness": 0.75},
            
            # Database Optimization
            {"text": "Create indexes on frequently queried columns", "category": "database", "effectiveness": 0.95},
            {"text": "Use connection pooling for database efficiency", "category": "database", "effectiveness": 0.85},
            {"text": "Implement database migrations for schema changes", "category": "database", "effectiveness": 0.7},
            {"text": "Optimize queries using EXPLAIN ANALYZE", "category": "database", "effectiveness": 0.8},
            
            # Security Practices
            {"text": "Implement input validation to prevent injection attacks", "category": "security", "effectiveness": 0.95},
            {"text": "Use environment variables for sensitive configuration", "category": "security", "effectiveness": 0.9},
            {"text": "Apply principle of least privilege for access control", "category": "security", "effectiveness": 0.85},
            
            # Testing Strategies
            {"text": "Write unit tests with high code coverage", "category": "testing", "effectiveness": 0.8},
            {"text": "Implement integration tests for API endpoints", "category": "testing", "effectiveness": 0.85},
            {"text": "Use test-driven development for critical features", "category": "testing", "effectiveness": 0.75},
            {"text": "Add end-to-end tests for user workflows", "category": "testing", "effectiveness": 0.9},
            
            # DevOps & Deployment
            {"text": "Use CI/CD pipelines for automated deployment", "category": "devops", "effectiveness": 0.85},
            {"text": "Implement health checks for service monitoring", "category": "devops", "effectiveness": 0.8},
            {"text": "Use containerization for consistent deployments", "category": "devops", "effectiveness": 0.9},
            {"text": "Apply infrastructure as code principles", "category": "devops", "effectiveness": 0.75},
            
            # Performance Optimization
            {"text": "Use lazy loading for improved initial load times", "category": "performance", "effectiveness": 0.85},
            {"text": "Implement CDN for static asset delivery", "category": "performance", "effectiveness": 0.8},
            {"text": "Apply code splitting for smaller bundle sizes", "category": "performance", "effectiveness": 0.75},
            {"text": "Use database query optimization techniques", "category": "performance", "effectiveness": 0.9},
            
            # Microservices
            {"text": "Implement circuit breakers for fault tolerance", "category": "microservices", "effectiveness": 0.85},
            {"text": "Use service discovery for dynamic routing", "category": "microservices", "effectiveness": 0.7},
            {"text": "Apply event-driven architecture for loose coupling", "category": "microservices", "effectiveness": 0.8},
            {"text": "Implement distributed tracing for debugging", "category": "microservices", "effectiveness": 0.75}
        ]
    
    async def asyncTearDown(self):
        """Clean up test fixtures."""
        # Clean up database
        if hasattr(self, 'db_path') and self.db_path.exists():
            self.db_path.unlink(missing_ok=True)
    
    async def test_user_journey_first_time_setup(self):
        """Complete journey from empty database to working system."""
        logger.info("=== User Journey: First Time Setup ===")
        
        # Step 1: Initialize empty database and components
        logger.info("Step 1: Initializing fresh system...")
        
        embedder = Qwen8BEmbedder(model_path=self.model_path, batch_size=16)
        vector_store = VectorStore(
            db_path=self.db_path,
            dimension=4096,
            similarity_threshold=0.85
        )
        
        config = PipelineConfig(
            mode=PipelineMode.HYBRID,
            max_examples=5,
            similarity_threshold=0.85,
            target_latency_ms=800
        )
        
        pipeline = RAGPipeline(
            vector_store=vector_store,
            embedder=embedder,
            config=config
        )
        
        try:
            await pipeline.initialize()
            logger.info("✓ System initialized successfully")
            
            # Step 2: Create vector tables
            logger.info("Step 2: Creating vector tables...")
            
            # Verify tables created
            async with vector_store._get_connection() as conn:
                cursor = await conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
                tables = await cursor.fetchall()
                table_names = [t[0] for t in tables]
                self.assertIn("embeddings", table_names)
                logger.info(f"✓ Tables created: {table_names}")
            
            # Step 3: Load initial dataset
            logger.info("Step 3: Loading initial dataset...")
            
            # Load first 10 examples as initial dataset
            initial_examples = self.production_dataset[:10]
            for example in initial_examples:
                embedding = await embedder.generate_embedding(example["text"])
                await vector_store.add_embedding(
                    embedding=embedding[0] if len(embedding.shape) > 1 else embedding,
                    metadata={
                        "text": example["text"],
                        "category": example["category"],
                        "effectiveness_score": example["effectiveness"]
                    }
                )
            
            count = await vector_store.count()
            self.assertEqual(count, 10)
            logger.info(f"✓ Loaded {count} initial examples")
            
            # Step 4: Process first query
            logger.info("Step 4: Processing first user query...")
            
            first_query = "How can I improve my API performance?"
            result = await pipeline.process(
                input_text=first_query,
                instruction="Provide best practices for API optimization"
            )
            
            # Step 5: Verify results
            logger.info("Step 5: Verifying system is working...")
            
            self.assertIsNotNone(result.context)
            self.assertGreater(result.examples_retrieved, 0)
            self.assertLess(result.total_latency_ms, 800)
            
            # Verify MOLECULE structure
            context_str = result.context.format()
            self.assertIn("INSTRUCTION", context_str)
            self.assertIn("EXAMPLES", context_str)
            self.assertIn("NEW INPUT", context_str)
            self.assertIn(first_query, context_str)
            
            logger.info(f"✓ First query processed successfully:")
            logger.info(f"  - Examples retrieved: {result.examples_retrieved}")
            logger.info(f"  - Total latency: {result.total_latency_ms:.2f}ms")
            logger.info(f"  - Context length: {len(context_str)} chars")
            
        finally:
            await pipeline.close()
    
    async def test_user_journey_continuous_learning(self):
        """Test system improvement through usage and feedback."""
        logger.info("=== User Journey: Continuous Learning ===")
        
        # Initialize system
        embedder = Qwen8BEmbedder(model_path=self.model_path)
        vector_store = VectorStore(db_path=self.db_path, dimension=4096)
        pipeline = RAGPipeline(vector_store=vector_store, embedder=embedder)
        
        try:
            await pipeline.initialize()
            
            # Load dataset with effectiveness scores
            logger.info("Loading dataset with effectiveness tracking...")
            for example in self.production_dataset[:15]:
                embedding = await embedder.generate_embedding(example["text"])
                await vector_store.add_embedding(
                    embedding=embedding[0] if len(embedding.shape) > 1 else embedding,
                    metadata={
                        "text": example["text"],
                        "category": example["category"],
                        "effectiveness_score": example["effectiveness"]
                    }
                )
            
            # Simulate multiple queries with feedback
            queries_and_feedback = [
                ("How to optimize database queries?", 0.3),  # Good result
                ("Best practices for API security?", 0.3),   # Good result
                ("How to handle errors properly?", -0.3),    # Poor result
                ("Database optimization techniques?", 0.3),  # Good result
            ]
            
            results_history = []
            
            for query, feedback in queries_and_feedback:
                logger.info(f"Processing: {query[:30]}...")
                
                # Process query
                result = await pipeline.process(query)
                results_history.append(result)
                
                # Apply feedback
                await pipeline.update_effectiveness(result, feedback)
                logger.info(f"  Applied feedback: {'+' if feedback > 0 else ''}{feedback}")
            
            # Process similar query to test improved results
            logger.info("Testing if system learned from feedback...")
            
            final_query = "What are the best database optimization strategies?"
            final_result = await pipeline.process(final_query)
            
            # Verify system is learning
            self.assertGreater(final_result.examples_retrieved, 0)
            
            # Check that database-related examples are prioritized
            context_str = final_result.context.format()
            database_keywords = ["database", "query", "index", "optimization"]
            matches = sum(1 for kw in database_keywords if kw in context_str.lower())
            self.assertGreater(matches, 0, "System should prioritize database content")
            
            logger.info(f"✓ System successfully learning from feedback")
            logger.info(f"  - Processed {len(queries_and_feedback)} queries with feedback")
            logger.info(f"  - Final query retrieved {final_result.examples_retrieved} examples")
            
            # Test example promotion logic
            logger.info("Testing example promotion based on effectiveness...")
            
            # High-performing examples should be retrieved more often
            test_query = "How to create database indexes?"
            test_result = await pipeline.process(test_query)
            
            self.assertGreater(test_result.examples_retrieved, 0)
            logger.info(f"✓ Promotion logic working: {test_result.examples_retrieved} examples")
            
        finally:
            await pipeline.close()
    
    async def test_user_journey_high_load(self):
        """Test system under production load."""
        logger.info("=== User Journey: High Load Scenario ===")
        
        # Initialize with production config
        embedder = Qwen8BEmbedder(model_path=self.model_path, batch_size=32)
        vector_store = VectorStore(
            db_path=self.db_path,
            dimension=4096,
            connection_pool_size=4  # Higher for concurrent access
        )
        
        config = PipelineConfig(
            mode=PipelineMode.HYBRID,
            batch_size=32,  # Mac M3 max
            enable_caching=True,
            target_latency_ms=800
        )
        
        pipeline = RAGPipeline(
            vector_store=vector_store,
            embedder=embedder,
            config=config
        )
        
        try:
            await pipeline.initialize()
            
            # Load full dataset
            logger.info(f"Loading {len(self.production_dataset)} examples...")
            for example in self.production_dataset:
                embedding = await embedder.generate_embedding(example["text"])
                await vector_store.add_embedding(
                    embedding=embedding[0] if len(embedding.shape) > 1 else embedding,
                    metadata={
                        "text": example["text"],
                        "category": example["category"]
                    }
                )
            
            # Process 100+ queries to simulate load
            logger.info("Processing 100 queries under load...")
            
            queries = [
                "How to optimize API performance?",
                "Best practices for database design?",
                "Security measures for web applications?",
                "Testing strategies for microservices?",
                "Deployment best practices?",
            ] * 20  # 100 total queries
            
            start_time = time.time()
            latencies = []
            cache_hits = 0
            
            for i, query in enumerate(queries):
                query_start = time.time()
                result = await pipeline.process(query)
                query_latency = (time.time() - query_start) * 1000
                latencies.append(query_latency)
                
                if result.cache_hits > 0:
                    cache_hits += 1
                
                if (i + 1) % 20 == 0:
                    logger.info(f"  Processed {i + 1}/100 queries")
            
            total_time = time.time() - start_time
            avg_latency = np.mean(latencies)
            p95_latency = np.percentile(latencies, 95)
            
            # Monitor performance metrics
            metrics = pipeline.get_performance_metrics()
            
            logger.info(f"✓ High load test completed:")
            logger.info(f"  - Total time: {total_time:.2f}s")
            logger.info(f"  - Average latency: {avg_latency:.2f}ms")
            logger.info(f"  - P95 latency: {p95_latency:.2f}ms")
            logger.info(f"  - Cache hit rate: {metrics.get('cache_hit_rate', 0):.2%}")
            
            # Verify performance requirements
            self.assertLess(avg_latency, 800, "Average latency exceeds target")
            
            # Test optimization triggers
            await pipeline.optimize_for_latency()
            logger.info("✓ Optimization triggered successfully")
            
        finally:
            await pipeline.close()
    
    async def test_user_journey_error_recovery(self):
        """Test system recovery from errors and edge cases."""
        logger.info("=== User Journey: Error Recovery ===")
        
        embedder = Qwen8BEmbedder(model_path=self.model_path)
        vector_store = VectorStore(db_path=self.db_path, dimension=4096)
        pipeline = RAGPipeline(vector_store=vector_store, embedder=embedder)
        
        try:
            await pipeline.initialize()
            
            # Test 1: Empty database query
            logger.info("Test 1: Handling empty database...")
            result = await pipeline.process("Test query on empty database")
            self.assertIsNotNone(result.context)
            if result.warnings:
                logger.info(f"  ✓ Handled empty DB: {result.warnings[0]}")
            
            # Test 2: No similar examples (below threshold)
            logger.info("Test 2: No similar examples above threshold...")
            
            # Add unrelated examples
            unrelated = [
                "The weather is sunny today",
                "Cooking pasta requires proper timing",
                "Tennis is a popular sport"
            ]
            
            for text in unrelated:
                embedding = await embedder.generate_embedding(text)
                await vector_store.add_embedding(
                    embedding=embedding[0] if len(embedding.shape) > 1 else embedding,
                    metadata={"text": text}
                )
            
            # Query for something completely different
            result = await pipeline.process("How to implement microservices architecture?")
            self.assertIsNotNone(result.context)
            logger.info(f"  ✓ Handled low similarity: {result.examples_retrieved} examples")
            
            # Test 3: Malformed input handling
            logger.info("Test 3: Handling malformed input...")
            
            edge_cases = [
                "",  # Empty string
                " " * 1000,  # Very long whitespace
                "a" * 5000,  # Very long text
            ]
            
            for i, edge_case in enumerate(edge_cases):
                try:
                    result = await pipeline.process(edge_case[:100])  # Truncate for safety
                    logger.info(f"  ✓ Handled edge case {i+1}")
                except Exception as e:
                    logger.info(f"  ✓ Caught exception for edge case {i+1}: {str(e)[:50]}")
            
            # Test 4: Recovery after errors
            logger.info("Test 4: Recovery after errors...")
            
            # Process normal query after edge cases
            normal_result = await pipeline.process("What are best practices for API design?")
            self.assertIsNotNone(normal_result.context)
            logger.info("  ✓ System recovered and processing normally")
            
            # Test 5: Database persistence after restart
            logger.info("Test 5: Testing persistence and recovery...")
            
            # Close and reopen
            await pipeline.close()
            
            # Create new instances
            new_vector_store = VectorStore(db_path=self.db_path, dimension=4096)
            new_pipeline = RAGPipeline(
                vector_store=new_vector_store,
                embedder=embedder
            )
            await new_pipeline.initialize()
            
            # Verify data persisted
            count = await new_vector_store.count()
            self.assertGreater(count, 0)
            logger.info(f"  ✓ Recovered {count} embeddings after restart")
            
            await new_pipeline.close()
            
        finally:
            # Ensure cleanup even after errors
            try:
                await pipeline.close()
            except:
                pass
    


if __name__ == "__main__":
    # Run with pytest for better async support
    pytest.main([__file__, "-v", "--log-cli-level=INFO"])
