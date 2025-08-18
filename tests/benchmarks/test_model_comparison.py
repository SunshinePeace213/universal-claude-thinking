"""
Unit tests for Model Benchmarking Framework.

Tests the ModelBenchmark class that compares Qwen3 embedding models
on accuracy, latency, memory usage, and throughput metrics.
"""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from datetime import datetime
from typing import List, Dict, Any

import numpy as np
import pytest
import torch

from src.rag.benchmarks.model_benchmark import (
    ModelBenchmark,
    BenchmarkMetric,
    BenchmarkResult,
    BenchmarkReport,
)
from src.rag.embedder import ModelType


class TestBenchmarkResult(unittest.TestCase):
    """Test cases for BenchmarkResult dataclass."""
    
    def test_benchmark_result_creation(self):
        """Test creating a benchmark result."""
        result = BenchmarkResult(
            model_type=ModelType.QWEN3_8B,
            metric=BenchmarkMetric.LATENCY,
            value=150.5,
            unit="ms",
            metadata={"batch_size": 32},
        )
        
        self.assertEqual(result.model_type, ModelType.QWEN3_8B)
        self.assertEqual(result.metric, BenchmarkMetric.LATENCY)
        self.assertEqual(result.value, 150.5)
        self.assertEqual(result.unit, "ms")
        self.assertEqual(result.metadata["batch_size"], 32)
        self.assertIsInstance(result.timestamp, datetime)


class TestModelBenchmark(unittest.TestCase):
    """Test cases for ModelBenchmark class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.benchmark = ModelBenchmark(
            num_samples=10,
            batch_sizes=[1, 4, 8],
            warmup_iterations=2,
        )
        
    async def test_initialization(self):
        """Test benchmark initialization."""
        await self.benchmark.initialize()
        
        # Should generate test data
        self.assertEqual(len(self.benchmark.test_texts), 10)
        self.assertGreater(len(self.benchmark.test_texts[0]), 0)
        
    def test_generate_test_data(self):
        """Test synthetic test data generation."""
        self.benchmark._generate_test_data()
        
        # Should generate correct number of samples
        self.assertEqual(len(self.benchmark.test_texts), 10)
        
        # Each sample should be non-empty string
        for text in self.benchmark.test_texts:
            self.assertIsInstance(text, str)
            self.assertGreater(len(text), 0)
            
        # Should have variety in texts
        unique_texts = set(self.benchmark.test_texts)
        self.assertGreater(len(unique_texts), 5)
        
    def test_load_test_data(self):
        """Test loading test data from file."""
        # Create temporary test data file
        test_data = {
            "texts": ["Text 1", "Text 2", "Text 3", "Text 4", "Text 5"]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_path = Path(f.name)
            
        try:
            self.benchmark.test_data_path = temp_path
            self.benchmark.num_samples = 3
            self.benchmark._load_test_data()
            
            # Should load specified number of samples
            self.assertEqual(len(self.benchmark.test_texts), 3)
            self.assertEqual(self.benchmark.test_texts[0], "Text 1")
            self.assertEqual(self.benchmark.test_texts[2], "Text 3")
            
        finally:
            temp_path.unlink()
            
    async def test_warmup(self):
        """Test model warmup."""
        mock_model = AsyncMock()
        mock_model.generate_embedding = AsyncMock(return_value=np.zeros((10, 4096)))
        
        self.benchmark.test_texts = ["Text 1", "Text 2", "Text 3"]
        self.benchmark.warmup_iterations = 3
        
        await self.benchmark._warmup(mock_model)
        
        # Should call generate_embedding for warmup
        self.assertEqual(mock_model.generate_embedding.call_count, 3)
        
    async def test_benchmark_embedding_quality(self):
        """Test embedding quality benchmark (AC 9)."""
        mock_model = AsyncMock()
        
        # Create diverse embeddings
        embeddings = []
        for i in range(10):
            emb = np.random.randn(4096)
            emb = emb / np.linalg.norm(emb)  # Normalize
            embeddings.append(emb)
        embeddings = np.array(embeddings)
        
        # Mock batch generation
        mock_model.generate_embedding = AsyncMock(return_value=embeddings)
        
        self.benchmark.test_texts = ["Text " + str(i) for i in range(10)]
        
        results = await self.benchmark._benchmark_embedding_quality(
            mock_model,
            ModelType.QWEN3_8B
        )
        
        # Should return quality metrics
        self.assertGreater(len(results), 0)
        
        # Check self-similarity result
        self_sim_result = next(
            (r for r in results if r.metadata.get("test") == "self_similarity"),
            None
        )
        self.assertIsNotNone(self_sim_result)
        self.assertAlmostEqual(self_sim_result.value, 1.0, places=5)
        
        # Check distinctiveness result
        distinct_result = next(
            (r for r in results if r.metadata.get("test") == "pairwise_distinctiveness"),
            None
        )
        self.assertIsNotNone(distinct_result)
        self.assertGreaterEqual(distinct_result.value, 0.0)
        self.assertLessEqual(distinct_result.value, 1.0)
        
    async def test_benchmark_latency(self):
        """Test latency benchmark (AC 6, AC 8)."""
        mock_model = AsyncMock()
        
        # Mock fast embedding generation
        async def fast_generate(batch):
            await asyncio.sleep(0.01)  # 10ms
            return np.random.randn(len(batch), 4096)
            
        mock_model.generate_embedding = fast_generate
        
        self.benchmark.test_texts = ["Text " + str(i) for i in range(10)]
        self.benchmark.batch_sizes = [1, 2, 4]
        
        results = await self.benchmark._benchmark_latency(
            mock_model,
            ModelType.QWEN3_8B
        )
        
        # Should return latency results for each batch size
        self.assertEqual(len(results), 3)
        
        for result in results:
            self.assertEqual(result.metric, BenchmarkMetric.LATENCY)
            self.assertGreater(result.value, 0)
            self.assertEqual(result.unit, "ms")
            
            # Check metadata
            self.assertIn("batch_size", result.metadata)
            self.assertIn("per_sample", result.metadata)
            
            # Latency should be reasonable
            self.assertLess(result.value, 1000)  # Less than 1 second
            
    @patch('src.rag.benchmarks.model_benchmark.memory_usage')
    @patch('src.rag.benchmarks.model_benchmark.psutil.Process')
    async def test_benchmark_memory(self, mock_process_class, mock_memory_usage):
        """Test memory usage benchmark (AC 10)."""
        mock_model = AsyncMock()
        mock_model.generate_embedding = AsyncMock(
            return_value=np.random.randn(1, 4096)
        )
        
        # Mock process memory info
        mock_process = MagicMock()
        mock_process.memory_info.return_value = MagicMock(rss=4 * 1024**3)  # 4GB
        mock_process_class.return_value = mock_process
        
        # Mock memory usage measurement
        mock_memory_usage.return_value = [4096, 4200, 4500, 4300]  # MB
        
        self.benchmark.test_texts = ["Text 1", "Text 2"]
        self.benchmark.batch_sizes = [1]
        
        results = await self.benchmark._benchmark_memory(
            mock_model,
            ModelType.QWEN3_8B
        )
        
        # Should return memory results
        self.assertEqual(len(results), 1)
        
        result = results[0]
        self.assertEqual(result.metric, BenchmarkMetric.MEMORY_USAGE)
        self.assertEqual(result.unit, "GB")
        
        # Memory usage should be reasonable
        self.assertGreaterEqual(result.value, 0)  # Additional memory used
        self.assertLess(result.value, 20)  # Less than 20GB for 8B model
        
    async def test_benchmark_throughput(self):
        """Test throughput benchmark (AC 7)."""
        mock_model = AsyncMock()
        
        # Mock fast batch processing
        async def batch_generate(batch):
            await asyncio.sleep(0.001 * len(batch))  # 1ms per sample
            return np.random.randn(len(batch), 4096)
            
        mock_model.generate_embedding = batch_generate
        
        self.benchmark.test_texts = ["Text " + str(i) for i in range(20)]
        self.benchmark.batch_sizes = [4, 8]
        
        results = await self.benchmark._benchmark_throughput(
            mock_model,
            ModelType.QWEN3_8B
        )
        
        # Should return throughput results
        self.assertEqual(len(results), 2)
        
        for result in results:
            self.assertEqual(result.metric, BenchmarkMetric.THROUGHPUT)
            self.assertEqual(result.unit, "samples/second")
            
            # Throughput should be positive
            self.assertGreater(result.value, 0)
            
            # Check metadata
            self.assertIn("batch_size", result.metadata)
            self.assertIn("total_samples", result.metadata)
            
    @patch('src.rag.benchmarks.model_benchmark.torch.backends.mps.is_available')
    @patch('src.rag.benchmarks.model_benchmark.psutil.Process')
    async def test_benchmark_energy(self, mock_process_class, mock_mps):
        """Test energy efficiency benchmark (Mac M3 specific)."""
        mock_mps.return_value = True  # Simulate Mac M3
        
        mock_model = AsyncMock()
        mock_model.generate_embedding = AsyncMock(
            return_value=np.random.randn(32, 4096)
        )
        
        # Mock process CPU usage
        mock_process = MagicMock()
        mock_process.cpu_percent.side_effect = [20.0, 35.0]  # Start and end CPU
        mock_process_class.return_value = mock_process
        
        self.benchmark.test_texts = ["Text " + str(i) for i in range(32)]
        
        results = await self.benchmark._benchmark_energy(
            mock_model,
            ModelType.QWEN3_8B
        )
        
        # Should return energy results
        self.assertEqual(len(results), 1)
        
        result = results[0]
        self.assertEqual(result.metric, BenchmarkMetric.ENERGY_EFFICIENCY)
        self.assertEqual(result.unit, "relative_score")
        
        # Energy score should be positive
        self.assertGreater(result.value, 0)
        
        # Check metadata
        self.assertIn("avg_cpu_percent", result.metadata)
        self.assertIn("samples_processed", result.metadata)
        
    @patch('src.rag.benchmarks.model_benchmark.Qwen8BEmbedder')
    async def test_benchmark_model(self, mock_8b_class):
        """Test benchmarking a single model."""
        # Setup mock model
        mock_model = AsyncMock()
        mock_model.initialize = AsyncMock()
        mock_model.close = AsyncMock()
        mock_model.generate_embedding = AsyncMock(
            return_value=np.random.randn(10, 4096)
        )
        mock_8b_class.return_value = mock_model
        
        # Setup benchmark
        self.benchmark.test_texts = ["Text " + str(i) for i in range(10)]
        
        # Mock individual benchmark methods
        with patch.object(self.benchmark, '_warmup', new_callable=AsyncMock):
            with patch.object(self.benchmark, '_benchmark_embedding_quality', new_callable=AsyncMock) as mock_quality:
                with patch.object(self.benchmark, '_benchmark_latency', new_callable=AsyncMock) as mock_latency:
                    with patch.object(self.benchmark, '_benchmark_memory', new_callable=AsyncMock) as mock_memory:
                        with patch.object(self.benchmark, '_benchmark_throughput', new_callable=AsyncMock) as mock_throughput:
                            
                            # Setup return values
                            mock_quality.return_value = [
                                BenchmarkResult(
                                    ModelType.QWEN3_8B,
                                    BenchmarkMetric.EMBEDDING_QUALITY,
                                    0.95, "similarity", {}
                                )
                            ]
                            mock_latency.return_value = [
                                BenchmarkResult(
                                    ModelType.QWEN3_8B,
                                    BenchmarkMetric.LATENCY,
                                    100, "ms", {}
                                )
                            ]
                            mock_memory.return_value = [
                                BenchmarkResult(
                                    ModelType.QWEN3_8B,
                                    BenchmarkMetric.MEMORY_USAGE,
                                    8.0, "GB", {}
                                )
                            ]
                            mock_throughput.return_value = [
                                BenchmarkResult(
                                    ModelType.QWEN3_8B,
                                    BenchmarkMetric.THROUGHPUT,
                                    50, "samples/second", {}
                                )
                            ]
                            
                            results = await self.benchmark.benchmark_model(ModelType.QWEN3_8B)
                            
                            # Should initialize and close model
                            mock_model.initialize.assert_called_once()
                            mock_model.close.assert_called_once()
                            
                            # Should run all benchmarks
                            mock_quality.assert_called_once()
                            mock_latency.assert_called_once()
                            mock_memory.assert_called_once()
                            mock_throughput.assert_called_once()
                            
                            # Should return results
                            self.assertEqual(len(results), 4)
                            
    def test_generate_summary(self):
        """Test summary generation from results."""
        # Add test results
        self.benchmark.results = [
            BenchmarkResult(
                ModelType.QWEN3_8B,
                BenchmarkMetric.LATENCY,
                100, "ms", {"batch_size": 1}
            ),
            BenchmarkResult(
                ModelType.QWEN3_8B,
                BenchmarkMetric.LATENCY,
                200, "ms", {"batch_size": 8}
            ),
            BenchmarkResult(
                ModelType.QWEN3_8B,
                BenchmarkMetric.MEMORY_USAGE,
                8.0, "GB", {}
            ),
        ]
        
        summary = self.benchmark._generate_summary()
        
        # Check structure
        self.assertIn(ModelType.QWEN3_8B.value, summary)
        
        # Check 8B model metrics
        model_8b = summary[ModelType.QWEN3_8B.value]
        self.assertIn("latency", model_8b)
        self.assertIn("memory_usage", model_8b)
        
        # Check latency statistics
        latency_stats = model_8b["latency"]
        self.assertAlmostEqual(latency_stats["mean"], 150.0)  # (100+200)/2
        self.assertEqual(latency_stats["min"], 100.0)
        self.assertEqual(latency_stats["max"], 200.0)
        
        # Check memory statistics
        memory_stats = model_8b["memory_usage"]
        self.assertAlmostEqual(memory_stats["mean"], 8.0)
        
    def test_generate_recommendations(self):
        """Test recommendation generation."""
        summary = {
            ModelType.QWEN3_8B.value: {
                "latency": {"mean": 200, "std": 20, "min": 180, "max": 220},
                "memory_usage": {"mean": 8.0, "std": 0.5, "min": 7.5, "max": 8.5},
                "embedding_quality": {"mean": 0.95, "std": 0.02, "min": 0.93, "max": 0.97},
            },
        }
        
        recommendations = self.benchmark._generate_recommendations(summary)
        
        # Should generate recommendations
        self.assertGreater(len(recommendations), 0)
        
        # Should provide performance recommendations based on metrics
        has_performance_rec = any("performance" in r.lower() or "latency" in r.lower() for r in recommendations)
        self.assertTrue(has_performance_rec)
        
    async def test_benchmark_single_model(self):
        """Test benchmarking a single model."""
        # Mock initialize
        with patch.object(self.benchmark, 'initialize', new_callable=AsyncMock):
            # Mock benchmark_model
            with patch.object(self.benchmark, 'benchmark_model', new_callable=AsyncMock) as mock_benchmark:
                
                # Setup results for the model
                mock_benchmark.return_value = [
                    BenchmarkResult(
                        ModelType.QWEN3_8B,
                        BenchmarkMetric.LATENCY,
                        150, "ms", {}
                    ),
                    BenchmarkResult(
                        ModelType.QWEN3_8B,
                        BenchmarkMetric.MEMORY_USAGE,
                        8.0, "GB", {}
                    ),
                ]
                
                # Run benchmark for single model
                results = await self.benchmark.benchmark_model(ModelType.QWEN3_8B)
                
                # Should benchmark the model
                self.assertEqual(mock_benchmark.call_count, 1)
                
                # Check results
                self.assertEqual(len(results), 2)
                self.assertEqual(results[0].model_type, ModelType.QWEN3_8B)
                self.assertEqual(results[0].metric, BenchmarkMetric.LATENCY)
                self.assertEqual(results[1].metric, BenchmarkMetric.MEMORY_USAGE)
                
    def test_save_report(self):
        """Test saving benchmark report to file."""
        # Create test report
        report = BenchmarkReport(
            results=[
                BenchmarkResult(
                    ModelType.QWEN3_8B,
                    BenchmarkMetric.LATENCY,
                    100, "ms", {"test": "value"}
                ),
            ],
            summary={
                ModelType.QWEN3_8B.value: {
                    "latency": {"mean": 100, "std": 10}
                }
            },
            recommendations=["Use 8B model for quality"],
            test_conditions={"samples": 10},
            duration_seconds=60.5,
        )
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = Path(f.name)
            
        try:
            self.benchmark.save_report(report, temp_path)
            
            # Load and verify
            with open(temp_path, 'r') as f:
                saved_data = json.load(f)
                
            self.assertIn("summary", saved_data)
            self.assertIn("recommendations", saved_data)
            self.assertIn("test_conditions", saved_data)
            self.assertEqual(saved_data["duration_seconds"], 60.5)
            
            # Check results
            self.assertEqual(len(saved_data["results"]), 1)
            result = saved_data["results"][0]
            self.assertEqual(result["model"], ModelType.QWEN3_8B.value)
            self.assertEqual(result["metric"], BenchmarkMetric.LATENCY.value)
            self.assertEqual(result["value"], 100)
            
        finally:
            temp_path.unlink()


class TestBenchmarkIntegration(unittest.TestCase):
    """Integration tests for benchmarking framework."""
    
    @pytest.mark.integration
    async def test_performance_analysis(self):
        """Test performance analysis and recommendations."""
        benchmark = ModelBenchmark(num_samples=5, batch_sizes=[1, 2])
        
        # Create results for performance analysis
        benchmark.results = [
            BenchmarkResult(
                ModelType.QWEN3_8B,
                BenchmarkMetric.LATENCY,
                900, "ms", {}  # Exceeds 800ms target
            ),
            BenchmarkResult(
                ModelType.QWEN3_8B,
                BenchmarkMetric.MEMORY_USAGE,
                8.0, "GB", {}
            ),
        ]
        
        summary = benchmark._generate_summary()
        recommendations = benchmark._generate_recommendations(summary)
        
        # Should provide performance recommendations
        self.assertTrue(
            any("latency" in r.lower() or "optimization" in r.lower() 
                for r in recommendations)
        )


if __name__ == "__main__":
    # Run async tests with pytest
    pytest.main([__file__, "-v"])