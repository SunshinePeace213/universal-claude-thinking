"""
Model Benchmarking Framework.

Comprehensive benchmarking system for comparing Qwen3 embedding models
on accuracy, latency, memory usage, and throughput metrics.
"""

import asyncio
import gc
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil
import torch
from memory_profiler import memory_usage

from src.rag.embedder import ModelType, Qwen8B4BitEmbedder, Qwen8BEmbedder

logger = logging.getLogger(__name__)


class BenchmarkMetric(Enum):
    """Benchmark metrics to measure."""
    
    EMBEDDING_QUALITY = "embedding_quality"
    LATENCY = "latency"
    MEMORY_USAGE = "memory_usage"
    THROUGHPUT = "throughput"
    ENERGY_EFFICIENCY = "energy_efficiency"


@dataclass
class BenchmarkResult:
    """Result from a single benchmark test."""
    
    model_type: ModelType
    metric: BenchmarkMetric
    value: float
    unit: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class BenchmarkReport:
    """Complete benchmark report for model comparison."""
    
    results: List[BenchmarkResult]
    summary: Dict[str, Dict[str, float]]  # model -> metric -> value
    recommendations: List[str]
    test_conditions: Dict[str, Any]
    duration_seconds: float


class ModelBenchmark:
    """
    Comprehensive benchmarking framework for embedding model comparison.
    
    Measures and compares embedding quality, latency, memory usage,
    throughput, and energy efficiency between models.
    """
    
    def __init__(
        self,
        test_data_path: Optional[Path] = None,
        num_samples: int = 100,
        batch_sizes: List[int] = None,
        warmup_iterations: int = 5,
    ) -> None:
        """
        Initialize the benchmark framework.
        
        Args:
            test_data_path: Path to test data file
            num_samples: Number of samples for testing
            batch_sizes: Batch sizes to test
            warmup_iterations: Warmup iterations before measurement
        """
        self.test_data_path = test_data_path
        self.num_samples = num_samples
        self.batch_sizes = batch_sizes or [1, 8, 16, 32]
        self.warmup_iterations = warmup_iterations
        
        # Test data
        self.test_texts: List[str] = []
        self.test_embeddings: Dict[ModelType, np.ndarray] = {}
        
        # Results storage
        self.results: List[BenchmarkResult] = []
        
        # Models to benchmark
        self.models: Dict[ModelType, Any] = {}
        
    async def initialize(self) -> None:
        """Initialize benchmark framework and load test data."""
        logger.info("Initializing benchmark framework...")
        
        # Load or generate test data
        if self.test_data_path and self.test_data_path.exists():
            self._load_test_data()
        else:
            self._generate_test_data()
            
        logger.info(f"Loaded {len(self.test_texts)} test samples")
        
    def _generate_test_data(self) -> None:
        """Generate synthetic test data."""
        # Generate diverse text samples
        templates = [
            "The quick brown fox jumps over the lazy dog in {context}.",
            "Machine learning models can {action} when given {input}.",
            "The {adjective} system processes {num} requests per second.",
            "Understanding {concept} requires knowledge of {prerequisite}.",
            "In {year}, the technology will {prediction} significantly.",
        ]
        
        contexts = ["morning", "evening", "summer", "winter", "space"]
        actions = ["learn", "adapt", "optimize", "generalize", "predict"]
        adjectives = ["distributed", "scalable", "efficient", "robust", "intelligent"]
        concepts = ["embeddings", "attention", "transformers", "convolution", "gradients"]
        
        self.test_texts = []
        for i in range(self.num_samples):
            template = templates[i % len(templates)]
            text = template.format(
                context=contexts[i % len(contexts)],
                action=actions[i % len(actions)],
                adjective=adjectives[i % len(adjectives)],
                num=(i + 1) * 100,
                concept=concepts[i % len(concepts)],
                prerequisite=concepts[(i + 1) % len(concepts)],
                year=2025 + (i % 5),
                prediction=actions[(i + 2) % len(actions)],
                input="data",
            )
            self.test_texts.append(text)
            
    def _load_test_data(self) -> None:
        """Load test data from file."""
        with open(self.test_data_path, "r") as f:
            data = json.load(f)
            self.test_texts = data.get("texts", [])[:self.num_samples]
            
    async def benchmark_model(
        self,
        model_type: ModelType,
        model_path: Optional[Path] = None,
    ) -> List[BenchmarkResult]:
        """
        Run all benchmarks for a single model.
        
        Args:
            model_type: Type of model to benchmark
            model_path: Optional path to model files
            
        Returns:
            List of benchmark results
        """
        logger.info(f"Benchmarking {model_type.value}...")
        
        # Initialize model
        if model_type == ModelType.QWEN3_8B:
            model = Qwen8BEmbedder(model_path=model_path)
        else:
            model = Qwen8B4BitEmbedder(model_path=model_path)
            
        try:
            await model.initialize()
            self.models[model_type] = model
            
            # Warmup
            await self._warmup(model)
            
            # Run benchmarks
            results = []
            
            # 1. Embedding Quality
            quality_results = await self._benchmark_embedding_quality(model, model_type)
            results.extend(quality_results)
            
            # 2. Latency
            latency_results = await self._benchmark_latency(model, model_type)
            results.extend(latency_results)
            
            # 3. Memory Usage
            memory_results = await self._benchmark_memory(model, model_type)
            results.extend(memory_results)
            
            # 4. Throughput
            throughput_results = await self._benchmark_throughput(model, model_type)
            results.extend(throughput_results)
            
            # 5. Energy Efficiency (Mac M3 specific)
            if torch.backends.mps.is_available():
                energy_results = await self._benchmark_energy(model, model_type)
                results.extend(energy_results)
                
            self.results.extend(results)
            return results
            
        finally:
            await model.close()
            
    async def _warmup(self, model: Any) -> None:
        """Warmup model before benchmarking."""
        logger.debug(f"Warming up model ({self.warmup_iterations} iterations)...")
        
        warmup_texts = self.test_texts[:min(10, len(self.test_texts))]
        for _ in range(self.warmup_iterations):
            await model.generate_embedding(warmup_texts)
            
    async def _benchmark_embedding_quality(
        self,
        model: Any,
        model_type: ModelType,
    ) -> List[BenchmarkResult]:
        """Benchmark embedding quality through similarity preservation."""
        results = []
        
        # Generate embeddings for test set
        embeddings = []
        for i in range(0, len(self.test_texts), 32):
            batch = self.test_texts[i:i+32]
            batch_embeddings = await model.generate_embedding(batch)
            embeddings.append(batch_embeddings)
            
        all_embeddings = np.vstack(embeddings)
        self.test_embeddings[model_type] = all_embeddings
        
        # Test 1: Self-similarity (should be 1.0)
        self_similarities = []
        for emb in all_embeddings[:10]:
            sim = np.dot(emb, emb) / (np.linalg.norm(emb) ** 2)
            self_similarities.append(sim)
            
        avg_self_sim = np.mean(self_similarities)
        results.append(BenchmarkResult(
            model_type=model_type,
            metric=BenchmarkMetric.EMBEDDING_QUALITY,
            value=avg_self_sim,
            unit="cosine_similarity",
            metadata={"test": "self_similarity"},
        ))
        
        # Test 2: Distinctiveness (embeddings should be different)
        if len(all_embeddings) > 1:
            pairwise_sims = []
            for i in range(min(20, len(all_embeddings))):
                for j in range(i+1, min(20, len(all_embeddings))):
                    sim = np.dot(all_embeddings[i], all_embeddings[j]) / (
                        np.linalg.norm(all_embeddings[i]) * np.linalg.norm(all_embeddings[j])
                    )
                    pairwise_sims.append(sim)
                    
            avg_distinctiveness = 1.0 - np.mean(pairwise_sims)
            results.append(BenchmarkResult(
                model_type=model_type,
                metric=BenchmarkMetric.EMBEDDING_QUALITY,
                value=avg_distinctiveness,
                unit="distinctiveness",
                metadata={"test": "pairwise_distinctiveness"},
            ))
            
        return results
        
    async def _benchmark_latency(
        self,
        model: Any,
        model_type: ModelType,
    ) -> List[BenchmarkResult]:
        """Benchmark inference latency."""
        results = []
        
        for batch_size in self.batch_sizes:
            if batch_size > len(self.test_texts):
                continue
                
            batch = self.test_texts[:batch_size]
            
            # Measure latency
            latencies = []
            for _ in range(10):  # 10 runs for averaging
                start = time.perf_counter()
                await model.generate_embedding(batch)
                end = time.perf_counter()
                latencies.append((end - start) * 1000)  # Convert to ms
                
            avg_latency = np.mean(latencies)
            std_latency = np.std(latencies)
            
            results.append(BenchmarkResult(
                model_type=model_type,
                metric=BenchmarkMetric.LATENCY,
                value=avg_latency,
                unit="ms",
                metadata={
                    "batch_size": batch_size,
                    "std": std_latency,
                    "per_sample": avg_latency / batch_size,
                },
            ))
            
        return results
        
    async def _benchmark_memory(
        self,
        model: Any,
        model_type: ModelType,
    ) -> List[BenchmarkResult]:
        """Benchmark memory usage."""
        results = []
        
        # Get baseline memory
        process = psutil.Process()
        baseline_memory = process.memory_info().rss / (1024 ** 3)  # GB
        
        for batch_size in self.batch_sizes:
            if batch_size > len(self.test_texts):
                continue
                
            batch = self.test_texts[:batch_size]
            
            # Measure memory during inference
            def inference():
                asyncio.run(model.generate_embedding(batch))
                
            mem_usage = memory_usage(inference, interval=0.1, timeout=30)
            peak_memory = max(mem_usage) / 1024  # Convert to GB
            
            results.append(BenchmarkResult(
                model_type=model_type,
                metric=BenchmarkMetric.MEMORY_USAGE,
                value=peak_memory - baseline_memory,
                unit="GB",
                metadata={
                    "batch_size": batch_size,
                    "baseline_gb": baseline_memory,
                    "peak_gb": peak_memory,
                },
            ))
            
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        return results
        
    async def _benchmark_throughput(
        self,
        model: Any,
        model_type: ModelType,
    ) -> List[BenchmarkResult]:
        """Benchmark throughput (samples/second)."""
        results = []
        
        for batch_size in self.batch_sizes:
            if batch_size > len(self.test_texts):
                continue
                
            # Process multiple batches
            num_batches = min(10, len(self.test_texts) // batch_size)
            total_samples = num_batches * batch_size
            
            start = time.perf_counter()
            for i in range(num_batches):
                batch = self.test_texts[i*batch_size:(i+1)*batch_size]
                await model.generate_embedding(batch)
            end = time.perf_counter()
            
            duration = end - start
            throughput = total_samples / duration
            
            results.append(BenchmarkResult(
                model_type=model_type,
                metric=BenchmarkMetric.THROUGHPUT,
                value=throughput,
                unit="samples/second",
                metadata={
                    "batch_size": batch_size,
                    "total_samples": total_samples,
                    "duration_seconds": duration,
                },
            ))
            
        return results
        
    async def _benchmark_energy(
        self,
        model: Any,
        model_type: ModelType,
    ) -> List[BenchmarkResult]:
        """Benchmark energy efficiency (Mac M3 specific)."""
        results = []
        
        # This is a simplified energy measurement
        # In production, would use powermetrics or similar tools
        
        batch = self.test_texts[:32]
        
        # Measure power draw proxy through execution time and resource usage
        process = psutil.Process()
        
        start_cpu = process.cpu_percent()
        start_time = time.perf_counter()
        
        # Run inference multiple times
        for _ in range(10):
            await model.generate_embedding(batch)
            
        end_time = time.perf_counter()
        end_cpu = process.cpu_percent()
        
        duration = end_time - start_time
        avg_cpu = (start_cpu + end_cpu) / 2
        
        # Simplified energy score (lower is better)
        energy_score = duration * avg_cpu / 100
        
        results.append(BenchmarkResult(
            model_type=model_type,
            metric=BenchmarkMetric.ENERGY_EFFICIENCY,
            value=energy_score,
            unit="relative_score",
            metadata={
                "duration_seconds": duration,
                "avg_cpu_percent": avg_cpu,
                "samples_processed": 320,  # 10 * 32
            },
        ))
        
        return results
        
    async def compare_models(
        self,
        model_paths: Optional[Dict[ModelType, Path]] = None,
    ) -> BenchmarkReport:
        """
        Compare all models and generate report.
        
        Args:
            model_paths: Optional paths to model files
            
        Returns:
            Comprehensive benchmark report
        """
        start_time = time.time()
        
        # Initialize
        await self.initialize()
        
        # Benchmark each model
        model_types = [ModelType.QWEN3_8B, ModelType.QWEN3_8B_4BIT]
        
        for model_type in model_types:
            model_path = model_paths.get(model_type) if model_paths else None
            try:
                await self.benchmark_model(model_type, model_path)
            except Exception as e:
                logger.error(f"Failed to benchmark {model_type.value}: {e}")
                
        # Generate summary
        summary = self._generate_summary()
        
        # Generate recommendations
        recommendations = self._generate_recommendations(summary)
        
        # Test conditions
        test_conditions = {
            "num_samples": self.num_samples,
            "batch_sizes": self.batch_sizes,
            "warmup_iterations": self.warmup_iterations,
            "platform": "Mac M3" if torch.backends.mps.is_available() else "CPU",
            "timestamp": datetime.now().isoformat(),
        }
        
        duration = time.time() - start_time
        
        return BenchmarkReport(
            results=self.results,
            summary=summary,
            recommendations=recommendations,
            test_conditions=test_conditions,
            duration_seconds=duration,
        )
        
    def _generate_summary(self) -> Dict[str, Dict[str, float]]:
        """Generate summary statistics from results."""
        summary = {}
        
        for result in self.results:
            model = result.model_type.value
            metric = result.metric.value
            
            if model not in summary:
                summary[model] = {}
                
            # Aggregate by metric type
            if metric not in summary[model]:
                summary[model][metric] = []
                
            summary[model][metric].append(result.value)
            
        # Calculate averages
        for model in summary:
            for metric in summary[model]:
                values = summary[model][metric]
                summary[model][metric] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                }
                
        return summary
        
    def _generate_recommendations(
        self,
        summary: Dict[str, Dict[str, Any]],
    ) -> List[str]:
        """Generate recommendations based on benchmark results."""
        recommendations = []
        
        if not summary:
            return ["No benchmark data available for recommendations"]
            
        # Compare models if both tested
        if len(summary) >= 2:
            model_8b = ModelType.QWEN3_8B.value
            model_4bit = ModelType.QWEN3_8B_4BIT.value
            
            # Latency comparison
            if "latency" in summary.get(model_8b, {}) and "latency" in summary.get(model_4bit, {}):
                latency_8b = summary[model_8b]["latency"]["mean"]
                latency_4bit = summary[model_4bit]["latency"]["mean"]
                
                if latency_4bit < latency_8b * 0.7:
                    recommendations.append(
                        f"Use {model_4bit} for latency-critical applications "
                        f"({latency_4bit:.2f}ms vs {latency_8b:.2f}ms)"
                    )
                    
            # Memory comparison
            if "memory_usage" in summary.get(model_8b, {}) and "memory_usage" in summary.get(model_4bit, {}):
                memory_8b = summary[model_8b]["memory_usage"]["mean"]
                memory_4bit = summary[model_4bit]["memory_usage"]["mean"]
                
                if memory_4bit < memory_8b * 0.5:
                    recommendations.append(
                        f"Use {model_4bit} for memory-constrained environments "
                        f"({memory_4bit:.2f}GB vs {memory_8b:.2f}GB)"
                    )
                    
            # Quality comparison
            if "embedding_quality" in summary.get(model_8b, {}) and "embedding_quality" in summary.get(model_4bit, {}):
                quality_8b = summary[model_8b]["embedding_quality"]["mean"]
                quality_4bit = summary[model_4bit]["embedding_quality"]["mean"]
                
                if quality_8b > quality_4bit * 1.05:  # 5% better triggers recommendation
                    recommendations.append(
                        f"Use {model_8b} when embedding quality is critical "
                        f"({quality_8b:.3f} vs {quality_4bit:.3f})"
                    )
                    
        # General recommendations
        for model, metrics in summary.items():
            if "latency" in metrics and metrics["latency"]["mean"] > 800:
                recommendations.append(
                    f"Consider batch size optimization for {model} "
                    f"(current latency: {metrics['latency']['mean']:.2f}ms)"
                )
                
        if not recommendations:
            recommendations.append("Both models perform within acceptable parameters")
            
        return recommendations
        
    def save_report(
        self,
        report: BenchmarkReport,
        output_path: Path,
    ) -> None:
        """Save benchmark report to file."""
        report_dict = {
            "summary": report.summary,
            "recommendations": report.recommendations,
            "test_conditions": report.test_conditions,
            "duration_seconds": report.duration_seconds,
            "results": [
                {
                    "model": r.model_type.value,
                    "metric": r.metric.value,
                    "value": r.value,
                    "unit": r.unit,
                    "metadata": r.metadata,
                    "timestamp": r.timestamp.isoformat(),
                }
                for r in report.results
            ],
        }
        
        with open(output_path, "w") as f:
            json.dump(report_dict, f, indent=2)
            
        logger.info(f"Benchmark report saved to {output_path}")