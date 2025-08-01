# Mac M3 Optimization Guide

## Overview

This guide provides comprehensive optimization strategies for running Universal Claude Thinking v2 on Apple Silicon Mac systems, specifically targeting the M3 Max with 128GB RAM. The optimizations leverage Metal Performance Shaders (MPS) for GPU acceleration and ensure maximum performance for embedding models, rerankers, and vector operations.

## System Requirements

### Hardware Specifications
- **Minimum**: M3 with 36GB RAM
- **Recommended**: M3 Pro/Max with 64GB+ RAM
- **Optimal**: M3 Max with 128GB RAM (your configuration)

### Software Requirements
- macOS Sonoma 14.0 or later
- Xcode Command Line Tools 15.0+
- Python 3.12+ (3.12.8 recommended)
- Homebrew for dependency management

## Environment Setup

### 1. System Configuration
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install system dependencies
brew install cmake
brew install libomp
brew install rust  # For some Python packages
```

### 2. Python Environment
```bash
# Install Python 3.12 with Homebrew
brew install python@3.12

# Create virtual environment
python3.12 -m venv venv
source venv/bin/activate

# Upgrade pip and essential tools
pip install --upgrade pip setuptools wheel

# Note: All library versions are specified in the main architecture document
# to ensure consistency across the project
```

### 3. Environment Variables
```bash
# Add to ~/.zshrc or ~/.bash_profile
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Metal Performance Shaders
export METAL_DEVICE_WRAPPER_TYPE=1
export METAL_DEBUG_ERROR_MODE=0

# Memory optimization
export PYTORCH_MPS_ALLOCATOR_POLICY=garbage_collection
export PYTORCH_MPS_MEMORY_POOL_SIZE=0
```

## Library Requirements

For the complete list of library dependencies with specific versions optimized for Python 3.12 and Mac M3, please refer to the [Comprehensive Library Dependencies](../../docs/architecture.md#comprehensive-library-dependencies) section in the main architecture document.

The architecture document includes:
- Latest stable versions for all libraries
- Mac M3 specific optimization notes
- Memory allocation recommendations
- MPS configuration settings

### Installation Script
```bash
#!/bin/bash
# install-mac-m3.sh

echo "Installing Universal Claude Thinking v2 for Mac M3..."

# Activate virtual environment
source venv/bin/activate

# Install PyTorch with MPS support
pip install torch torchvision torchaudio

# Install core requirements
pip install transformers accelerate sentence-transformers

# Install scientific computing libraries
pip install numpy scipy scikit-learn numba

# Install vector databases
pip install faiss-cpu chromadb sqlite-vec hnswlib

# Install RAG frameworks
pip install langchain langchain-community llama-index

# Install remaining requirements
pip install -r requirements.txt

# Verify MPS availability
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"

echo "Installation complete!"
```

## Model Loading Optimizations

### Qwen3-Embedding-8B Configuration
```python
import torch
from sentence_transformers import SentenceTransformer

class OptimizedEmbeddingModel:
    def __init__(self):
        # Check MPS availability
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Load model with optimizations
        self.model = SentenceTransformer(
            "Qwen/Qwen3-Embedding-8B",
            device=self.device,
            model_kwargs={
                "torch_dtype": torch.float16,  # Use half precision
                "attn_implementation": "sdpa",  # Scaled dot-product attention
            }
        )
        
        # Enable eval mode
        self.model.eval()
        
    def encode(self, texts, batch_size=8):
        # Optimized batch size for M3 Max 128GB
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_tensor=False,  # Return numpy for compatibility
            device=self.device
        )
```

### Qwen3-Reranker-8B Configuration
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

class OptimizedRerankerModel:
    def __init__(self):
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-Reranker-8B",
            padding_side='left',
            use_fast=True  # Use fast tokenizer
        )
        
        # Load model with optimizations
        self.model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-Reranker-8B",
            torch_dtype=torch.float16,
            device_map={"": self.device},
            low_cpu_mem_usage=True,
            attn_implementation="sdpa"
        )
        
        # Enable eval mode and compile
        self.model.eval()
        # torch.compile() not yet fully supported on MPS
        
    @torch.no_grad()
    def score(self, query, documents, batch_size=4):
        # Process in smaller batches for stability
        scores = []
        
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_scores = self._score_batch(query, batch_docs)
            scores.extend(batch_scores)
            
        return scores
```

## Memory Management

### Optimizing for 128GB RAM
```python
class MemoryOptimizedVectorDB:
    def __init__(self, max_memory_gb=100):  # Leave 28GB for system
        self.max_memory = max_memory_gb * 1024 * 1024 * 1024
        self.index = None
        
    def build_index(self, embeddings):
        import faiss
        
        # Use memory-mapped index for large datasets
        if embeddings.nbytes > self.max_memory * 0.5:
            # Create on-disk index
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
            self.index = faiss.IndexIDMap(self.index)
            
            # Add in batches
            batch_size = 10000
            for i in range(0, len(embeddings), batch_size):
                batch = embeddings[i:i + batch_size]
                batch_ids = np.arange(i, i + len(batch))
                self.index.add_with_ids(batch, batch_ids)
        else:
            # In-memory index for smaller datasets
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
            self.index.add(embeddings)
```

### Batch Processing Strategy
```python
class BatchProcessor:
    def __init__(self):
        # Optimal batch sizes for M3 Max 128GB
        self.batch_sizes = {
            'embedding': 32,      # For Qwen3-Embedding-8B
            'reranking': 8,       # For Qwen3-Reranker-8B
            'vector_search': 1000, # For FAISS operations
            'chunking': 100       # For document processing
        }
        
    async def process_documents(self, documents):
        # Process in optimal batches
        results = []
        batch_size = self.batch_sizes['chunking']
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            # Process batch
            batch_results = await self._process_batch(batch)
            results.extend(batch_results)
            
            # Garbage collection every 10 batches
            if i % (batch_size * 10) == 0:
                import gc
                gc.collect()
                
        return results
```

## Performance Monitoring

### MPS GPU Monitoring
```python
import subprocess
import psutil

class SystemMonitor:
    @staticmethod
    def get_gpu_usage():
        # Use ioreg to get Metal GPU usage
        try:
            cmd = "ioreg -l | grep 'PerformanceStatistics'"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            # Parse output for GPU metrics
            return result.stdout
        except:
            return "GPU monitoring not available"
    
    @staticmethod
    def get_memory_usage():
        memory = psutil.virtual_memory()
        return {
            'total_gb': memory.total / (1024**3),
            'used_gb': memory.used / (1024**3),
            'available_gb': memory.available / (1024**3),
            'percent': memory.percent
        }
    
    @staticmethod
    def optimize_memory():
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear PyTorch cache
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
```

## Troubleshooting

### Common Issues and Solutions

1. **MPS Fallback Warnings**
```python
# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*MPS.*")
```

2. **Memory Pressure**
```python
# Monitor and manage memory
def check_memory_pressure():
    memory = psutil.virtual_memory()
    if memory.percent > 80:
        import gc
        gc.collect()
        torch.mps.empty_cache()
        print(f"Memory cleaned. Usage: {memory.percent}%")
```

3. **Model Loading Issues**
```python
# Safe model loading with fallback
def load_model_safe(model_name):
    try:
        model = AutoModel.from_pretrained(
            model_name,
            device_map="mps",
            torch_dtype=torch.float16
        )
    except Exception as e:
        print(f"MPS loading failed: {e}")
        model = AutoModel.from_pretrained(
            model_name,
            device_map="cpu",
            torch_dtype=torch.float32
        )
    return model
```

## Benchmarking Script

```python
# benchmark-m3.py
import time
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

def benchmark_embedding():
    print("Benchmarking Qwen3-Embedding-8B on M3 Max...")
    
    model = SentenceTransformer("Qwen/Qwen3-Embedding-8B")
    texts = ["Sample text " * 10] * 100  # 100 medium-length texts
    
    # Warmup
    _ = model.encode(texts[:5])
    
    # Benchmark
    start = time.time()
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
    duration = time.time() - start
    
    print(f"Processed {len(texts)} texts in {duration:.2f}s")
    print(f"Throughput: {len(texts)/duration:.2f} texts/second")
    print(f"Embedding shape: {embeddings.shape}")

if __name__ == "__main__":
    benchmark_embedding()
```

## Best Practices

1. **Use MPS for Neural Networks**: Always check `torch.backends.mps.is_available()`
2. **Optimize Batch Sizes**: M3 Max handles larger batches efficiently
3. **Monitor Memory Usage**: With 128GB, you can cache more but monitor usage
4. **Use Half Precision**: float16 works well on M3 for most models
5. **Parallel Processing**: Leverage all CPU cores for data preprocessing
6. **Profile Your Code**: Use `py-spy` or built-in profilers to find bottlenecks

This optimization guide ensures maximum performance for Universal Claude Thinking v2 on Mac M3 systems while maintaining stability and efficiency.