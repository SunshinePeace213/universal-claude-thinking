"""
Stage 2: Semantic Matching with Embeddings
Uses sentence-transformers and ChromaDB for similarity search
Part of Story 1.2: Request Classification Engine with Delegation Integration
"""

import logging
import platform
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SemanticMatchResult:
    """Result from semantic matching"""
    matched: bool
    agent: str | None
    confidence: float
    similarity_score: float
    processing_time_ms: float
    method: str  # "embeddings" or "fallback"


class SemanticMatcher:
    """
    Stage 2: Semantic matching using embeddings for agent selection.
    Uses sentence-transformers and ChromaDB for efficient similarity search.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_mlx: bool = None):
        """
        Initialize semantic matcher with embedding model.

        Args:
            model_name: Name of the sentence-transformer model
            use_mlx: Use MLX framework on Mac (auto-detect if None)
        """
        self.model_name = model_name
        self.use_mlx = use_mlx if use_mlx is not None else self._detect_mac_platform()
        self.embedder = None
        self.chroma_client = None
        self.collection = None
        self._initialized = False

    def _detect_mac_platform(self) -> bool:
        """Detect if running on Mac with Apple Silicon"""
        is_mac = platform.system() == "Darwin"
        if is_mac:
            # Check for Apple Silicon
            processor = platform.processor()
            return "arm" in processor.lower() or "apple" in processor.lower()
        return False

    async def initialize(self):
        """Initialize embedding model and ChromaDB collection"""
        if self._initialized:
            return

        try:
            # Import dependencies
            if self.use_mlx:
                logger.info("Using MLX framework for Mac optimization")
                # MLX imports would go here if available
                # For now, fall back to standard sentence-transformers
                from sentence_transformers import SentenceTransformer
                self.embedder = SentenceTransformer(self.model_name)
            else:
                from sentence_transformers import SentenceTransformer
                self.embedder = SentenceTransformer(self.model_name)

            # Initialize ChromaDB
            import chromadb
            from chromadb.config import Settings

            # Use in-memory ChromaDB for fast performance
            self.chroma_client = chromadb.Client(Settings(
                is_persistent=False,
                anonymized_telemetry=False
            ))

            # Create or get collection for agent embeddings
            self.collection = self.chroma_client.get_or_create_collection(
                name="agent_capabilities",
                metadata={"hnsw:space": "cosine"}
            )

            # Pre-populate with agent capability embeddings
            await self._populate_agent_embeddings()

            self._initialized = True
            logger.info(f"Semantic matcher initialized with {self.model_name}")

        except ImportError as e:
            logger.warning(f"Failed to import required libraries: {e}")
            logger.warning("Semantic matching will use fallback mode")
            self._initialized = False

    async def _populate_agent_embeddings(self):
        """Pre-populate ChromaDB with agent capability embeddings"""
        if not self.collection:
            return

        # Define agent capabilities and descriptions
        agent_descriptions = {
            'PE': [
                "enhance prompt quality and clarity",
                "validate and improve input requests",
                "assess prompt quality and structure",
                "clarify ambiguous requirements",
                "prompt engineering and optimization"
            ],
            'R1': [
                "research and gather information",
                "web search and data collection",
                "verify sources and facts",
                "compile research findings",
                "investigate topics and concepts"
            ],
            'A1': [
                "logical reasoning and analysis",
                "problem solving and debugging",
                "systematic thinking and deduction",
                "create reasoning chains",
                "analyze complex problems"
            ],
            'E1': [
                "evaluate quality and correctness",
                "validate solutions and outputs",
                "detect errors and issues",
                "quality assurance and testing",
                "review code and implementations"
            ],
            'T1': [
                "execute tools and commands",
                "browser automation and testing",
                "system integration and APIs",
                "automate workflows and tasks",
                "interact with external services"
            ],
            'W1': [
                "create written content",
                "technical documentation",
                "creative writing and copywriting",
                "generate reports and articles",
                "compose professional documents"
            ],
            'I1': [
                "user interaction and communication",
                "clarify requirements and intentions",
                "interface design and UX",
                "facilitate understanding",
                "manage conversations"
            ]
        }

        # Generate embeddings for each agent
        all_ids = []
        all_embeddings = []
        all_metadatas = []
        all_documents = []

        for agent, descriptions in agent_descriptions.items():
            for i, desc in enumerate(descriptions):
                doc_id = f"{agent}_{i}"
                embedding = self.embedder.encode(desc).tolist()

                all_ids.append(doc_id)
                all_embeddings.append(embedding)
                all_metadatas.append({"agent": agent, "capability": desc})
                all_documents.append(desc)

        # Add to ChromaDB collection
        self.collection.add(
            ids=all_ids,
            embeddings=all_embeddings,
            metadatas=all_metadatas,
            documents=all_documents
        )

        logger.info(f"Populated {len(all_ids)} agent capability embeddings")

    async def match(self, user_input: str, task_type: str | None = None,
                   keyword_result: Any | None = None) -> SemanticMatchResult:
        """
        Perform semantic matching using embeddings.

        Args:
            user_input: The user's request text
            task_type: Optional task type from classification
            keyword_result: Optional result from keyword matching

        Returns:
            SemanticMatchResult with agent and confidence
        """
        start_time = time.perf_counter()

        # Check if semantic matching is available
        if not self._initialized:
            await self.initialize()

        if not self._initialized or not self.collection:
            # Fallback mode without embeddings
            return await self._fallback_match(user_input, task_type)

        try:
            # Generate embedding for user input
            query_embedding = self.embedder.encode(user_input).tolist()

            # Query ChromaDB for similar agent capabilities
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=5,  # Get top 5 matches
                include=["metadatas", "distances"]
            )

            if results and results['metadatas'] and results['metadatas'][0]:
                # Process results
                agent_scores = {}
                for metadata, distance in zip(results['metadatas'][0], results['distances'][0], strict=False):
                    agent = metadata['agent']
                    # Convert distance to similarity (1 - distance for cosine)
                    similarity = 1 - distance

                    if agent not in agent_scores:
                        agent_scores[agent] = []
                    agent_scores[agent].append(similarity)

                # Calculate average similarity per agent
                agent_avg_scores = {
                    agent: np.mean(scores)
                    for agent, scores in agent_scores.items()
                }

                # Get best agent
                best_agent = max(agent_avg_scores, key=agent_avg_scores.get)
                best_score = agent_avg_scores[best_agent]

                # Apply task type boost if available
                if task_type:
                    best_score = self._apply_task_type_boost(best_agent, task_type, best_score)

                # Check if score meets threshold
                if best_score >= 0.7:  # 0.7 threshold for semantic matching
                    processing_time = (time.perf_counter() - start_time) * 1000
                    return SemanticMatchResult(
                        matched=True,
                        agent=best_agent,
                        confidence=min(best_score * 1.1, 1.0),  # Boost confidence slightly
                        similarity_score=best_score,
                        processing_time_ms=processing_time,
                        method="embeddings"
                    )

        except Exception as e:
            logger.error(f"Error in semantic matching: {e}")

        # No match found or error occurred
        processing_time = (time.perf_counter() - start_time) * 1000
        return SemanticMatchResult(
            matched=False,
            agent=None,
            confidence=0.0,
            similarity_score=0.0,
            processing_time_ms=processing_time,
            method="embeddings"
        )

    async def _fallback_match(self, user_input: str, task_type: str | None) -> SemanticMatchResult:
        """
        Fallback matching when embeddings are not available.
        Uses simple heuristics based on task type.
        """
        start_time = time.perf_counter()

        # Simple task type to agent mapping
        task_agent_map = {
            'simple_direct': 'PE',
            'complex_multi_step': 'A1',
            'research_required': 'R1',
            'web_testing': 'T1',
            'debugging_error': 'A1',
        }

        agent = None
        confidence = 0.5  # Lower confidence for fallback

        if task_type and task_type in task_agent_map:
            agent = task_agent_map[task_type]
            confidence = 0.6

        processing_time = (time.perf_counter() - start_time) * 1000

        if agent:
            return SemanticMatchResult(
                matched=True,
                agent=agent,
                confidence=confidence,
                similarity_score=confidence,
                processing_time_ms=processing_time,
                method="fallback"
            )
        else:
            return SemanticMatchResult(
                matched=False,
                agent=None,
                confidence=0.0,
                similarity_score=0.0,
                processing_time_ms=processing_time,
                method="fallback"
            )

    def _apply_task_type_boost(self, agent: str, task_type: str, score: float) -> float:
        """Apply confidence boost based on task type alignment"""
        task_agent_map = {
            'simple_direct': 'PE',
            'complex_multi_step': 'A1',
            'research_required': 'R1',
            'web_testing': 'T1',
            'debugging_error': 'A1',
        }

        if task_type in task_agent_map and task_agent_map[task_type] == agent:
            return min(score * 1.15, 1.0)

        return score
