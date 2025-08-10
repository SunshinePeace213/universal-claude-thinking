"""Data models for atomic prompt analysis."""

from dataclasses import dataclass
from typing import Any


@dataclass
class AtomicAnalysis:
    """Result of atomic prompt analysis."""

    structure: dict[str, str | None]
    quality_score: float
    gaps: list[str]
    enhancement_suggestions: list[str]
    processing_time_ms: float
    prompt_hash: str
    rationale: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "structure": self.structure,
            "quality_score": self.quality_score,
            "gaps": self.gaps,
            "enhancement_suggestions": self.enhancement_suggestions,
            "processing_time_ms": self.processing_time_ms,
            "prompt_hash": self.prompt_hash,
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AtomicAnalysis":
        """Create instance from dictionary.

        Args:
            data: Dictionary with analysis data

        Returns:
            AtomicAnalysis instance
        """
        return cls(
            structure=data["structure"],
            quality_score=data["quality_score"],
            gaps=data["gaps"],
            enhancement_suggestions=data.get("enhancement_suggestions", data.get("suggestions", [])),
            processing_time_ms=data.get("processing_time_ms", 0.0),
            prompt_hash=data["prompt_hash"],
            rationale=data.get("rationale"),
        )

    def is_high_quality(self) -> bool:
        """Check if analysis indicates high quality prompt.

        Returns:
            True if quality score is 7.0 or higher
        """
        return self.quality_score >= 7.0

    def needs_enhancement(self) -> bool:
        """Check if prompt needs enhancement.

        Returns:
            True if quality score is below 7.0
        """
        return self.quality_score < 7.0

    def has_critical_gaps(self) -> bool:
        """Check if prompt has critical gaps.

        Returns:
            True if task is missing from gaps
        """
        return "task" in self.gaps
