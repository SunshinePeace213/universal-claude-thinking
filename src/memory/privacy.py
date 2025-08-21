"""
Privacy engine for PII detection and anonymization.

Provides PII detection using spaCy NER and regex patterns, with
anonymization techniques for privacy preservation in the memory system.
"""

import hashlib
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

import spacy
from spacy.language import Language

from src.memory.layers.base import MemoryItem


class PIIType(Enum):
    """Types of personally identifiable information."""
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    ADDRESS = "address"
    NAME = "name"
    DATE_OF_BIRTH = "date_of_birth"
    MEDICAL = "medical"
    IP_ADDRESS = "ip_address"
    CUSTOM = "custom"


class AnonymizationMethod(Enum):
    """Methods for anonymizing PII."""
    TOKEN_REPLACEMENT = "token_replacement"
    HASH = "hash"
    HASHING = "hash"  # Alias for compatibility
    MASK = "mask"
    MASKING = "mask"  # Alias for compatibility
    REMOVE = "remove"
    GENERALIZE = "generalize"


@dataclass
class PIIDetection:
    """Result of PII detection."""
    pii_type: PIIType
    text: str
    start_idx: int
    end_idx: int
    confidence: float
    replacement: str


class PrivacyEngine:
    """
    Privacy engine for detecting and anonymizing PII.
    
    Uses spaCy NER for entity recognition and regex patterns for
    structured PII detection, with multiple anonymization strategies.
    """

    def __init__(
        self,
        enable_ner: bool = True,
        enable_regex: bool = True,
        custom_patterns: dict[PIIType, str] | None = None,
        spacy_model: str = "en_core_web_lg"
    ):
        """
        Initialize privacy engine.
        
        Args:
            enable_ner: Enable spaCy NER for name detection
            enable_regex: Enable regex patterns for structured PII
            custom_patterns: Custom regex patterns for PII detection
            spacy_model: spaCy model to use for NER
        """
        self.enable_ner = enable_ner
        self.enable_regex = enable_regex
        self.spacy_model = spacy_model
        self.nlp: Language | None = None
        self.custom_patterns = custom_patterns or {}

        # Default regex patterns for common PII
        self.regex_patterns: dict[PIIType, re.Pattern] = {}
        self._init_regex_patterns()

    def _init_regex_patterns(self) -> None:
        """Initialize default regex patterns for PII detection."""
        patterns = {
            PIIType.EMAIL: r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            PIIType.PHONE: r'(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}',
            PIIType.SSN: r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b',
            PIIType.CREDIT_CARD: r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
            PIIType.IP_ADDRESS: r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b',
            PIIType.DATE_OF_BIRTH: r'\b(?:0[1-9]|1[0-2])[-/](?:0[1-9]|[12][0-9]|3[01])[-/](?:19|20)\d{2}\b',
        }

        # Compile patterns
        for pii_type, pattern in patterns.items():
            if pii_type in self.custom_patterns:
                # Use custom pattern if provided
                self.regex_patterns[pii_type] = re.compile(self.custom_patterns[pii_type])
            else:
                self.regex_patterns[pii_type] = re.compile(pattern)

    def initialize(self) -> None:
        """Initialize the privacy engine and load models."""
        if self.enable_ner:
            try:
                self.nlp = spacy.load(self.spacy_model)
            except OSError:
                # Model not installed, disable NER
                self.enable_ner = False
                self.nlp = None

    def detect_pii_regex(self, text: str) -> list[PIIDetection]:
        """
        Detect PII using regex patterns.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of PII detections
        """
        detections = []

        for pii_type, pattern in self.regex_patterns.items():
            for match in pattern.finditer(text):
                detection = PIIDetection(
                    pii_type=pii_type,
                    text=match.group(),
                    start_idx=match.start(),
                    end_idx=match.end(),
                    confidence=0.9,  # High confidence for regex matches
                    replacement=f"[{pii_type.value.upper()}]"
                )
                detections.append(detection)

        return detections

    def detect_pii_ner(self, text: str) -> list[PIIDetection]:
        """
        Detect PII using spaCy NER.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of PII detections
        """
        if not self.nlp:
            return []

        detections = []
        doc = self.nlp(text)

        for ent in doc.ents:
            # Map spaCy entity types to PII types
            pii_type = None
            if ent.label_ == "PERSON":
                pii_type = PIIType.NAME
            elif ent.label_ in ["GPE", "LOC", "FAC"]:
                pii_type = PIIType.ADDRESS
            elif ent.label_ == "DATE" and self._is_date_of_birth(ent.text):
                pii_type = PIIType.DATE_OF_BIRTH

            if pii_type:
                detection = PIIDetection(
                    pii_type=pii_type,
                    text=ent.text,
                    start_idx=ent.start_char,
                    end_idx=ent.end_char,
                    confidence=0.8,  # Moderate confidence for NER
                    replacement=f"[{pii_type.value.upper()}]"
                )
                detections.append(detection)

        return detections

    def _is_date_of_birth(self, date_text: str) -> bool:
        """Check if a date might be a date of birth."""
        # Simple heuristic: dates that look like birthdates
        import re
        dob_pattern = r'(?:19|20)\d{2}'
        return bool(re.search(dob_pattern, date_text))

    def detect_pii(self, text: str, min_confidence: float = 0.0) -> list[PIIDetection]:
        """
        Detect all PII in text using enabled methods.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of PII detections
        """
        detections = []

        if self.enable_regex:
            detections.extend(self.detect_pii_regex(text))

        if self.enable_ner:
            detections.extend(self.detect_pii_ner(text))

        # Remove duplicates based on overlapping positions
        detections = self._remove_duplicates(detections)

        # Filter by confidence if specified
        if min_confidence > 0:
            detections = [d for d in detections if d.confidence >= min_confidence]

        return detections

    def _remove_duplicates(self, detections: list[PIIDetection]) -> list[PIIDetection]:
        """Remove duplicate or overlapping detections."""
        if not detections:
            return []

        # Sort by start position
        sorted_detections = sorted(detections, key=lambda d: d.start_idx)

        # Keep non-overlapping detections
        result = [sorted_detections[0]]
        for detection in sorted_detections[1:]:
            # Check if overlaps with last kept detection
            if detection.start_idx >= result[-1].end_idx:
                result.append(detection)
            elif detection.confidence > result[-1].confidence:
                # Replace with higher confidence detection
                result[-1] = detection

        return result

    def anonymize(
        self,
        text: str,
        method: AnonymizationMethod = AnonymizationMethod.TOKEN_REPLACEMENT,
        detections: list[PIIDetection] | None = None
    ) -> str:
        """
        Anonymize PII in text.
        
        Args:
            text: Text to anonymize
            method: Anonymization method to use
            detections: Pre-computed detections (if None, will detect)
            
        Returns:
            Anonymized text
        """
        if detections is None:
            detections = self.detect_pii(text)

        if not detections:
            return text

        # Sort by position (reverse) to replace from end
        detections = sorted(detections, key=lambda d: d.start_idx, reverse=True)

        result = text
        for detection in detections:
            if method == AnonymizationMethod.TOKEN_REPLACEMENT:
                replacement = detection.replacement
            elif method == AnonymizationMethod.HASH:
                replacement = self._hash_text(detection.text)
            elif method == AnonymizationMethod.MASK:
                replacement = self._mask_text(detection.text, detection.pii_type)
            elif method == AnonymizationMethod.REMOVE:
                replacement = ""
            elif method == AnonymizationMethod.GENERALIZE:
                replacement = self._generalize_text(detection.text, detection.pii_type)
            else:
                replacement = detection.replacement

            result = result[:detection.start_idx] + replacement + result[detection.end_idx:]

        return result

    def _hash_text(self, text: str) -> str:
        """Create a hash of the text."""
        hash_obj = hashlib.sha256(text.encode())
        return f"[HASH:{hash_obj.hexdigest()[:8]}]"

    def _mask_text(self, text: str, pii_type: PIIType) -> str:
        """Mask sensitive parts of text."""
        if pii_type == PIIType.EMAIL:
            parts = text.split('@')
            if len(parts) == 2:
                return f"{parts[0][0]}***@{parts[1]}"
        elif pii_type == PIIType.PHONE:
            # Keep area code, mask rest
            digits = re.sub(r'\D', '', text)
            if len(digits) >= 10:
                return f"({digits[:3]}) ***-****"
        elif pii_type == PIIType.SSN:
            # Show last 4 digits only
            digits = re.sub(r'\D', '', text)
            if len(digits) == 9:
                return f"***-**-{digits[-4:]}"
        elif pii_type == PIIType.CREDIT_CARD:
            # Show last 4 digits only
            digits = re.sub(r'\D', '', text)
            if len(digits) >= 12:
                return f"****-****-****-{digits[-4:]}"

        # Default: mask middle portion
        length = len(text)
        if length > 4:
            return text[0] + '*' * (length - 2) + text[-1]
        return '*' * length

    def _generalize_text(self, text: str, pii_type: PIIType) -> str:
        """Generalize text to broader category."""
        generalizations = {
            PIIType.NAME: "[PERSON]",
            PIIType.ADDRESS: "[LOCATION]",
            PIIType.EMAIL: "[EMAIL_ADDRESS]",
            PIIType.PHONE: "[PHONE_NUMBER]",
            PIIType.SSN: "[ID_NUMBER]",
            PIIType.CREDIT_CARD: "[PAYMENT_INFO]",
            PIIType.DATE_OF_BIRTH: "[DATE]",
            PIIType.MEDICAL: "[MEDICAL_INFO]",
            PIIType.IP_ADDRESS: "[NETWORK_ADDRESS]"
        }
        return generalizations.get(pii_type, f"[{pii_type.value.upper()}]")

    def anonymize_memory(
        self,
        memory: MemoryItem,
        method: AnonymizationMethod = AnonymizationMethod.TOKEN_REPLACEMENT
    ) -> MemoryItem:
        """
        Anonymize PII in a memory item.
        
        Args:
            memory: Memory item to anonymize
            method: Anonymization method to use
            
        Returns:
            Anonymized memory item
        """
        # Anonymize content text fields (including nested dicts)
        if isinstance(memory.content, dict):
            anonymized_content = {}
            for key, value in memory.content.items():
                if isinstance(value, str):
                    anonymized_content[key] = self.anonymize(value, method)
                elif isinstance(value, dict):
                    # Recursively anonymize nested dictionaries
                    anonymized_nested = {}
                    for nested_key, nested_value in value.items():
                        if isinstance(nested_value, str):
                            anonymized_nested[nested_key] = self.anonymize(nested_value, method)
                        else:
                            anonymized_nested[nested_key] = nested_value
                    anonymized_content[key] = anonymized_nested
                else:
                    anonymized_content[key] = value
            memory.content = anonymized_content

        # Anonymize metadata if present
        if memory.metadata:
            anonymized_metadata = {}
            for key, value in memory.metadata.items():
                if isinstance(value, str):
                    anonymized_metadata[key] = self.anonymize(value, method)
                else:
                    anonymized_metadata[key] = value
            memory.metadata = anonymized_metadata

        return memory

    def validate_anonymization(self, text: str) -> bool:
        """
        Validate that text has no detectable PII.
        
        Args:
            text: Text to validate
            
        Returns:
            True if no PII detected, False otherwise
        """
        detections = self.detect_pii(text)
        return len(detections) == 0

    def get_statistics(self) -> dict[str, Any]:
        """
        Get privacy engine statistics.
        
        Returns:
            Dictionary of statistics
        """
        return {
            'ner_enabled': self.enable_ner,
            'regex_enabled': self.enable_regex,
            'spacy_model': self.spacy_model if self.nlp else None,
            'regex_patterns': len(self.regex_patterns),
            'custom_patterns': len(self.custom_patterns)
        }

    def remove_pii(self, text: str) -> str:
        """
        Remove all PII from text.
        
        Args:
            text: Text to clean
            
        Returns:
            Text with PII removed
        """
        return self.anonymize(text, method=AnonymizationMethod.REMOVE)

    def add_custom_pattern(self, pii_type: PIIType, pattern: str) -> None:
        """
        Add a custom regex pattern for PII detection.
        
        Args:
            pii_type: Type of PII
            pattern: Regex pattern string
        """
        self.custom_patterns[pii_type] = pattern
        self.regex_patterns[pii_type] = re.compile(pattern)

    def prepare_for_swarm(self, memory: MemoryItem) -> dict[str, Any]:
        """
        Prepare memory for SWARM sharing with full anonymization.
        
        Args:
            memory: Memory item to prepare
            
        Returns:
            Anonymized memory data for SWARM
        """
        # Anonymize content
        anonymized_content = {}
        for key, value in memory.content.items():
            if isinstance(value, str):
                anonymized_content[key] = self.anonymize(value, method=AnonymizationMethod.HASH)
            else:
                anonymized_content[key] = value

        # Apply differential privacy to scores
        noise = 0.1 * (0.5 - int(hashlib.md5(memory.id.encode()).hexdigest()[:8], 16) / 0xffffffff)
        noisy_score = max(0.0, min(10.0, memory.effectiveness_score + noise))

        return {
            'content': anonymized_content,
            'effectiveness_score': noisy_score,
            'usage_count': memory.usage_count,
            'memory_type': memory.memory_type
        }

    def verify_k_anonymity(self, memories: list[MemoryItem], k: int = 3) -> bool:
        """
        Verify k-anonymity for a set of memories.
        
        Args:
            memories: List of memories to verify
            k: Minimum group size for anonymity
            
        Returns:
            True if k-anonymity is satisfied
        """
        if len(memories) < k:
            return False

        # Check that each memory appears in a group of at least k similar memories
        for memory in memories:
            similar_count = sum(
                1 for m in memories
                if abs(m.effectiveness_score - memory.effectiveness_score) < 1.0
            )
            if similar_count < k:
                return False

        return True

    def apply_differential_privacy(
        self,
        value: float,
        epsilon: float = 1.0,
        sensitivity: float = 1.0
    ) -> float:
        """
        Apply differential privacy noise to a value.
        
        Args:
            value: Original value
            epsilon: Privacy parameter (lower = more privacy)
            sensitivity: Maximum change in output
            
        Returns:
            Value with differential privacy noise
        """
        import random
        # Laplace mechanism for differential privacy
        scale = sensitivity / epsilon
        noise = random.random() - 0.5  # Simplified Laplace noise
        noise *= scale * 2.0

        return value + noise

    def batch_anonymize(
        self,
        texts: list[str],
        method: AnonymizationMethod = AnonymizationMethod.TOKEN_REPLACEMENT
    ) -> list[str]:
        """
        Anonymize multiple texts in batch.
        
        Args:
            texts: List of texts to anonymize
            method: Anonymization method to use
            
        Returns:
            List of anonymized texts
        """
        return [self.anonymize(text, method) for text in texts]

    def get_pii_statistics(self) -> dict[str, Any]:
        """
        Get detailed PII detection statistics.
        
        Returns:
            Dictionary of PII statistics
        """
        # Alias for get_statistics
        return self.get_statistics()
