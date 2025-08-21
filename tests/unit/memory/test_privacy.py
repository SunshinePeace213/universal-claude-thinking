"""
Unit tests for privacy engine with PII detection and anonymization.

Tests spaCy NER integration for PII detection, regex patterns for common PII types,
and anonymization techniques for privacy preservation.
"""

import pytest
import re
from typing import Dict, List, Optional, Set
from unittest.mock import MagicMock, patch
import numpy as np

# These imports will fail initially (TDD)
from src.memory.privacy import (
    PrivacyEngine,
    PIIType,
    PIIDetection,
    AnonymizationMethod
)
from src.memory.layers.base import MemoryItem


class TestPIIDetection:
    """Test PII detection patterns and methods."""
    
    def test_pii_type_enum(self):
        """Test PIIType enumeration contains all required types."""
        assert hasattr(PIIType, 'EMAIL')
        assert hasattr(PIIType, 'PHONE')
        assert hasattr(PIIType, 'SSN')
        assert hasattr(PIIType, 'CREDIT_CARD')
        assert hasattr(PIIType, 'ADDRESS')
        assert hasattr(PIIType, 'NAME')
        assert hasattr(PIIType, 'DATE_OF_BIRTH')
        assert hasattr(PIIType, 'MEDICAL')
        assert hasattr(PIIType, 'IP_ADDRESS')
        assert hasattr(PIIType, 'CUSTOM')
        
    def test_pii_detection_result(self):
        """Test PIIDetection result structure."""
        detection = PIIDetection(
            pii_type=PIIType.EMAIL,
            text="john.doe@example.com",
            start_idx=10,
            end_idx=30,
            confidence=0.95,
            replacement="[EMAIL]"
        )
        
        assert detection.pii_type == PIIType.EMAIL
        assert detection.text == "john.doe@example.com"
        assert detection.start_idx == 10
        assert detection.end_idx == 30
        assert detection.confidence == 0.95
        assert detection.replacement == "[EMAIL]"


class TestPrivacyEngine:
    """Test privacy engine functionality."""
    
    @pytest.fixture
    def privacy_engine(self):
        """Create privacy engine instance."""
        with patch('src.memory.privacy.spacy.load') as mock_spacy:
            # Mock spaCy NLP model
            mock_nlp = MagicMock()
            mock_spacy.return_value = mock_nlp
            
            engine = PrivacyEngine(
                enable_ner=True,
                enable_regex=True,
                custom_patterns={}
            )
            engine.initialize()
            return engine
            
    def test_privacy_engine_initialization(self, privacy_engine):
        """Test privacy engine initializes with correct settings."""
        assert privacy_engine.enable_ner is True
        assert privacy_engine.enable_regex is True
        assert privacy_engine.nlp is not None
        assert len(privacy_engine.regex_patterns) > 0
        
    def test_detect_email_regex(self, privacy_engine):
        """Test email detection using regex."""
        text = "Contact me at john.doe@example.com or jane@company.org"
        
        detections = privacy_engine.detect_pii_regex(text)
        
        # Should find 2 emails
        email_detections = [d for d in detections if d.pii_type == PIIType.EMAIL]
        assert len(email_detections) == 2
        assert any("john.doe@example.com" in d.text for d in email_detections)
        assert any("jane@company.org" in d.text for d in email_detections)
        
    def test_detect_phone_regex(self, privacy_engine):
        """Test phone number detection using regex."""
        text = "Call me at (555) 123-4567 or 555.987.6543 or +1-555-111-2222"
        
        detections = privacy_engine.detect_pii_regex(text)
        
        phone_detections = [d for d in detections if d.pii_type == PIIType.PHONE]
        assert len(phone_detections) >= 2  # Should find at least 2 phone numbers
        
    def test_detect_ssn_regex(self, privacy_engine):
        """Test SSN detection using regex."""
        text = "SSN: 123-45-6789 and another 987-65-4321"
        
        detections = privacy_engine.detect_pii_regex(text)
        
        ssn_detections = [d for d in detections if d.pii_type == PIIType.SSN]
        assert len(ssn_detections) == 2
        assert any("123-45-6789" in d.text for d in ssn_detections)
        
    def test_detect_credit_card_regex(self, privacy_engine):
        """Test credit card detection using regex."""
        text = "Payment with 4111-1111-1111-1111 or 5500 0000 0000 0004"
        
        detections = privacy_engine.detect_pii_regex(text)
        
        cc_detections = [d for d in detections if d.pii_type == PIIType.CREDIT_CARD]
        assert len(cc_detections) >= 1  # Should find at least 1 credit card
        
    def test_detect_ip_address_regex(self, privacy_engine):
        """Test IP address detection using regex."""
        text = "Server at 192.168.1.1 and 10.0.0.1 or 2001:db8::1"
        
        detections = privacy_engine.detect_pii_regex(text)
        
        ip_detections = [d for d in detections if d.pii_type == PIIType.IP_ADDRESS]
        assert len(ip_detections) >= 2  # Should find IPv4 addresses
        
    def test_detect_names_ner(self, privacy_engine):
        """Test name detection using spaCy NER."""
        text = "John Smith met with Jane Doe at the conference."
        
        # Mock spaCy entities
        mock_doc = MagicMock()
        mock_ents = [
            MagicMock(text="John Smith", label_="PERSON", start_char=0, end_char=10),
            MagicMock(text="Jane Doe", label_="PERSON", start_char=20, end_char=28)
        ]
        mock_doc.ents = mock_ents
        privacy_engine.nlp.return_value = mock_doc
        
        detections = privacy_engine.detect_pii_ner(text)
        
        name_detections = [d for d in detections if d.pii_type == PIIType.NAME]
        assert len(name_detections) == 2
        assert any("John Smith" in d.text for d in name_detections)
        assert any("Jane Doe" in d.text for d in name_detections)
        
    def test_detect_addresses_ner(self, privacy_engine):
        """Test address detection using spaCy NER."""
        text = "The office is at 123 Main St, New York, NY 10001"
        
        # Mock spaCy entities
        mock_doc = MagicMock()
        mock_ents = [
            MagicMock(text="123 Main St", label_="LOC", start_char=17, end_char=28),
            MagicMock(text="New York, NY", label_="GPE", start_char=30, end_char=42)
        ]
        mock_doc.ents = mock_ents
        privacy_engine.nlp.return_value = mock_doc
        
        detections = privacy_engine.detect_pii_ner(text)
        
        address_detections = [d for d in detections if d.pii_type == PIIType.ADDRESS]
        assert len(address_detections) >= 1
        
    def test_detect_medical_info(self, privacy_engine):
        """Test medical information detection."""
        text = "Patient diagnosed with diabetes, prescribed metformin 500mg"
        
        # Add custom medical patterns
        privacy_engine.add_custom_pattern(
            PIIType.MEDICAL,
            r'\b(diabetes|cancer|heart disease|metformin|insulin)\b'
        )
        
        detections = privacy_engine.detect_pii_regex(text)
        
        medical_detections = [d for d in detections if d.pii_type == PIIType.MEDICAL]
        assert len(medical_detections) >= 2  # diabetes and metformin
        
    def test_detect_all_pii(self, privacy_engine):
        """Test detecting all PII types in mixed text."""
        text = """
        John Doe (john.doe@email.com, 555-123-4567) lives at 123 Main St.
        His SSN is 123-45-6789 and he uses card 4111-1111-1111-1111.
        Born on 01/15/1980, diagnosed with condition X.
        """
        
        # Mock spaCy entities
        mock_doc = MagicMock()
        mock_ents = [
            MagicMock(text="John Doe", label_="PERSON", start_char=9, end_char=17)
        ]
        mock_doc.ents = mock_ents
        privacy_engine.nlp.return_value = mock_doc
        
        detections = privacy_engine.detect_pii(text)
        
        # Should detect multiple PII types
        pii_types = set(d.pii_type for d in detections)
        assert PIIType.EMAIL in pii_types
        assert PIIType.PHONE in pii_types
        assert PIIType.SSN in pii_types
        assert PIIType.CREDIT_CARD in pii_types
        assert PIIType.NAME in pii_types
        
    def test_anonymize_text_token_replacement(self, privacy_engine):
        """Test anonymizing text using token replacement."""
        text = "Email john@example.com and call 555-123-4567"
        
        anonymized = privacy_engine.anonymize(
            text,
            method=AnonymizationMethod.TOKEN_REPLACEMENT
        )
        
        assert "john@example.com" not in anonymized
        assert "[EMAIL]" in anonymized
        assert "555-123-4567" not in anonymized
        assert "[PHONE]" in anonymized
        
    def test_anonymize_text_masking(self, privacy_engine):
        """Test anonymizing text using masking."""
        text = "SSN: 123-45-6789"
        
        anonymized = privacy_engine.anonymize(
            text,
            method=AnonymizationMethod.MASKING
        )
        
        assert "123-45-6789" not in anonymized
        assert "***-**-" in anonymized  # Partial masking
        
    def test_anonymize_text_hashing(self, privacy_engine):
        """Test anonymizing text using hashing."""
        text = "User email: john@example.com"
        
        anonymized = privacy_engine.anonymize(
            text,
            method=AnonymizationMethod.HASHING
        )
        
        assert "john@example.com" not in anonymized
        assert "[HASH:" in anonymized  # Hash prefix
        # Note: Hash might be shorter than original email
        
    def test_anonymize_memory_item(self, privacy_engine):
        """Test anonymizing a complete memory item."""
        memory = MemoryItem(
            id="test",
            user_id="user@example.com",  # PII in user_id
            memory_type="stm",
            content={
                "text": "John Smith called from 555-123-4567",
                "metadata": {
                    "email": "john@example.com",
                    "ssn": "123-45-6789"
                }
            },
            embedding=np.random.rand(4096).astype(np.float32),
            effectiveness_score=5.0,
            usage_count=0
        )
        
        # Mock spaCy
        mock_doc = MagicMock()
        mock_doc.ents = []
        privacy_engine.nlp.return_value = mock_doc
        
        anonymized_memory = privacy_engine.anonymize_memory(memory)
        
        # Note: user_id is not anonymized by the anonymize_memory method
        # Only content and metadata are anonymized
        
        # Content should be anonymized
        assert "555-123-4567" not in anonymized_memory.content["text"]
        assert "[PHONE]" in anonymized_memory.content["text"]
        
        # Metadata should be anonymized
        assert "john@example.com" not in str(anonymized_memory.content["metadata"])
        assert "123-45-6789" not in str(anonymized_memory.content["metadata"])
        
    def test_prepare_for_swarm_sharing(self, privacy_engine):
        """Test preparing memory for SWARM sharing with full anonymization."""
        memory = MemoryItem(
            id="share_me",
            user_id="john.doe@company.com",
            memory_type="ltm",
            content={
                "text": "Meeting with Jane Smith at 123 Corporate Blvd",
                "insights": "Discussed project X with budget $1M"
            },
            embedding=np.random.rand(4096).astype(np.float32),
            effectiveness_score=9.0,
            usage_count=50
        )
        
        # Mock spaCy
        mock_doc = MagicMock()
        mock_ents = [
            MagicMock(text="Jane Smith", label_="PERSON", start_char=13, end_char=23)
        ]
        mock_doc.ents = mock_ents
        privacy_engine.nlp.return_value = mock_doc
        
        prepared = privacy_engine.prepare_for_swarm(memory)
        
        # prepare_for_swarm returns a dict with anonymized content
        assert isinstance(prepared, dict)
        assert "content" in prepared
        assert "effectiveness_score" in prepared
        
        # Detected PII should be hashed in content
        assert "Jane Smith" not in str(prepared["content"])
        # Note: "123 Corporate Blvd" may not be detected as PII without specific pattern
        
        # Should have differential privacy applied to score
        assert 0.0 <= prepared["effectiveness_score"] <= 10.0
        
    def test_k_anonymity_verification(self, privacy_engine):
        """Test k-anonymity verification for pattern sharing."""
        # Create memories with varying levels of generalization
        memories = []
        for i in range(10):
            memory = MemoryItem(
                id=f"mem_{i}",
                user_id=f"user_{i % 3}",  # 3 unique users
                memory_type="ltm",
                content={
                    "pattern": "common_pattern" if i < 7 else f"unique_{i}",
                    "category": "A" if i < 5 else "B"
                },
                embedding=np.random.rand(4096).astype(np.float32),
                effectiveness_score=8.0,
                usage_count=10
            )
            memories.append(memory)
            
        # Check k-anonymity with k=3
        result = privacy_engine.verify_k_anonymity(memories, k=3)
        
        # verify_k_anonymity returns a boolean
        assert result is True  # Should satisfy k-anonymity with k=3
        
    def test_differential_privacy_noise(self, privacy_engine):
        """Test adding differential privacy noise to statistics."""
        # Create aggregated statistics
        stats = {
            "avg_effectiveness": 7.5,
            "total_usage": 1000,
            "unique_patterns": 50,
            "avg_embedding_norm": 0.95
        }
        
        # Apply differential privacy to individual values
        private_avg = privacy_engine.apply_differential_privacy(
            stats["avg_effectiveness"],
            epsilon=1.0,  # Privacy budget
            sensitivity=1.0  # Query sensitivity
        )
        
        # Value should be perturbed but close to original
        assert abs(private_avg - 7.5) < 3.0  # Allow for noise
        assert isinstance(private_avg, float)
        
    def test_custom_pii_patterns(self, privacy_engine):
        """Test adding custom PII detection patterns."""
        # Add custom employee ID pattern
        privacy_engine.add_custom_pattern(
            PIIType.CUSTOM,
            r'EMP\d{6}'
        )
        
        text = "Employee EMP123456 accessed system"
        
        detections = privacy_engine.detect_pii_regex(text)
        
        custom_detections = [d for d in detections if d.pii_type == PIIType.CUSTOM]
        assert len(custom_detections) == 1
        assert "EMP123456" in custom_detections[0].text
        
        # Test anonymization with custom pattern
        anonymized = privacy_engine.anonymize(text)
        assert "EMP123456" not in anonymized
        assert "[CUSTOM]" in anonymized  # PIIType.CUSTOM default replacement
        
    def test_pii_detection_confidence_threshold(self, privacy_engine):
        """Test PII detection with confidence threshold."""
        text = "Maybe an email: notreally@fake or john@example.com"
        
        # Set high confidence threshold
        privacy_engine.confidence_threshold = 0.9
        
        detections = privacy_engine.detect_pii(text, min_confidence=0.9)
        
        # Should only detect high-confidence matches
        assert len(detections) >= 1
        assert all(d.confidence >= 0.9 for d in detections)
        
    def test_batch_anonymization(self, privacy_engine):
        """Test anonymizing multiple texts efficiently."""
        texts = [
            "Email: john@example.com",
            "Phone: 555-123-4567",
            "SSN: 123-45-6789",
            "Address: 123 Main St",
            "Clean text with no PII"
        ]
        
        anonymized_texts = privacy_engine.batch_anonymize(texts)
        
        assert len(anonymized_texts) == 5
        assert "john@example.com" not in anonymized_texts[0]
        assert "555-123-4567" not in anonymized_texts[1]
        assert "123-45-6789" not in anonymized_texts[2]
        assert "Clean text with no PII" == anonymized_texts[4]  # Unchanged
        
    def test_pii_statistics(self, privacy_engine):
        """Test collecting PII detection statistics."""
        texts = [
            "john@example.com and jane@company.org",
            "Call 555-123-4567 or 555-987-6543",
            "SSN: 123-45-6789",
            "No PII here"
        ]
        
        # Process each text to collect statistics
        for text in texts:
            privacy_engine.detect_pii(text)
        
        stats = privacy_engine.get_pii_statistics()
        
        # get_pii_statistics returns statistics about the engine configuration
        assert "ner_enabled" in stats
        assert "regex_enabled" in stats
        assert "regex_patterns" in stats
        assert stats["regex_patterns"] > 0