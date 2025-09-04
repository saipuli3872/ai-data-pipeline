"""
Unit tests for data classifiers.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from ai_pipeline.core.classifiers.base import DataType, PIILevel, ColumnProfile
from ai_pipeline.core.classifiers.pattern_classifier import PatternClassifier
from ai_pipeline.core.classifiers.ai_classifier import MultiAIClassifier
from ai_pipeline.core.classifiers.hybrid_classifier import HybridClassifier


class TestPatternClassifier:
    """Test cases for PatternClassifier."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.classifier = PatternClassifier(sample_size=100)
    
    def test_email_detection(self, sample_customer_data):
        """Test email pattern detection."""
        profile = self.classifier.analyze_column(sample_customer_data, 'email')
        
        assert profile.data_type == DataType.EMAIL
        assert profile.pii_level == PIILevel.HIGH
        assert not profile.is_primary_key
        assert profile.confidence_score > 0.7
    
    def test_phone_detection(self, sample_customer_data):
        """Test phone number pattern detection.""" 
        profile = self.classifier.analyze_column(sample_customer_data, 'phone')
        
        assert profile.data_type == DataType.PHONE
        assert profile.pii_level == PIILevel.HIGH
        assert profile.confidence_score > 0.7
    
    def test_identifier_detection(self, sample_customer_data):
        """Test identifier detection."""
        profile = self.classifier.analyze_column(sample_customer_data, 'customer_id')
        
        assert profile.data_type == DataType.IDENTIFIER
        assert profile.is_primary_key == True
        assert profile.unique_ratio > 0.9
    
    def test_boolean_detection(self, sample_customer_data):
        """Test boolean pattern detection."""
        profile = self.classifier.analyze_column(sample_customer_data, 'is_premium')
        
        assert profile.data_type == DataType.BOOLEAN
        assert not profile.is_primary_key
        assert profile.pii_level == PIILevel.NONE
    
    def test_date_detection(self, sample_customer_data):
        """Test date pattern detection."""
        profile = self.classifier.analyze_column(sample_customer_data, 'signup_date')
        
        assert profile.data_type == DataType.DATE
        assert profile.pii_level == PIILevel.NONE
    
    def test_business_key_detection(self, sample_customer_data):
        """Test business key detection."""
        profile = self.classifier.analyze_column(sample_customer_data, 'status')
        
        assert profile.is_business_key == True
        assert profile.unique_ratio < 0.5  # Low cardinality
    
    def test_numeric_detection(self, sample_customer_data):
        """Test numeric pattern detection."""
        profile = self.classifier.analyze_column(sample_customer_data, 'age')
        
        assert profile.data_type == DataType.NUMERIC
        assert not profile.is_primary_key
    
    def test_dataframe_analysis(self, sample_customer_data):
        """Test complete dataframe analysis."""
        result = self.classifier.analyze_dataframe(sample_customer_data)
        
        assert len(result.column_profiles) == len(sample_customer_data.columns)
        assert result.processing_time > 0
        assert result.classification_method == "pattern_based"
        assert result.overall_confidence > 0.5
    
    def test_messy_data_handling(self, sample_messy_data):
        """Test handling of messy data."""
        result = self.classifier.analyze_dataframe(sample_messy_data)
        
        # Should handle all columns without crashing
        assert len(result.column_profiles) == len(sample_messy_data.columns)
        
        # Find specific classifications
        profiles_by_name = {p.original_name: p for p in result.column_profiles}
        
        assert profiles_by_name['ID_FIELD'].data_type == DataType.IDENTIFIER
        assert profiles_by_name['uuid_field'].data_type == DataType.IDENTIFIER
        assert profiles_by_name['category_codes'].is_business_key == True
    
    def test_empty_dataframe(self, empty_dataframe):
        """Test handling of empty dataframe."""
        result = self.classifier.analyze_dataframe(empty_dataframe)
        
        assert len(result.column_profiles) == 0
        assert result.processing_time >= 0
    
    def test_suggested_name_generation(self):
        """Test suggested name standardization."""
        test_cases = [
            ('Customer ID', 'customer_id'),
            ('first-name', 'first_name'),
            ('EmailAddress', 'email_address'),
            ('phone_number', 'phone_number')
        ]
        
        for original, expected in test_cases:
            suggested = self.classifier._generate_suggested_name(original)
            assert suggested == expected


class TestMultiAIClassifier:
    """Test cases for MultiAIClassifier."""
    
    def setup_method(self):
        """Setup test fixtures."""
        # Initialize without real API keys for testing
        self.classifier = MultiAIClassifier(
            sample_size=100,
            anthropic_key=None,
            google_key=None, 
            openai_key=None
        )
    
    def test_initialization_without_keys(self):
        """Test initialization without API keys."""
        classifier = MultiAIClassifier()
        
        assert not classifier.ai_enabled
        assert len(classifier.clients) == 0
    
    def test_initialization_with_mock_keys(self):
        """Test initialization with API keys."""
        with patch.dict(os.environ, {
            'ANTHROPIC_API_KEY': 'test_key',
            'GOOGLE_AI_API_KEY': 'test_key',
            'OPENAI_API_KEY': 'test_key'
        }):
            classifier = MultiAIClassifier()
            assert len(classifier.clients) > 0
    
    @patch('ai_pipeline.core.classifiers.ai_classifier.anthropic.Anthropic')
    def test_claude_api_call(self, mock_anthropic, sample_customer_data):
        """Test Claude API integration."""
        # Mock Claude response
        mock_response = Mock()
        mock_response.content = [Mock(text='{"data_type": "email", "is_primary_key": false, "is_business_key": false, "pii_level": "high", "confidence_score": 0.9, "business_meaning": "Customer email address"}')]
        
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        # Initialize classifier with mock
        classifier = MultiAIClassifier(anthropic_key='test_key')
        classifier.clients['claude'] = mock_client
        classifier.ai_enabled = True
        
        # Test analysis
        profile = classifier.analyze_column(sample_customer_data, 'email')
        
        assert profile.data_type == DataType.EMAIL
        assert profile.pii_level == PIILevel.HIGH
        assert profile.confidence_score == 0.9
    
    def test_fallback_when_ai_fails(self, sample_customer_data):
        """Test fallback behavior when AI fails."""
        profile = self.classifier.analyze_column(sample_customer_data, 'email')
        
        # Should create basic profile when AI is unavailable
        assert profile is not None
        assert profile.confidence_score == 0.5  # Default fallback confidence
        assert profile.business_meaning != ""
    
    def test_json_response_parsing(self):
        """Test AI response parsing with various formats."""
        test_responses = [
            '{"data_type": "email", "is_primary_key": false, "is_business_key": false, "pii_level": "high", "confidence_score": 0.9}',
            '```json\n{"data_type": "text", "is_primary_key": false, "is_business_key": true, "pii_level": "none", "confidence_score": 0.8}\n```',
            '```\n{"data_type": "identifier", "is_primary_key": true, "is_business_key": false, "pii_level": "none", "confidence_score": 0.95}\n```'
        ]
        
        for response_text in test_responses:
            result = self.classifier._parse_ai_response(response_text, 'test_provider')
            assert result is not None
            assert 'data_type' in result
            assert 'confidence_score' in result
    
    def test_invalid_json_handling(self):
        """Test handling of invalid JSON responses."""
        invalid_responses = [
            'This is not JSON',
            '{"incomplete": "json"',
            '{"data_type": "invalid_type", "confidence_score": "not_a_number"}'
        ]
        
        for response_text in invalid_responses:
            result = self.classifier._parse_ai_response(response_text, 'test_provider')
            # Should handle gracefully (return None or corrected result)
            if result is not None:
                assert 'data_type' in result


class TestHybridClassifier:
    """Test cases for HybridClassifier."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.classifier = HybridClassifier(sample_size=100, enable_ai=False)  # Disable AI for testing
    
    def test_pattern_only_mode(self, sample_customer_data):
        """Test hybrid classifier in pattern-only mode."""
        result = self.classifier.analyze_dataframe(sample_customer_data)
        
        assert len(result.column_profiles) == len(sample_customer_data.columns)
        assert result.classification_method == "hybrid_pattern_only"
        assert not self.classifier.ai_enabled
    
    def test_consensus_agreement(self, sample_customer_data):
        """Test behavior when pattern and AI agree."""
        # Mock AI classifier to return same results as pattern
        mock_ai = Mock()
        mock_ai_profile = ColumnProfile(
            original_name='email',
            suggested_name='email',
            data_type=DataType.EMAIL,
            pii_level=PIILevel.HIGH,
            confidence_score=0.9
        )
        mock_ai.analyze_column.return_value = mock_ai_profile
        
        self.classifier.ai_classifier = mock_ai
        self.classifier.ai_enabled = True
        
        profile = self.classifier.analyze_column(sample_customer_data, 'email')
        
        assert profile.data_type == DataType.EMAIL
        assert profile.confidence_score > 0.8  # Should be enhanced due to consensus
    
    def test_ai_override_scenario(self, sample_customer_data):
        """Test AI override when AI has high confidence."""
        # Mock AI classifier with high confidence different result
        mock_ai = Mock()
        mock_ai_profile = ColumnProfile(
            original_name='customer_id',
            suggested_name='customer_id', 
            data_type=DataType.BUSINESS_KEY,  # Different from pattern's IDENTIFIER
            confidence_score=0.95,  # High confidence
            business_meaning='Business identifier'
        )
        mock_ai.analyze_column.return_value = mock_ai_profile
        
        self.classifier.ai_classifier = mock_ai
        self.classifier.ai_enabled = True
        self.classifier.ai_confidence_threshold = 0.8
        
        profile = self.classifier.analyze_column(sample_customer_data, 'customer_id')
        
        assert profile.data_type == DataType.BUSINESS_KEY  # Should use AI classification
        assert "AI override" in profile.data_quality_notes
    
    def test_hybrid_statistics(self, sample_customer_data):
        """Test hybrid classification statistics tracking."""
        self.classifier.analyze_dataframe(sample_customer_data)
        
        stats = self.classifier.get_hybrid_stats()
        
        assert 'total_processed' in stats
        assert 'ai_enabled' in stats
        assert 'breakdown' in stats
        assert stats['total_processed'] == len(sample_customer_data.columns)
    
    def test_confidence_thresholds(self):
        """Test configurable confidence thresholds."""
        # Test different threshold configurations
        classifier_low = HybridClassifier(ai_confidence_threshold=0.6, enable_ai=False)
        classifier_high = HybridClassifier(ai_confidence_threshold=0.9, enable_ai=False)
        
        assert classifier_low.ai_confidence_threshold == 0.6
        assert classifier_high.ai_confidence_threshold == 0.9
    
    def test_ai_enable_disable(self):
        """Test dynamic AI enabling/disabling."""
        classifier = HybridClassifier(enable_ai=False)
        
        # Should be disabled initially
        assert not classifier.ai_enabled
        
        # Trying to enable without AI classifier should fail
        result = classifier.set_ai_enabled(True)
        assert not result
        assert not classifier.ai_enabled


# Integration tests
class TestClassifierIntegration:
    """Integration tests across all classifiers."""
    
    def test_all_classifiers_same_input(self, sample_customer_data):
        """Test all classifiers on same dataset for consistency."""
        pattern_classifier = PatternClassifier()
        hybrid_classifier = HybridClassifier(enable_ai=False)
        
        pattern_result = pattern_classifier.analyze_dataframe(sample_customer_data)
        hybrid_result = hybrid_classifier.analyze_dataframe(sample_customer_data)
        
        # Results should be similar for pattern-only mode
        assert len(pattern_result.column_profiles) == len(hybrid_result.column_profiles)
        
        # Compare specific columns
        pattern_profiles = {p.original_name: p for p in pattern_result.column_profiles}
        hybrid_profiles = {p.original_name: p for p in hybrid_result.column_profiles}
        
        for col in sample_customer_data.columns:
            assert pattern_profiles[col].data_type == hybrid_profiles[col].data_type
    
    def test_performance_benchmarks(self, sample_customer_data):
        """Test performance benchmarks for classifiers."""
        classifiers = {
            'pattern': PatternClassifier(),
            'hybrid_no_ai': HybridClassifier(enable_ai=False)
        }
        
        for name, classifier in classifiers.items():
            result = classifier.analyze_dataframe(sample_customer_data)
            
            # Performance assertions
            assert result.processing_time < 10.0  # Should complete within 10 seconds
            assert len(result.column_profiles) == len(sample_customer_data.columns)
            
            print(f"{name} classifier: {result.processing_time:.3f}s for {len(sample_customer_data.columns)} columns")