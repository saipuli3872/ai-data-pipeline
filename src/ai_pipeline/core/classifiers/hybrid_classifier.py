"""
Hybrid AI Data Classifier
========================

Intelligent hybrid system combining fast pattern recognition with AI enhancement.
This is our core competitive advantage - reliable, fast, and intelligent.
"""

import time
import os
from typing import Dict, List, Optional, Tuple
import pandas as pd
from loguru import logger

from .base import BaseClassifier, ColumnProfile, ClassificationResult, DataType, PIILevel
from .pattern_classifier import PatternClassifier
from .ai_classifier import MultiAIClassifier


class HybridClassifier(BaseClassifier):
    """
    Hybrid classifier that combines pattern-based speed with AI intelligence.
    
    Strategy:
    1. Run fast pattern classification first
    2. Use AI to validate and enhance results
    3. Intelligent merging based on confidence scores
    4. Graceful fallback when AI is unavailable
    """
    
    def __init__(self, sample_size: int = 1000, 
                 enable_ai: bool = True,
                 ai_confidence_threshold: float = 0.8,
                 pattern_confidence_threshold: float = 0.7):
        """Initialize hybrid classifier with both engines."""
        super().__init__(sample_size)
        
        # Initialize pattern classifier (always available)
        self.pattern_classifier = PatternClassifier(sample_size)
        logger.info("Pattern classifier initialized")
        
        # Initialize AI classifier (if enabled and keys available)
        self.ai_classifier = None
        self.ai_enabled = False
        
        if enable_ai:
            self.ai_classifier = self._initialize_ai_classifier()
            self.ai_enabled = self.ai_classifier is not None
        
        # Configuration thresholds
        self.ai_confidence_threshold = ai_confidence_threshold
        self.pattern_confidence_threshold = pattern_confidence_threshold
        
        # Performance tracking
        self.hybrid_stats = {
            "pattern_only": 0,
            "ai_enhanced": 0,
            "ai_override": 0,
            "consensus_agreement": 0,
            "consensus_disagreement": 0
        }
        
        logger.info(f"Hybrid classifier initialized - AI enabled: {self.ai_enabled}")
    
    def _initialize_ai_classifier(self) -> Optional[MultiAIClassifier]:
        """Initialize AI classifier with available API keys."""
        try:
            # Try to get API keys from environment
            anthropic_key = os.getenv('ANTHROPIC_API_KEY')
            google_key = os.getenv('GOOGLE_AI_API_KEY')  
            openai_key = os.getenv('OPENAI_API_KEY')
            
            # Only initialize if we have at least one API key
            if any([anthropic_key, google_key, openai_key]):
                return MultiAIClassifier(
                    sample_size=self.sample_size,
                    anthropic_key=anthropic_key,
                    google_key=google_key,
                    openai_key=openai_key
                )
            else:
                logger.warning("No AI API keys found - running in pattern-only mode")
                return None
                
        except Exception as e:
            logger.error(f"Failed to initialize AI classifier: {e}")
            return None
    
    def analyze_column(self, df: pd.DataFrame, column_name: str) -> ColumnProfile:
        """Analyze column using hybrid approach."""
        logger.debug(f"Hybrid analyzing column: {column_name}")
        
        # Step 1: Always run pattern analysis (fast and reliable)
        pattern_profile = self.pattern_classifier.analyze_column(df, column_name)
        logger.debug(f"Pattern analysis complete for {column_name}: {pattern_profile.data_type.value}")
        
        # Step 2: If AI is available, get AI analysis
        ai_profile = None
        if self.ai_enabled:
            try:
                ai_profile = self.ai_classifier.analyze_column(df, column_name)
                logger.debug(f"AI analysis complete for {column_name}: {ai_profile.data_type.value}")
            except Exception as e:
                logger.warning(f"AI analysis failed for {column_name}: {e}")
                ai_profile = None
        
        # Step 3: Merge results intelligently
        final_profile = self._merge_classifications(pattern_profile, ai_profile, column_name)
        
        return final_profile
    
    def analyze_dataframe(self, df: pd.DataFrame) -> ClassificationResult:
        """Analyze entire dataframe using hybrid approach."""
        start_time = time.time()
        logger.info(f"Starting hybrid analysis of {len(df.columns)} columns")
        
        profiles = []
        
        for column_name in df.columns:
            try:
                profile = self.analyze_column(df, column_name)
                profiles.append(profile)
                self.classification_stats["successful_classifications"] += 1
            except Exception as e:
                logger.error(f"Failed to analyze column {column_name}: {e}")
                self.classification_stats["failed_classifications"] += 1
        
        processing_time = time.time() - start_time
        self.classification_stats["total_columns_analyzed"] += len(df.columns)
        
        # Calculate overall confidence and method used
        avg_confidence = sum(p.confidence_score for p in profiles) / len(profiles) if profiles else 0
        method = "hybrid_ai" if self.ai_enabled else "hybrid_pattern_only"
        
        result = ClassificationResult(
            column_profiles=profiles,
            processing_time=processing_time,
            ai_provider_used="hybrid" if self.ai_enabled else None,
            classification_method=method,
            overall_confidence=avg_confidence
        )
        
        # Log performance summary
        self._log_performance_summary(processing_time, len(profiles))
        
        return result
    
    def _merge_classifications(self, pattern_profile: ColumnProfile, 
                             ai_profile: Optional[ColumnProfile], 
                             column_name: str) -> ColumnProfile:
        """Intelligently merge pattern and AI classifications."""
        
        # If no AI analysis, return enhanced pattern result
        if ai_profile is None:
            self.hybrid_stats["pattern_only"] += 1
            pattern_profile.business_meaning = pattern_profile.business_meaning or f"Pattern-classified {pattern_profile.data_type.value}"
            pattern_profile.data_quality_notes = "Pattern-based classification only"
            return pattern_profile
        
        # Compare classifications
        pattern_type = pattern_profile.data_type
        ai_type = ai_profile.data_type
        
        logger.debug(f"Merging {column_name}: Pattern={pattern_type.value}, AI={ai_type.value}")
        
        # Case 1: Perfect agreement - high confidence result
        if pattern_type == ai_type:
            self.hybrid_stats["consensus_agreement"] += 1
            merged = self._create_consensus_profile(pattern_profile, ai_profile)
            merged.confidence_score = min(0.95, (pattern_profile.confidence_score + ai_profile.confidence_score) / 2 + 0.1)
            merged.data_quality_notes = "Pattern and AI consensus"
            return merged
        
        # Case 2: Disagreement - need to decide
        self.hybrid_stats["consensus_disagreement"] += 1
        
        # High-confidence AI overrides pattern
        if ai_profile.confidence_score >= self.ai_confidence_threshold:
            self.hybrid_stats["ai_override"] += 1
            logger.debug(f"AI override for {column_name}: high AI confidence ({ai_profile.confidence_score})")
            merged = self._enhance_with_ai(pattern_profile, ai_profile)
            merged.data_quality_notes = f"AI override (confidence: {ai_profile.confidence_score:.2f})"
            return merged
        
        # High-confidence pattern wins
        if pattern_profile.confidence_score >= self.pattern_confidence_threshold:
            self.hybrid_stats["ai_enhanced"] += 1
            logger.debug(f"Pattern wins for {column_name}: high pattern confidence ({pattern_profile.confidence_score})")
            merged = self._enhance_with_ai(pattern_profile, ai_profile, use_ai_classification=False)
            merged.data_quality_notes = f"Pattern classification, AI enhanced"
            return merged
        
        # Case 3: Both medium confidence - use AI insights but pattern classification
        self.hybrid_stats["ai_enhanced"] += 1
        merged = self._enhance_with_ai(pattern_profile, ai_profile, use_ai_classification=False)
        merged.confidence_score = (pattern_profile.confidence_score + ai_profile.confidence_score) / 2
        merged.data_quality_notes = f"Hybrid result: pattern type, AI insights"
        
        return merged
    
    def _create_consensus_profile(self, pattern_profile: ColumnProfile, ai_profile: ColumnProfile) -> ColumnProfile:
        """Create enhanced profile when pattern and AI agree."""
        # Start with pattern profile and enhance with AI insights
        consensus_profile = ColumnProfile(
            original_name=pattern_profile.original_name,
            suggested_name=pattern_profile.suggested_name,
            data_type=pattern_profile.data_type,  # They agree on this
            is_primary_key=pattern_profile.is_primary_key or ai_profile.is_primary_key,  # OR logic for keys
            is_business_key=pattern_profile.is_business_key or ai_profile.is_business_key,
            pii_level=max(pattern_profile.pii_level, ai_profile.pii_level, key=lambda x: x.value),  # Higher PII level
            unique_ratio=pattern_profile.unique_ratio,
            null_ratio=pattern_profile.null_ratio,
            sample_values=pattern_profile.sample_values,
            business_meaning=ai_profile.business_meaning or pattern_profile.business_meaning,
            suggested_improvements=ai_profile.suggested_improvements,
            total_rows=pattern_profile.total_rows,
            unique_count=pattern_profile.unique_count,
            null_count=pattern_profile.null_count
        )
        
        return consensus_profile
    
    def _enhance_with_ai(self, pattern_profile: ColumnProfile, ai_profile: ColumnProfile, 
                        use_ai_classification: bool = True) -> ColumnProfile:
        """Enhance pattern profile with AI insights."""
        enhanced_profile = ColumnProfile(
            original_name=pattern_profile.original_name,
            suggested_name=pattern_profile.suggested_name,
            data_type=ai_profile.data_type if use_ai_classification else pattern_profile.data_type,
            is_primary_key=pattern_profile.is_primary_key or ai_profile.is_primary_key,
            is_business_key=pattern_profile.is_business_key or ai_profile.is_business_key,
            pii_level=max(pattern_profile.pii_level, ai_profile.pii_level, key=lambda x: x.value),
            unique_ratio=pattern_profile.unique_ratio,
            null_ratio=pattern_profile.null_ratio,
            sample_values=pattern_profile.sample_values,
            confidence_score=ai_profile.confidence_score if use_ai_classification else pattern_profile.confidence_score,
            business_meaning=ai_profile.business_meaning or pattern_profile.business_meaning,
            data_quality_notes=ai_profile.data_quality_notes,
            suggested_improvements=ai_profile.suggested_improvements,
            total_rows=pattern_profile.total_rows,
            unique_count=pattern_profile.unique_count,
            null_count=pattern_profile.null_count
        )
        
        return enhanced_profile
    
    def _log_performance_summary(self, processing_time: float, total_columns: int):
        """Log hybrid classifier performance summary."""
        logger.info(f"Hybrid analysis complete in {processing_time:.2f}s for {total_columns} columns")
        
        if total_columns > 0:
            logger.info(f"Classification breakdown:")
            logger.info(f"  Pattern only: {self.hybrid_stats['pattern_only']} ({self.hybrid_stats['pattern_only']/total_columns*100:.1f}%)")
            logger.info(f"  AI enhanced: {self.hybrid_stats['ai_enhanced']} ({self.hybrid_stats['ai_enhanced']/total_columns*100:.1f}%)")
            logger.info(f"  AI override: {self.hybrid_stats['ai_override']} ({self.hybrid_stats['ai_override']/total_columns*100:.1f}%)")
            logger.info(f"  Consensus agreement: {self.hybrid_stats['consensus_agreement']} ({self.hybrid_stats['consensus_agreement']/total_columns*100:.1f}%)")
    
    def get_hybrid_stats(self) -> Dict[str, any]:
        """Get detailed hybrid classifier statistics."""
        total_processed = sum(self.hybrid_stats.values())
        
        stats = {
            "total_processed": total_processed,
            "ai_enabled": self.ai_enabled,
            "breakdown": self.hybrid_stats.copy()
        }
        
        if total_processed > 0:
            stats["percentages"] = {
                key: round(value / total_processed * 100, 1) 
                for key, value in self.hybrid_stats.items()
            }
        
        return stats
    
    def set_ai_enabled(self, enabled: bool):
        """Enable or disable AI processing."""
        if enabled and self.ai_classifier is None:
            logger.warning("Cannot enable AI - no AI classifier available")
            return False
        
        self.ai_enabled = enabled
        logger.info(f"AI processing {'enabled' if enabled else 'disabled'}")
        return True