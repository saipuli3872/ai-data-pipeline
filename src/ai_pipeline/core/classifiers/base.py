"""
Base AI Data Classifier - Abstract Foundation
==========================================

This module provides the abstract base class for all data classification
engines in our AI pipeline. It defines the standard interface that all
classifiers must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional, Dict
import pandas as pd


class DataType(Enum):
    """Standardized data type classifications."""
    IDENTIFIER = "identifier"
    BUSINESS_KEY = "business_key" 
    DATE = "date"
    NUMERIC = "numeric"
    TEXT = "text"
    BOOLEAN = "boolean"
    EMAIL = "email"
    PHONE = "phone"
    ADDRESS = "address"


class PIILevel(Enum):
    """Privacy/PII classification levels."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class ColumnProfile:
    """Comprehensive column analysis profile."""
    # Basic identification
    original_name: str
    suggested_name: str
    data_type: DataType
    
    # Key classifications
    is_primary_key: bool = False
    is_business_key: bool = False
    is_foreign_key: bool = False
    references: List[str] = field(default_factory=list)
    
    # Privacy & compliance
    pii_level: PIILevel = PIILevel.NONE
    contains_sensitive_data: bool = False
    
    # Data quality metrics
    unique_ratio: float = 0.0
    null_ratio: float = 0.0
    sample_values: List[Any] = field(default_factory=list)
    
    # AI insights
    confidence_score: float = 0.0
    business_meaning: str = ""
    data_quality_notes: str = ""
    suggested_improvements: str = ""
    
    # Metadata
    total_rows: int = 0
    unique_count: int = 0
    null_count: int = 0


@dataclass 
class ClassificationResult:
    """Results from classification analysis."""
    column_profiles: List[ColumnProfile] = field(default_factory=list)
    processing_time: float = 0.0
    ai_provider_used: Optional[str] = None
    classification_method: str = ""
    overall_confidence: float = 0.0
    
    def get_summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        if not self.column_profiles:
            return {}
            
        type_counts = {}
        pii_counts = {}
        
        for profile in self.column_profiles:
            # Count data types
            dtype = profile.data_type.value
            type_counts[dtype] = type_counts.get(dtype, 0) + 1
            
            # Count PII levels
            pii = profile.pii_level.value
            pii_counts[pii] = pii_counts.get(pii, 0) + 1
        
        return {
            "total_columns": len(self.column_profiles),
            "data_type_breakdown": type_counts,
            "pii_level_breakdown": pii_counts,
            "avg_confidence": sum(p.confidence_score for p in self.column_profiles) / len(self.column_profiles),
            "processing_time": self.processing_time,
            "classification_method": self.classification_method
        }


class BaseClassifier(ABC):
    """Abstract base class for all data classifiers."""
    
    def __init__(self, sample_size: int = 1000):
        """Initialize classifier with configuration."""
        self.sample_size = sample_size
        self.classification_stats = {
            "total_columns_analyzed": 0,
            "successful_classifications": 0,
            "failed_classifications": 0
        }
    
    @abstractmethod
    def analyze_column(self, df: pd.DataFrame, column_name: str) -> ColumnProfile:
        """Analyze a single column and return classification profile."""
        pass
    
    @abstractmethod
    def analyze_dataframe(self, df: pd.DataFrame) -> ClassificationResult:
        """Analyze entire dataframe and return complete results."""
        pass
    
    def get_sample_data(self, series: pd.Series) -> pd.Series:
        """Get representative sample from a pandas Series."""
        clean_series = series.dropna()
        if len(clean_series) == 0:
            return clean_series
            
        sample_size = min(len(clean_series), self.sample_size)
        return clean_series.sample(n=sample_size, random_state=42)
    
    def calculate_basic_stats(self, series: pd.Series) -> Dict[str, Any]:
        """Calculate basic statistics for a column."""
        total_rows = len(series)
        null_count = series.isnull().sum()
        unique_count = series.nunique()
        
        return {
            "total_rows": total_rows,
            "null_count": null_count,
            "unique_count": unique_count,
            "null_ratio": null_count / total_rows if total_rows > 0 else 0,
            "unique_ratio": unique_count / total_rows if total_rows > 0 else 0,
        }