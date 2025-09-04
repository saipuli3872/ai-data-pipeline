"""
Pattern-Based Data Classifier
============================

Fast, deterministic data classification using regex patterns and statistical analysis.
This provides the reliable foundation for our hybrid AI approach.
"""

import re
import time
from typing import Dict, List, Pattern
import pandas as pd
from loguru import logger

from .base import BaseClassifier, ColumnProfile, ClassificationResult, DataType, PIILevel


class PatternClassifier(BaseClassifier):
    """Pattern-based data classifier using regex and statistical analysis."""
    
    def __init__(self, sample_size: int = 1000):
        """Initialize with pattern definitions."""
        super().__init__(sample_size)
        self.patterns = self._compile_patterns()
        
    def _compile_patterns(self) -> Dict[DataType, List[Pattern]]:
        """Compile regex patterns for each data type."""
        return {
            DataType.EMAIL: [
                re.compile(r'^[\w.+-]+@[\w-]+\.[\w.-]+$', re.IGNORECASE),
                re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', re.IGNORECASE)
            ],
            DataType.PHONE: [
                re.compile(r'^\+?1?\d{9,15}$'),
                re.compile(r'^\(?([0-9]{3})\)?[-. ]?([0-9]{3})[-. ]?([0-9]{4})$'),
                re.compile(r'^\+?[\d\s\-\(\)\.]{10,18}$')
            ],
            DataType.DATE: [
                re.compile(r'^\d{4}-\d{2}-\d{2}$'),  # YYYY-MM-DD
                re.compile(r'^\d{2}[/-]\d{2}[/-]\d{4}$'),  # MM/DD/YYYY or DD/MM/YYYY
                re.compile(r'^\d{4}/\d{2}/\d{2}$'),  # YYYY/MM/DD
                re.compile(r'^\w+\s+\d{1,2},\s+\d{4}$')  # Month DD, YYYY
            ],
            DataType.NUMERIC: [
                re.compile(r'^-?\d+$'),  # Integers
                re.compile(r'^-?\d*\.\d+$'),  # Decimals
                re.compile(r'^-?\d{1,3}(,\d{3})*(\.\d+)?$'),  # Numbers with commas
                re.compile(r'^\$?\d+(\.\d{2})?$')  # Currency
            ],
            DataType.BOOLEAN: [
                re.compile(r'^(true|false)$', re.IGNORECASE),
                re.compile(r'^(yes|no)$', re.IGNORECASE),
                re.compile(r'^(y|n)$', re.IGNORECASE),
                re.compile(r'^(1|0)$'),
                re.compile(r'^(on|off)$', re.IGNORECASE)
            ],
            DataType.IDENTIFIER: [
                re.compile(r'^[A-Z0-9]{8,}$'),  # Long alphanumeric codes
                re.compile(r'^\d{6,}$'),  # Long numeric IDs
                re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.IGNORECASE)  # UUIDs
            ]
        }
    
    def analyze_column(self, df: pd.DataFrame, column_name: str) -> ColumnProfile:
        """Analyze a single column using pattern matching."""
        logger.debug(f"Pattern analyzing column: {column_name}")
        
        series = df[column_name]
        stats = self.calculate_basic_stats(series)
        sample = self.get_sample_data(series)
        
        # Convert to string for pattern matching
        sample_str = sample.astype(str)
        
        # Detect data type using patterns
        data_type = self._detect_data_type(sample_str, column_name, stats)
        
        # Determine key classifications
        is_primary_key = self._is_primary_key(column_name, stats, data_type)
        is_business_key = self._is_business_key(column_name, sample_str, stats, data_type)
        
        # Detect PII level
        pii_level = self._detect_pii_level(column_name, sample_str, data_type)
        
        # Generate suggested name
        suggested_name = self._generate_suggested_name(column_name)
        
        # Get sample values for inspection
        sample_values = sample.head(5).tolist()
        
        profile = ColumnProfile(
            original_name=column_name,
            suggested_name=suggested_name,
            data_type=data_type,
            is_primary_key=is_primary_key,
            is_business_key=is_business_key,
            pii_level=pii_level,
            unique_ratio=stats["unique_ratio"],
            null_ratio=stats["null_ratio"],
            sample_values=sample_values,
            confidence_score=0.8,  # Pattern matching is generally reliable
            business_meaning=self._generate_business_meaning(column_name, data_type),
            total_rows=stats["total_rows"],
            unique_count=stats["unique_count"],
            null_count=stats["null_count"]
        )
        
        return profile
    
    def analyze_dataframe(self, df: pd.DataFrame) -> ClassificationResult:
        """Analyze entire dataframe using pattern matching."""
        start_time = time.time()
        logger.info(f"Starting pattern analysis of {len(df.columns)} columns")
        
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
        
        result = ClassificationResult(
            column_profiles=profiles,
            processing_time=processing_time,
            ai_provider_used=None,
            classification_method="pattern_based",
            overall_confidence=0.8
        )
        
        logger.info(f"Pattern analysis complete in {processing_time:.2f}s")
        return result
    
    def _detect_data_type(self, sample: pd.Series, column_name: str, stats: Dict) -> DataType:
        """Detect data type using pattern matching and heuristics."""
        if len(sample) == 0:
            return DataType.TEXT
        
        # Check specific patterns first (most specific to least specific)
        for data_type, patterns in self.patterns.items():
            if data_type == DataType.IDENTIFIER:
                continue  # Handle identifiers separately
                
            for pattern in patterns:
                matches = sample.str.match(pattern, na=False)
                match_ratio = matches.sum() / len(sample)
                
                if match_ratio >= 0.8:  # 80% of samples match
                    return data_type
        
        # Special handling for identifiers
        if self._is_likely_identifier(column_name, stats):
            return DataType.IDENTIFIER
        
        # Fallback heuristics
        if stats["unique_ratio"] > 0.95 and "id" in column_name.lower():
            return DataType.IDENTIFIER
        elif stats["unique_ratio"] < 0.1 and len(sample) > 10:
            return DataType.BUSINESS_KEY
        else:
            return DataType.TEXT
    
    def _is_likely_identifier(self, column_name: str, stats: Dict) -> bool:
        """Check if column is likely an identifier."""
        name_lower = column_name.lower()
        
        # Name-based detection
        if any(keyword in name_lower for keyword in ["id", "key", "uuid", "guid"]):
            if stats["unique_ratio"] > 0.9:
                return True
        
        return False
    
    def _is_primary_key(self, column_name: str, stats: Dict, data_type: DataType) -> bool:
        """Determine if column is likely a primary key."""
        name_lower = column_name.lower()
        
        # High uniqueness + ID-like name
        if stats["unique_ratio"] >= 0.99 and any(keyword in name_lower for keyword in ["id", "key"]):
            return True
        
        # Perfect uniqueness + identifier type
        if stats["unique_ratio"] == 1.0 and data_type == DataType.IDENTIFIER:
            return True
            
        return False
    
    def _is_business_key(self, column_name: str, sample: pd.Series, stats: Dict, data_type: DataType) -> bool:
        """Determine if column is a business key."""
        name_lower = column_name.lower()
        
        # Business key patterns in name
        business_patterns = ["code", "type", "status", "category", "class", "group", "dept", "region"]
        if any(pattern in name_lower for pattern in business_patterns):
            return True
        
        # Low cardinality with meaningful values
        if data_type == DataType.TEXT and stats["unique_ratio"] < 0.3 and len(sample) > 10:
            # Check if values look like business codes
            code_patterns = sample.str.match(r'^[A-Z]{2,4}$|^[A-Z][0-9]{1,3}$', na=False)
            if code_patterns.sum() / len(sample) > 0.5:
                return True
        
        return False
    
    def _detect_pii_level(self, column_name: str, sample: pd.Series, data_type: DataType) -> PIILevel:
        """Detect PII level based on data type and content."""
        name_lower = column_name.lower()
        
        # High PII data types
        if data_type == DataType.EMAIL:
            return PIILevel.HIGH
        elif data_type == DataType.PHONE:
            return PIILevel.HIGH
        
        # Check for sensitive names
        high_pii_names = ["ssn", "social", "credit", "password", "secret"]
        if any(keyword in name_lower for keyword in high_pii_names):
            return PIILevel.HIGH
        
        medium_pii_names = ["name", "address", "location", "birth"]
        if any(keyword in name_lower for keyword in medium_pii_names):
            return PIILevel.MEDIUM
        
        low_pii_names = ["first", "last", "city", "state", "zip"]
        if any(keyword in name_lower for keyword in low_pii_names):
            return PIILevel.LOW
        
        return PIILevel.NONE
    
    def _generate_suggested_name(self, column_name: str) -> str:
        """Generate a standardized column name."""
        # Convert to snake_case
        name = re.sub(r'[^0-9a-zA-Z]+', '_', column_name)
        name = re.sub(r'([a-z])([A-Z])', r'\1_\2', name)  # camelCase to snake_case
        return name.lower().strip('_')
    
    def _generate_business_meaning(self, column_name: str, data_type: DataType) -> str:
        """Generate basic business meaning from column name and type."""
        name_words = re.sub(r'[_\-]', ' ', column_name).lower().split()
        
        meaning_templates = {
            DataType.IDENTIFIER: f"Unique identifier for {' '.join(name_words)}",
            DataType.BUSINESS_KEY: f"Business classification code for {' '.join(name_words)}",
            DataType.EMAIL: f"Email address field for {' '.join(name_words)}",
            DataType.PHONE: f"Phone number for {' '.join(name_words)}",
            DataType.DATE: f"Date/timestamp for {' '.join(name_words)}",
            DataType.NUMERIC: f"Numeric measurement of {' '.join(name_words)}",
            DataType.BOOLEAN: f"Boolean flag indicating {' '.join(name_words)}",
            DataType.TEXT: f"Text description of {' '.join(name_words)}"
        }
        
        return meaning_templates.get(data_type, f"Data field for {' '.join(name_words)}")