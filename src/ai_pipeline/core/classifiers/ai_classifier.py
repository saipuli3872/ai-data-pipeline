"""
Multi-AI Data Classifier
========================

AI-powered data classification using multiple providers (Claude, Gemini, OpenAI)
with intelligent fallback and consensus mechanisms.
"""

import json
import time
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import asdict
import pandas as pd
from loguru import logger

# AI Provider imports
import anthropic
import google.generativeai as genai
import openai
from openai import OpenAI

from .base import BaseClassifier, ColumnProfile, ClassificationResult, DataType, PIILevel


class MultiAIClassifier(BaseClassifier):
    """Multi-provider AI classifier with intelligent consensus."""
    
    def __init__(self, sample_size: int = 1000, 
                 anthropic_key: Optional[str] = None,
                 google_key: Optional[str] = None,
                 openai_key: Optional[str] = None):
        """Initialize with API keys for multiple providers."""
        super().__init__(sample_size)
        
        # Initialize AI clients
        self.clients = {}
        
        if anthropic_key:
            self.clients['claude'] = anthropic.Anthropic(api_key=anthropic_key)
            logger.info("Claude AI client initialized")
        
        if google_key:
            genai.configure(api_key=google_key)
            self.clients['gemini'] = genai.GenerativeModel("gemini-pro")
            logger.info("Gemini AI client initialized")
        
        if openai_key:
            self.clients['openai'] = OpenAI(api_key=openai_key)
            logger.info("OpenAI client initialized")
        
        if not self.clients:
            logger.warning("No AI providers initialized - running in pattern-only mode")
        
        self.provider_priority = ['claude', 'gemini', 'openai']
        
    def analyze_column(self, df: pd.DataFrame, column_name: str) -> ColumnProfile:
        """Analyze a single column using AI providers."""
        logger.debug(f"AI analyzing column: {column_name}")
        
        series = df[column_name]
        stats = self.calculate_basic_stats(series)
        sample = self.get_sample_data(series)
        
        # Get AI analysis
        ai_result = self._get_ai_analysis(column_name, sample, stats)
        
        if ai_result:
            profile = self._create_profile_from_ai(column_name, ai_result, stats, sample)
            profile.confidence_score = ai_result.get('confidence_score', 0.7)
        else:
            # Fallback to basic heuristic classification
            profile = self._create_basic_profile(column_name, stats, sample)
            profile.confidence_score = 0.5
        
        return profile
    
    def analyze_dataframe(self, df: pd.DataFrame) -> ClassificationResult:
        """Analyze entire dataframe using AI providers."""
        start_time = time.time()
        logger.info(f"Starting AI analysis of {len(df.columns)} columns")
        
        profiles = []
        successful_ai_calls = 0
        
        for column_name in df.columns:
            try:
                profile = self.analyze_column(df, column_name)
                profiles.append(profile)
                
                if profile.confidence_score > 0.6:
                    successful_ai_calls += 1
                
                self.classification_stats["successful_classifications"] += 1
            except Exception as e:
                logger.error(f"Failed to analyze column {column_name}: {e}")
                self.classification_stats["failed_classifications"] += 1
        
        processing_time = time.time() - start_time
        self.classification_stats["total_columns_analyzed"] += len(df.columns)
        
        # Determine which AI provider was most successful
        ai_provider = self._get_most_successful_provider()
        
        result = ClassificationResult(
            column_profiles=profiles,
            processing_time=processing_time,
            ai_provider_used=ai_provider,
            classification_method="ai_powered",
            overall_confidence=sum(p.confidence_score for p in profiles) / len(profiles) if profiles else 0
        )
        
        logger.info(f"AI analysis complete in {processing_time:.2f}s, {successful_ai_calls}/{len(profiles)} successful AI calls")
        return result
    
    def _get_ai_analysis(self, column_name: str, sample: pd.Series, stats: Dict) -> Optional[Dict]:
        """Get AI analysis from available providers with fallback."""
        sample_values = sample.head(10).tolist()
        
        prompt = self._create_analysis_prompt(column_name, sample_values, stats)
        
        # Try providers in priority order
        for provider_name in self.provider_priority:
            if provider_name in self.clients:
                try:
                    result = self._call_ai_provider(provider_name, prompt)
                    if result:
                        logger.debug(f"Successful AI analysis from {provider_name} for {column_name}")
                        return result
                except Exception as e:
                    logger.warning(f"AI provider {provider_name} failed for {column_name}: {e}")
                    continue
        
        logger.warning(f"All AI providers failed for {column_name}")
        return None
    
    def _create_analysis_prompt(self, column_name: str, sample_values: List, stats: Dict) -> str:
        """Create a comprehensive prompt for AI analysis."""
        return f"""Analyze this database column and provide classification in JSON format.

Column Name: {column_name}
Sample Values: {sample_values[:5]}
Total Rows: {stats['total_rows']}
Unique Values: {stats['unique_count']}
Null Values: {stats['null_count']}
Unique Ratio: {stats['unique_ratio']:.3f}

Classify this column and return ONLY a valid JSON object with these exact keys:
{{
    "data_type": "one of: identifier, business_key, date, numeric, text, boolean, email, phone, address",
    "is_primary_key": boolean,
    "is_business_key": boolean,
    "pii_level": "one of: none, low, medium, high",
    "confidence_score": float between 0.0 and 1.0,
    "business_meaning": "Brief description of what this column represents",
    "data_quality_notes": "Any data quality observations",
    "suggested_improvements": "Recommendations for data quality improvement"
}}

Consider:
- Column name patterns and business context
- Sample value patterns and formats
- Statistical properties (uniqueness, nulls)
- Privacy implications (PII classification)
- Business meaning and purpose

Return only the JSON object, no additional text."""
    
    def _call_ai_provider(self, provider_name: str, prompt: str) -> Optional[Dict]:
        """Call specific AI provider and parse response."""
        try:
            if provider_name == 'claude':
                return self._call_claude(prompt)
            elif provider_name == 'gemini':
                return self._call_gemini(prompt)
            elif provider_name == 'openai':
                return self._call_openai(prompt)
        except Exception as e:
            logger.error(f"Error calling {provider_name}: {e}")
            return None
    
    def _call_claude(self, prompt: str) -> Optional[Dict]:
        """Call Claude AI API."""
        response = self.clients['claude'].messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = response.content[0].text.strip()
        return self._parse_ai_response(response_text, 'claude')
    
    def _call_gemini(self, prompt: str) -> Optional[Dict]:
        """Call Gemini AI API."""
        response = self.clients['gemini'].generate_content(prompt)
        response_text = response.text.strip()
        return self._parse_ai_response(response_text, 'gemini')
    
    def _call_openai(self, prompt: str) -> Optional[Dict]:
        """Call OpenAI API."""
        response = self.clients['openai'].chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.1
        )
        
        response_text = response.choices[0].message.content.strip()
        return self._parse_ai_response(response_text, 'openai')
    
    def _parse_ai_response(self, response_text: str, provider: str) -> Optional[Dict]:
        """Parse AI response and validate JSON structure."""
        try:
            # Clean response text (remove markdown code blocks if present)
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
            
            # Parse JSON
            result = json.loads(response_text.strip())