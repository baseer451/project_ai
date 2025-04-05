"""
Arabic Marketing Content Generator

A package for ingesting Arabic Twitter datasets, detecting trending topics,
and generating culturally relevant marketing content.
"""

from .preprocessing import DataLoader, ArabicTextPreprocessor, PreprocessingPipeline
from .trend_detection import ArabicFeatureExtractor, TrendDetector, TrendDetectionPipeline
from .content_generation import ArabicContentGenerator, ContentGenerationPipeline

__version__ = "0.1.0"
__all__ = [
    'DataLoader', 
    'ArabicTextPreprocessor', 
    'PreprocessingPipeline',
    'ArabicFeatureExtractor', 
    'TrendDetector', 
    'TrendDetectionPipeline',
    'ArabicContentGenerator', 
    'ContentGenerationPipeline'
]
