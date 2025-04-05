"""
Trend Detection Pipeline for Arabic Marketing Content Generator

This module combines feature extraction and trend detection into a single pipeline.
"""

import pandas as pd
import numpy as np
import json
from typing import List, Dict, Any, Optional, Union, Tuple

from .feature_extractor import ArabicFeatureExtractor
from .trend_detector import TrendDetector


class TrendDetectionPipeline:
    """
    Pipeline for detecting trends in Arabic text data.
    """
    
    def __init__(self, min_cluster_size: int = 5, max_clusters: int = 20):
        """
        Initialize the trend detection pipeline.
        
        Args:
            min_cluster_size: Minimum number of items to consider a cluster as a trend
            max_clusters: Maximum number of clusters to try
        """
        self.feature_extractor = ArabicFeatureExtractor()
        self.trend_detector = TrendDetector(min_cluster_size, max_clusters)
        self.trends = []
    
    def extract_features(self, df: pd.DataFrame, text_column: str, 
                        raw_text_column: Optional[str] = None) -> pd.DataFrame:
        """
        Extract features from preprocessed text.
        
        Args:
            df: DataFrame containing preprocessed text
            text_column: Name of column containing preprocessed text
            raw_text_column: Name of column containing raw text (for hashtag extraction)
            
        Returns:
            DataFrame with extracted features
        """
        return self.feature_extractor.process_dataframe(
            df, text_column, raw_text_column=raw_text_column
        )
    
    def detect_trends(self, df: pd.DataFrame, 
                     embedding_column: str = 'embeddings',
                     keywords_column: str = 'keywords',
                     hashtags_column: Optional[str] = 'hashtags',
                     timestamp_column: Optional[str] = 'created_at') -> pd.DataFrame:
        """
        Detect trends in the data.
        
        Args:
            df: DataFrame with extracted features
            embedding_column: Name of column containing embeddings
            keywords_column: Name of column containing keywords
            hashtags_column: Name of column containing hashtags
            timestamp_column: Name of column containing timestamps
            
        Returns:
            DataFrame with cluster assignments
        """
        return self.trend_detector.detect_trends(
            df, embedding_column, keywords_column, hashtags_column, timestamp_column
        )
    
    def get_top_trends(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        Get the top N trends.
        
        Args:
            n: Number of top trends to return
            
        Returns:
            List of trend dictionaries
        """
        self.trends = self.trend_detector.get_top_trends(n)
        return self.trends
    
    def run_pipeline(self, df: pd.DataFrame, text_column: str, 
                    raw_text_column: Optional[str] = None,
                    timestamp_column: Optional[str] = 'created_at',
                    n_trends: int = 10) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Run the complete trend detection pipeline.
        
        Args:
            df: DataFrame containing preprocessed text
            text_column: Name of column containing preprocessed text
            raw_text_column: Name of column containing raw text
            timestamp_column: Name of column containing timestamps
            n_trends: Number of top trends to return
            
        Returns:
            Tuple of (DataFrame with cluster assignments, list of top trends)
        """
        # Extract features
        df = self.extract_features(df, text_column, raw_text_column)
        
        # Detect trends
        df = self.detect_trends(
            df, 'embeddings', 'keywords', 'hashtags', timestamp_column
        )
        
        # Get top trends
        trends = self.get_top_trends(n_trends)
        
        return df, trends
    
    def save_trends(self, output_path: str) -> str:
        """
        Save the detected trends to a JSON file.
        
        Args:
            output_path: Path to save the trends
            
        Returns:
            Path to the saved file
        """
        if not self.trends:
            raise ValueError("No trends detected. Run the pipeline first.")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.trends, f, ensure_ascii=False, indent=2)
        
        return output_path
