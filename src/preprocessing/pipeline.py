"""
Main preprocessing pipeline for Arabic Marketing Content Generator.

This module combines data loading and text preprocessing into a single pipeline.
"""

import pandas as pd
import os
from typing import Dict, Any, Optional, Union, List, Tuple

from .data_loader import DataLoader
from .text_preprocessor import ArabicTextPreprocessor


class PreprocessingPipeline:
    """
    Pipeline for loading and preprocessing Arabic Twitter data.
    """
    
    def __init__(self, text_column: str = 'text', timestamp_column: str = 'created_at'):
        """
        Initialize the preprocessing pipeline.
        
        Args:
            text_column: Name of the column containing tweet text
            timestamp_column: Name of the column containing tweet timestamp
        """
        self.text_column = text_column
        self.timestamp_column = timestamp_column
        self.loader = None
        self.preprocessor = ArabicTextPreprocessor()
        self.data = None
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from file.
        
        Args:
            file_path: Path to the dataset file
            
        Returns:
            DataFrame containing the loaded data
        """
        self.loader = DataLoader(file_path)
        self.data = self.loader.load()
        return self.data
    
    def validate_and_clean_data(self, required_columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Validate and clean the loaded data.
        
        Args:
            required_columns: List of required column names
            
        Returns:
            Tuple of (cleaned DataFrame, validation results)
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Set default required columns if not provided
        if required_columns is None:
            required_columns = [self.text_column]
            if self.timestamp_column:
                required_columns.append(self.timestamp_column)
        
        # Validate data
        validation_results = self.loader.validate_data(required_columns)
        
        # Handle missing data
        self.data = self.preprocessor.handle_missing_data(self.data, self.text_column)
        
        return self.data, validation_results
    
    def preprocess_text(self, new_column: str = 'processed_text') -> pd.DataFrame:
        """
        Preprocess the text column.
        
        Args:
            new_column: Name of the new column to store preprocessed text
            
        Returns:
            DataFrame with preprocessed text
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        self.data = self.preprocessor.preprocess_dataframe(
            self.data, self.text_column, new_column
        )
        
        return self.data
    
    def run_pipeline(self, file_path: str, new_column: str = 'processed_text') -> pd.DataFrame:
        """
        Run the complete preprocessing pipeline.
        
        Args:
            file_path: Path to the dataset file
            new_column: Name of the new column to store preprocessed text
            
        Returns:
            Preprocessed DataFrame
        """
        # Load data
        self.load_data(file_path)
        
        # Validate and clean data
        self.validate_and_clean_data()
        
        # Preprocess text
        self.preprocess_text(new_column)
        
        return self.data
    
    def save_processed_data(self, output_path: str, index: bool = False) -> str:
        """
        Save the processed data to a file.
        
        Args:
            output_path: Path to save the processed data
            index: Whether to include the index in the output
            
        Returns:
            Path to the saved file
        """
        if self.data is None:
            raise ValueError("Data not processed. Run the pipeline first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to CSV
        self.data.to_csv(output_path, index=index, encoding='utf-8')
        
        return output_path
