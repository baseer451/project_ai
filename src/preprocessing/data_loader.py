"""
Data Loader Module for Arabic Marketing Content Generator

This module handles loading Twitter datasets in CSV/JSON format.
"""

import pandas as pd
import json
import os
from typing import Dict, Any, Optional, Union


class DataLoader:
    """
    Class for loading Twitter datasets in various formats.
    """
    
    def __init__(self, file_path: str):
        """
        Initialize the DataLoader with a file path.
        
        Args:
            file_path: Path to the dataset file (CSV or JSON)
        """
        self.file_path = file_path
        self.data = None
        self.file_extension = os.path.splitext(file_path)[1].lower()
    
    def load(self) -> pd.DataFrame:
        """
        Load the dataset based on file extension.
        
        Returns:
            DataFrame containing the loaded data
        
        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file does not exist
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
        
        if self.file_extension == '.csv':
            self.data = self._load_csv()
        elif self.file_extension == '.json':
            self.data = self._load_json()
        else:
            raise ValueError(f"Unsupported file format: {self.file_extension}. "
                             f"Supported formats are: .csv, .json")
        
        return self.data
    
    def _load_csv(self) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Returns:
            DataFrame containing the loaded data
        """
        try:
            # Try to detect encoding automatically
            return pd.read_csv(self.file_path)
        except UnicodeDecodeError:
            # If automatic detection fails, try with utf-8
            return pd.read_csv(self.file_path, encoding='utf-8')
    
    def _load_json(self) -> pd.DataFrame:
        """
        Load data from JSON file.
        
        Returns:
            DataFrame containing the loaded data
        """
        with open(self.file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            if 'data' in data:
                return pd.DataFrame(data['data'])
            else:
                # Flatten the dictionary if it's not in a standard format
                return pd.DataFrame([data])
        else:
            raise ValueError("Unsupported JSON structure")
    
    def validate_data(self, required_columns: Optional[list] = None) -> Dict[str, Any]:
        """
        Validate that the loaded data contains the required columns.
        
        Args:
            required_columns: List of column names that must be present
        
        Returns:
            Dictionary with validation results
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load() first.")
        
        result = {
            "row_count": len(self.data),
            "column_count": len(self.data.columns),
            "columns": list(self.data.columns),
            "missing_columns": [],
            "is_valid": True
        }
        
        if required_columns:
            missing = [col for col in required_columns if col not in self.data.columns]
            result["missing_columns"] = missing
            result["is_valid"] = len(missing) == 0
        
        return result
    
    def get_sample(self, n: int = 5) -> pd.DataFrame:
        """
        Get a sample of the loaded data.
        
        Args:
            n: Number of rows to sample
        
        Returns:
            DataFrame containing the sampled data
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load() first.")
        
        return self.data.head(n)
