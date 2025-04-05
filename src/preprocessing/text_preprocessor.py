"""
Text Preprocessor Module for Arabic Marketing Content Generator

This module handles cleaning and preprocessing Arabic text data.
"""

import re
import pandas as pd
import pyarabic.araby as araby
from pyarabic.normalize import normalize_text
from farasapy.stemmer import FarasaStemmer
from typing import List, Dict, Any, Optional, Union


class ArabicTextPreprocessor:
    """
    Class for preprocessing Arabic text data from Twitter datasets.
    """
    
    def __init__(self):
        """
        Initialize the ArabicTextPreprocessor with necessary tools.
        """
        # Initialize Farasa stemmer for lemmatization
        self.stemmer = FarasaStemmer()
        
        # Regex patterns for cleaning
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F700-\U0001F77F"  # alchemical symbols
            "\U0001F780-\U0001F7FF"  # Geometric Shapes
            "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
            "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
            "\U0001FA00-\U0001FA6F"  # Chess Symbols
            "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
            "\U00002702-\U000027B0"  # Dingbats
            "\U000024C2-\U0001F251" 
            "]+"
        )
        self.non_arabic_pattern = re.compile(r'[^\u0600-\u06FF\s]')
        self.whitespace_pattern = re.compile(r'\s+')
    
    def preprocess_text(self, text: str) -> str:
        """
        Apply full preprocessing pipeline to a single text.
        
        Args:
            text: Raw Arabic text
            
        Returns:
            Preprocessed text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Normalize text (standardize Arabic characters)
        normalized_text = self.normalize_text(cleaned_text)
        
        # Tokenize and lemmatize
        lemmatized_text = self.lemmatize_text(normalized_text)
        
        return lemmatized_text
    
    def clean_text(self, text: str) -> str:
        """
        Clean text by removing URLs, emojis, and non-Arabic characters.
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Remove URLs
        text = self.url_pattern.sub(' ', text)
        
        # Remove emojis
        text = self.emoji_pattern.sub(' ', text)
        
        # Remove non-Arabic characters (keeping spaces)
        text = self.non_arabic_pattern.sub(' ', text)
        
        # Normalize whitespace
        text = self.whitespace_pattern.sub(' ', text)
        
        return text.strip()
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize Arabic text using PyArabic.
        
        Args:
            text: Cleaned text
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        # Use PyArabic's normalize_text function
        normalized = normalize_text(text)
        
        # Additional normalization with araby module
        normalized = araby.strip_tashkeel(normalized)  # Remove diacritics
        normalized = araby.strip_tatweel(normalized)   # Remove tatweel
        
        return normalized
    
    def tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize Arabic text.
        
        Args:
            text: Normalized text
            
        Returns:
            List of tokens
        """
        if not text:
            return []
        
        # Use PyArabic's tokenize function
        tokens = araby.tokenize(text)
        
        # Filter out empty tokens
        tokens = [token for token in tokens if token.strip()]
        
        return tokens
    
    def lemmatize_text(self, text: str) -> str:
        """
        Lemmatize Arabic text using Farasa.
        
        Args:
            text: Normalized text
            
        Returns:
            Lemmatized text
        """
        if not text:
            return ""
        
        # Use Farasa stemmer for lemmatization
        try:
            lemmatized = self.stemmer.stem(text)
            return lemmatized
        except Exception as e:
            # Fallback if Farasa fails
            print(f"Farasa lemmatization failed: {e}")
            return text
    
    def preprocess_dataframe(self, df: pd.DataFrame, text_column: str, 
                            new_column: Optional[str] = None) -> pd.DataFrame:
        """
        Preprocess text in a DataFrame column.
        
        Args:
            df: DataFrame containing text data
            text_column: Name of column containing text to preprocess
            new_column: Name of new column to store preprocessed text (if None, overwrites original)
            
        Returns:
            DataFrame with preprocessed text
        """
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")
        
        target_column = new_column if new_column else text_column
        
        # Apply preprocessing to each row
        df[target_column] = df[text_column].apply(self.preprocess_text)
        
        return df
    
    def handle_missing_data(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """
        Handle missing data in the text column.
        
        Args:
            df: DataFrame containing text data
            text_column: Name of column containing text
            
        Returns:
            DataFrame with handled missing values
        """
        # Replace NaN values with empty string
        df[text_column] = df[text_column].fillna("")
        
        # Remove rows with empty text after filling NaN
        df = df[df[text_column].str.strip() != ""]
        
        return df
