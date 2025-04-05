"""
Feature Extractor Module for Arabic Marketing Content Generator

This module handles extracting features from preprocessed Arabic text using AraBERT.
"""

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Any, Optional, Union, Tuple


class ArabicFeatureExtractor:
    """
    Class for extracting features from Arabic text using AraBERT.
    """
    
    def __init__(self, model_name: str = "aubmindlab/bert-base-arabertv2"):
        """
        Initialize the ArabicFeatureExtractor with AraBERT model.
        
        Args:
            model_name: Name of the pretrained model to use
        """
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Set device (use CPU as we're working with constraints)
        self.device = torch.device("cpu")
        self.model.to(self.device)
        
        # Put model in evaluation mode
        self.model.eval()
    
    def extract_embeddings(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        """
        Extract embeddings from a list of texts using AraBERT.
        
        Args:
            texts: List of preprocessed Arabic texts
            batch_size: Batch size for processing
            
        Returns:
            Array of embeddings for each text
        """
        all_embeddings = []
        
        # Process in batches to avoid memory issues
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # Use the [CLS] token embedding as the sentence embedding
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
                all_embeddings.append(batch_embeddings)
        
        # Concatenate all batches
        return np.vstack(all_embeddings)
    
    def extract_keywords(self, text: str, top_n: int = 5) -> List[str]:
        """
        Extract keywords from a single text using AraBERT token importance.
        
        Args:
            text: Preprocessed Arabic text
            top_n: Number of top keywords to extract
            
        Returns:
            List of extracted keywords
        """
        # Tokenize the text
        tokens = self.tokenizer.tokenize(text)
        
        # If no tokens, return empty list
        if not tokens:
            return []
        
        # Convert tokens to IDs and create input tensors
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor([token_ids]).to(self.device)
        
        # Get attention weights
        with torch.no_grad():
            outputs = self.model(input_ids, output_attentions=True)
            
            # Average attention weights across all layers and heads
            attentions = outputs.attentions
            avg_attention = torch.mean(torch.cat([att.mean(dim=1) for att in attentions]), dim=0)
            
            # Sum attention for each token (how much attention it receives)
            token_importance = avg_attention.sum(dim=-1).cpu().numpy()[0]
        
        # Get the original tokens (not wordpieces)
        original_tokens = []
        importance_scores = []
        
        i = 0
        while i < len(tokens):
            # Skip special tokens
            if tokens[i].startswith("##"):
                i += 1
                continue
                
            # Get the full token (including wordpieces)
            full_token = tokens[i]
            score = token_importance[i]
            
            j = i + 1
            while j < len(tokens) and tokens[j].startswith("##"):
                full_token += tokens[j][2:]  # Remove ## prefix
                score += token_importance[j]
                j += 1
            
            # Skip very short tokens and special tokens
            if len(full_token) > 2 and not full_token.startswith("["):
                original_tokens.append(full_token)
                importance_scores.append(score)
            
            i = j if j > i else i + 1
        
        # Sort tokens by importance
        sorted_indices = np.argsort(importance_scores)[::-1]
        
        # Return top N keywords
        top_keywords = [original_tokens[idx] for idx in sorted_indices[:top_n] 
                        if idx < len(original_tokens)]
        
        return top_keywords
    
    def extract_hashtags(self, text: str) -> List[str]:
        """
        Extract hashtags from text.
        
        Args:
            text: Raw Arabic text
            
        Returns:
            List of hashtags without the # symbol
        """
        import re
        hashtag_pattern = re.compile(r'#[\u0600-\u06FF0-9_]+')
        hashtags = hashtag_pattern.findall(text)
        return [tag[1:] for tag in hashtags]  # Remove # symbol
    
    def process_dataframe(self, df: pd.DataFrame, text_column: str, 
                         embedding_column: str = 'embeddings',
                         keywords_column: str = 'keywords',
                         hashtags_column: str = 'hashtags',
                         raw_text_column: Optional[str] = None) -> pd.DataFrame:
        """
        Process a DataFrame to extract embeddings, keywords, and hashtags.
        
        Args:
            df: DataFrame containing preprocessed text
            text_column: Name of column containing preprocessed text
            embedding_column: Name of column to store embeddings
            keywords_column: Name of column to store keywords
            hashtags_column: Name of column to store hashtags
            raw_text_column: Name of column containing raw text (for hashtag extraction)
            
        Returns:
            DataFrame with extracted features
        """
        # Extract embeddings
        texts = df[text_column].tolist()
        embeddings = self.extract_embeddings(texts)
        df[embedding_column] = list(embeddings)
        
        # Extract keywords
        df[keywords_column] = df[text_column].apply(self.extract_keywords)
        
        # Extract hashtags from raw text if provided
        if raw_text_column:
            df[hashtags_column] = df[raw_text_column].apply(self.extract_hashtags)
        
        return df
