"""
Content Generation Pipeline for Arabic Marketing Content Generator

This module combines content generation with trend detection results.
"""

import json
import os
from typing import List, Dict, Any, Optional, Union, Tuple

from .content_generator import ArabicContentGenerator


class ContentGenerationPipeline:
    """
    Pipeline for generating marketing content based on detected trends.
    """
    
    def __init__(self, model_name: str = "aubmindlab/aragpt2-medium"):
        """
        Initialize the content generation pipeline.
        
        Args:
            model_name: Name of the pretrained model to use
        """
        self.content_generator = ArabicContentGenerator(model_name)
        self.generated_content = []
    
    def generate_content(self, trends: List[Dict[str, Any]], 
                        num_captions: int = 3,
                        num_hashtags: int = 5,
                        num_ads: int = 1) -> List[Dict[str, Any]]:
        """
        Generate marketing content for detected trends.
        
        Args:
            trends: List of trend dictionaries
            num_captions: Number of captions to generate per trend
            num_hashtags: Number of hashtags to generate per trend
            num_ads: Number of ad scripts to generate per trend
            
        Returns:
            List of content dictionaries
        """
        self.generated_content = []
        
        for trend in trends:
            content = self.content_generator.generate_content_for_trend(
                trend, num_captions, num_hashtags, num_ads
            )
            self.generated_content.append(content)
        
        return self.generated_content
    
    def save_content(self, output_path: str) -> str:
        """
        Save the generated content to a JSON file.
        
        Args:
            output_path: Path to save the content
            
        Returns:
            Path to the saved file
        """
        if not self.generated_content:
            raise ValueError("No content generated. Run generate_content() first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.generated_content, f, ensure_ascii=False, indent=2)
        
        return output_path
    
    def filter_content(self, sensitive_terms: List[str] = None) -> List[Dict[str, Any]]:
        """
        Filter out content containing sensitive terms.
        
        Args:
            sensitive_terms: List of sensitive terms to filter out
            
        Returns:
            Filtered list of content dictionaries
        """
        if not self.generated_content:
            raise ValueError("No content generated. Run generate_content() first.")
        
        if not sensitive_terms:
            # Default sensitive terms in Arabic
            sensitive_terms = [
                "سياسة", "دين", "طائفة", "مذهب", "حرب", "صراع", "خلاف",
                "عنصرية", "تمييز", "إرهاب", "متطرف", "عنف"
            ]
        
        filtered_content = []
        
        for content in self.generated_content:
            # Check captions
            safe_captions = [
                caption for caption in content['captions']
                if not any(term in caption.lower() for term in sensitive_terms)
            ]
            
            # Check hashtags
            safe_hashtags = [
                hashtag for hashtag in content['hashtags']
                if not any(term in hashtag.lower() for term in sensitive_terms)
            ]
            
            # Check ad scripts
            safe_ad_scripts = [
                script for script in content['ad_scripts']
                if not any(term in script.lower() for term in sensitive_terms)
            ]
            
            # If we have at least some safe content, include it
            if safe_captions or safe_hashtags or safe_ad_scripts:
                filtered_content.append({
                    'trend': content['trend'],
                    'captions': safe_captions,
                    'hashtags': safe_hashtags,
                    'ad_scripts': safe_ad_scripts
                })
        
        return filtered_content
    
    def run_pipeline(self, trends_path: str, output_path: str) -> List[Dict[str, Any]]:
        """
        Run the complete content generation pipeline.
        
        Args:
            trends_path: Path to the JSON file containing trends
            output_path: Path to save the generated content
            
        Returns:
            List of content dictionaries
        """
        # Load trends
        with open(trends_path, 'r', encoding='utf-8') as f:
            trends = json.load(f)
        
        # Generate content
        self.generate_content(trends)
        
        # Filter content
        filtered_content = self.filter_content()
        
        # Save content
        self.save_content(output_path)
        
        return filtered_content
