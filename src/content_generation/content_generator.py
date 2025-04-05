"""
Content Generator Module for Arabic Marketing Content Generator

This module handles generating marketing content based on detected trends using AraGPT2.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import numpy as np
import re
from typing import List, Dict, Any, Optional, Union, Tuple


class ArabicContentGenerator:
    """
    Class for generating Arabic marketing content using AraGPT2.
    """
    
    def __init__(self, model_name: str = "aubmindlab/aragpt2-medium"):
        """
        Initialize the ArabicContentGenerator with AraGPT2 model.
        
        Args:
            model_name: Name of the pretrained model to use
        """
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Set device (use CPU as we're working with constraints)
        self.device = torch.device("cpu")
        self.model.to(self.device)
        
        # Put model in evaluation mode
        self.model.eval()
        
        # Marketing-specific prompts
        self.caption_prompts = [
            "إعلان: {topic}. ",
            "عرض خاص: {topic}. ",
            "لا تفوت فرصة {topic}. ",
            "اكتشف {topic} اليوم. ",
            "جرب {topic} الآن. "
        ]
        
        self.hashtag_prompts = [
            "الهاشتاغات الشائعة لـ {topic}: ",
            "هاشتاغات {topic}: ",
            "استخدم هذه الهاشتاغات: {topic} "
        ]
        
        self.ad_prompts = [
            "إعلان لـ {topic}:\n",
            "نص إعلاني عن {topic}:\n",
            "محتوى تسويقي لـ {topic}:\n"
        ]
    
    def generate_text(self, prompt: str, max_length: int = 100, 
                     num_return_sequences: int = 1, temperature: float = 0.8,
                     top_p: float = 0.9, top_k: int = 50) -> List[str]:
        """
        Generate text based on a prompt.
        
        Args:
            prompt: Text prompt to generate from
            max_length: Maximum length of generated text
            num_return_sequences: Number of sequences to generate
            temperature: Sampling temperature (higher = more creative)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            
        Returns:
            List of generated texts
        """
        # Encode the prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Generate text
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode the generated text
        generated_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output]
        
        # Remove the prompt from the generated text
        generated_texts = [text[len(prompt):].strip() for text in generated_texts]
        
        return generated_texts
    
    def generate_caption(self, topic: str, keywords: List[str] = None, 
                        max_length: int = 50, num_captions: int = 3) -> List[str]:
        """
        Generate social media captions for a topic.
        
        Args:
            topic: Topic to generate captions for
            keywords: List of keywords to incorporate
            max_length: Maximum length of generated captions
            num_captions: Number of captions to generate
            
        Returns:
            List of generated captions
        """
        # Select a random prompt template
        prompt_template = np.random.choice(self.caption_prompts)
        
        # Fill in the topic
        prompt = prompt_template.format(topic=topic)
        
        # Add keywords if provided
        if keywords and len(keywords) > 0:
            keyword_str = " ".join(keywords[:3])  # Use up to 3 keywords
            prompt += f" {keyword_str} "
        
        # Generate captions
        captions = self.generate_text(
            prompt, 
            max_length=max_length, 
            num_return_sequences=num_captions,
            temperature=0.8  # Slightly higher temperature for creativity
        )
        
        # Clean up captions
        captions = [self._clean_caption(caption) for caption in captions]
        
        return captions
    
    def generate_hashtags(self, topic: str, existing_hashtags: List[str] = None, 
                         num_hashtags: int = 5) -> List[str]:
        """
        Generate hashtags for a topic.
        
        Args:
            topic: Topic to generate hashtags for
            existing_hashtags: List of existing hashtags to avoid duplicates
            num_hashtags: Number of hashtags to generate
            
        Returns:
            List of generated hashtags
        """
        # Select a random prompt template
        prompt_template = np.random.choice(self.hashtag_prompts)
        
        # Fill in the topic
        prompt = prompt_template.format(topic=topic)
        
        # Add existing hashtags if provided
        if existing_hashtags and len(existing_hashtags) > 0:
            hashtag_str = " ".join([f"#{tag}" for tag in existing_hashtags[:2]])
            prompt += f" {hashtag_str} "
        
        # Generate hashtag text
        hashtag_text = self.generate_text(
            prompt, 
            max_length=50,  # Shorter length for hashtags
            num_return_sequences=1,
            temperature=0.7  # Lower temperature for more focused hashtags
        )[0]
        
        # Extract hashtags
        hashtags = self._extract_hashtags(hashtag_text)
        
        # If no hashtags were extracted, create some from the topic
        if not hashtags:
            topic_words = topic.split()
            hashtags = [f"{topic.replace(' ', '_')}", f"{topic_words[0] if topic_words else topic}"]
        
        # Add # symbol if not present
        hashtags = [tag if tag.startswith('#') else f"#{tag}" for tag in hashtags]
        
        # Remove duplicates and limit to requested number
        hashtags = list(dict.fromkeys(hashtags))[:num_hashtags]
        
        return hashtags
    
    def generate_ad_script(self, topic: str, keywords: List[str] = None,
                          max_length: int = 150, num_ads: int = 1) -> List[str]:
        """
        Generate ad scripts for a topic.
        
        Args:
            topic: Topic to generate ad scripts for
            keywords: List of keywords to incorporate
            max_length: Maximum length of generated ad scripts
            num_ads: Number of ad scripts to generate
            
        Returns:
            List of generated ad scripts
        """
        # Select a random prompt template
        prompt_template = np.random.choice(self.ad_prompts)
        
        # Fill in the topic
        prompt = prompt_template.format(topic=topic)
        
        # Add keywords if provided
        if keywords and len(keywords) > 0:
            keyword_str = " ".join(keywords[:3])  # Use up to 3 keywords
            prompt += f" {keyword_str} "
        
        # Generate ad scripts
        ad_scripts = self.generate_text(
            prompt, 
            max_length=max_length, 
            num_return_sequences=num_ads,
            temperature=0.8  # Slightly higher temperature for creativity
        )
        
        # Clean up ad scripts
        ad_scripts = [self._clean_ad_script(script) for script in ad_scripts]
        
        return ad_scripts
    
    def _clean_caption(self, caption: str) -> str:
        """
        Clean up a generated caption.
        
        Args:
            caption: Raw generated caption
            
        Returns:
            Cleaned caption
        """
        # Remove extra whitespace
        caption = re.sub(r'\s+', ' ', caption).strip()
        
        # Ensure it ends with a punctuation mark
        if caption and not re.search(r'[.!?]$', caption):
            caption += '.'
        
        return caption
    
    def _clean_ad_script(self, script: str) -> str:
        """
        Clean up a generated ad script.
        
        Args:
            script: Raw generated ad script
            
        Returns:
            Cleaned ad script
        """
        # Remove extra whitespace
        script = re.sub(r'\s+', ' ', script).strip()
        
        # Ensure it ends with a punctuation mark
        if script and not re.search(r'[.!?]$', script):
            script += '.'
        
        return script
    
    def _extract_hashtags(self, text: str) -> List[str]:
        """
        Extract hashtags from text.
        
        Args:
            text: Text containing hashtags
            
        Returns:
            List of hashtags
        """
        # Pattern to match Arabic hashtags
        hashtag_pattern = re.compile(r'#[\u0600-\u06FF0-9_]+')
        
        # Find all hashtags
        hashtags = hashtag_pattern.findall(text)
        
        # If no hashtags with # symbol, try to extract words
        if not hashtags:
            # Extract Arabic words that might be hashtags
            word_pattern = re.compile(r'[\u0600-\u06FF0-9_]+')
            words = word_pattern.findall(text)
            hashtags = [word for word in words if len(word) > 2]
        
        return hashtags
    
    def generate_content_for_trend(self, trend: Dict[str, Any], 
                                 num_captions: int = 3,
                                 num_hashtags: int = 5,
                                 num_ads: int = 1) -> Dict[str, Any]:
        """
        Generate complete marketing content for a trend.
        
        Args:
            trend: Trend dictionary with topic and keywords
            num_captions: Number of captions to generate
            num_hashtags: Number of hashtags to generate
            num_ads: Number of ad scripts to generate
            
        Returns:
            Dictionary with generated content
        """
        topic = trend['topic']
        keywords = trend.get('keywords', [])
        existing_hashtags = trend.get('hashtags', [])
        
        # Generate content
        captions = self.generate_caption(topic, keywords, num_captions=num_captions)
        hashtags = self.generate_hashtags(topic, existing_hashtags, num_hashtags=num_hashtags)
        ad_scripts = self.generate_ad_script(topic, keywords, num_ads=num_ads)
        
        # Create content dictionary
        content = {
            'trend': topic,
            'captions': captions,
            'hashtags': hashtags,
            'ad_scripts': ad_scripts
        }
        
        return content
    
    def generate_content_for_trends(self, trends: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate marketing content for multiple trends.
        
        Args:
            trends: List of trend dictionaries
            
        Returns:
            List of content dictionaries
        """
        content_list = []
        
        for trend in trends:
            content = self.generate_content_for_trend(trend)
            content_list.append(content)
        
        return content_list
