"""
Configuration module for Arabic Marketing Content Generator.

This module handles loading and validating configuration settings.
"""

import json
import os
from typing import Dict, Any, Optional


class Config:
    """
    Class for handling configuration settings.
    """
    
    DEFAULT_CONFIG = {
        # Preprocessing settings
        "text_column": "text",
        "timestamp_column": "created_at",
        
        # Trend detection settings
        "min_cluster_size": 5,
        "max_clusters": 20,
        "num_trends": 10,
        
        # Content generation settings
        "model_name": "aubmindlab/aragpt2-medium",
        "num_captions": 3,
        "num_hashtags": 5,
        "num_ads": 1,
        "filter_content": True,
        "sensitive_terms": [
            "سياسة", "دين", "طائفة", "مذهب", "حرب", "صراع", "خلاف",
            "عنصرية", "تمييز", "إرهاب", "متطرف", "عنف"
        ],
        
        # Brand voice settings (for customization)
        "brand_voice": "neutral",  # Options: neutral, formal, casual, enthusiastic
        "emoji_usage": "moderate",  # Options: none, minimal, moderate, heavy
        "content_length": "medium"  # Options: short, medium, long
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration with default values and optional config file.
        
        Args:
            config_path: Path to configuration file (JSON)
        """
        self.config = self.DEFAULT_CONFIG.copy()
        
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from a JSON file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Loaded configuration dictionary
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
            
            # Update config with user values
            self.config.update(user_config)
            
            return self.config
        except Exception as e:
            print(f"Error loading configuration: {e}")
            return self.config
    
    def save_config(self, config_path: str) -> str:
        """
        Save current configuration to a JSON file.
        
        Args:
            config_path: Path to save configuration
            
        Returns:
            Path to saved configuration file
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)
        
        return config_path
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key is not found
            
        Returns:
            Configuration value
        """
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
        """
        self.config[key] = value
    
    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update multiple configuration values.
        
        Args:
            updates: Dictionary of configuration updates
        """
        self.config.update(updates)
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get the complete configuration dictionary.
        
        Returns:
            Configuration dictionary
        """
        return self.config.copy()
    
    def reset(self) -> None:
        """Reset configuration to default values."""
        self.config = self.DEFAULT_CONFIG.copy()


# Create a default configuration instance
default_config = Config()
