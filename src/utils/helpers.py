"""
Utility functions for Arabic Marketing Content Generator.

This module provides utility functions used across the package.
"""

import os
import json
from typing import Dict, Any, List, Optional

def ensure_dir(directory: str) -> str:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Directory path
        
    Returns:
        Directory path
    """
    os.makedirs(directory, exist_ok=True)
    return directory

def save_json(data: Any, file_path: str, ensure_ascii: bool = False) -> str:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        file_path: Path to save the data
        ensure_ascii: Whether to ensure ASCII characters only
        
    Returns:
        Path to the saved file
    """
    # Create directory if it doesn't exist
    ensure_dir(os.path.dirname(file_path))
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=ensure_ascii, indent=2)
    
    return file_path

def load_json(file_path: str) -> Any:
    """
    Load data from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Loaded data
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data

def validate_file_path(file_path: str, extensions: List[str] = None) -> bool:
    """
    Validate that a file exists and has the correct extension.
    
    Args:
        file_path: Path to the file
        extensions: List of valid extensions (e.g., ['.csv', '.json'])
        
    Returns:
        True if file is valid, False otherwise
    """
    if not os.path.exists(file_path):
        return False
    
    if extensions:
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in extensions:
            return False
    
    return True

def get_file_extension(file_path: str) -> str:
    """
    Get the extension of a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File extension (e.g., '.csv')
    """
    return os.path.splitext(file_path)[1].lower()

def format_arabic_text(text: str, add_emoji: bool = False) -> str:
    """
    Format Arabic text for display.
    
    Args:
        text: Arabic text
        add_emoji: Whether to add emoji based on content
        
    Returns:
        Formatted text
    """
    # Add basic formatting
    formatted = text.strip()
    
    # Add emoji if requested
    if add_emoji:
        # Simple emoji mapping based on keywords
        emoji_mapping = {
            'رمضان': '🌙',
            'عيد': '🎉',
            'تخفيضات': '🛍️',
            'عروض': '💰',
            'خصومات': '💯',
            'جديد': '✨',
            'تسوق': '🛒',
            'هدية': '🎁',
            'مجاني': '🆓',
            'صحة': '💪',
            'طعام': '🍽️',
            'سفر': '✈️',
            'رياضة': '⚽'
        }
        
        for keyword, emoji in emoji_mapping.items():
            if keyword in text.lower():
                formatted += f" {emoji}"
                break
    
    return formatted

def is_sensitive_content(text: str, sensitive_terms: List[str]) -> bool:
    """
    Check if text contains sensitive content.
    
    Args:
        text: Text to check
        sensitive_terms: List of sensitive terms
        
    Returns:
        True if text contains sensitive content, False otherwise
    """
    return any(term in text.lower() for term in sensitive_terms)
