"""
Utility functions for handling Arabic text data.

This module provides helper functions for working with Arabic text.
"""

import re
import unicodedata
from typing import List, Dict, Set


def extract_hashtags(text: str) -> List[str]:
    """
    Extract hashtags from Arabic text.
    
    Args:
        text: Arabic text containing hashtags
        
    Returns:
        List of hashtags without the # symbol
    """
    # Pattern to match Arabic hashtags
    hashtag_pattern = re.compile(r'#[\u0600-\u06FF0-9_]+')
    
    # Find all hashtags
    hashtags = hashtag_pattern.findall(text)
    
    # Remove the # symbol
    hashtags = [tag[1:] for tag in hashtags]
    
    return hashtags


def extract_mentions(text: str) -> List[str]:
    """
    Extract mentions from Arabic text.
    
    Args:
        text: Arabic text containing mentions
        
    Returns:
        List of mentions without the @ symbol
    """
    # Pattern to match mentions
    mention_pattern = re.compile(r'@[\w\u0600-\u06FF]+')
    
    # Find all mentions
    mentions = mention_pattern.findall(text)
    
    # Remove the @ symbol
    mentions = [mention[1:] for mention in mentions]
    
    return mentions


def is_arabic_text(text: str, threshold: float = 0.6) -> bool:
    """
    Check if text is primarily Arabic.
    
    Args:
        text: Text to check
        threshold: Minimum ratio of Arabic characters to consider text as Arabic
        
    Returns:
        True if text is primarily Arabic, False otherwise
    """
    if not text:
        return False
    
    # Count Arabic characters
    arabic_count = sum(1 for char in text if '\u0600' <= char <= '\u06FF')
    
    # Calculate ratio of Arabic characters
    ratio = arabic_count / len(text.replace(" ", ""))
    
    return ratio >= threshold


def normalize_arabic_digits(text: str) -> str:
    """
    Convert Eastern Arabic digits to Western Arabic digits.
    
    Args:
        text: Text containing Eastern Arabic digits
        
    Returns:
        Text with normalized digits
    """
    # Mapping of Eastern Arabic digits to Western Arabic digits
    digit_map = {
        '٠': '0',
        '١': '1',
        '٢': '2',
        '٣': '3',
        '٤': '4',
        '٥': '5',
        '٦': '6',
        '٧': '7',
        '٨': '8',
        '٩': '9'
    }
    
    # Replace each Eastern Arabic digit with its Western equivalent
    for eastern, western in digit_map.items():
        text = text.replace(eastern, western)
    
    return text


def remove_arabic_diacritics(text: str) -> str:
    """
    Remove Arabic diacritics (tashkeel) from text.
    
    Args:
        text: Arabic text with diacritics
        
    Returns:
        Text without diacritics
    """
    # Arabic diacritics Unicode ranges
    diacritics = set([
        '\u064B',  # Fathatan
        '\u064C',  # Dammatan
        '\u064D',  # Kasratan
        '\u064E',  # Fatha
        '\u064F',  # Damma
        '\u0650',  # Kasra
        '\u0651',  # Shadda
        '\u0652',  # Sukun
        '\u0653',  # Maddah
        '\u0654',  # Hamza above
        '\u0655',  # Hamza below
        '\u0656',  # Subscript alef
        '\u0657',  # Inverted damma
        '\u0658',  # Mark noon ghunna
        '\u0659',  # Zwarakay
        '\u065A',  # Vowel sign small v above
        '\u065B',  # Vowel sign inverted small v above
        '\u065C',  # Vowel sign dot below
        '\u065D',  # Reversed damma
        '\u065E',  # Fatha with two dots
        '\u065F',  # Wavy hamza below
        '\u0670',  # Superscript alef
    ])
    
    # Remove diacritics
    return ''.join(c for c in text if c not in diacritics)


def get_arabic_stopwords() -> Set[str]:
    """
    Get a set of common Arabic stopwords.
    
    Returns:
        Set of Arabic stopwords
    """
    # Common Arabic stopwords
    stopwords = {
        'من', 'إلى', 'عن', 'على', 'في', 'هو', 'هي', 'هم', 'هن', 'أنا', 'نحن',
        'أنت', 'أنتم', 'أنتن', 'هذا', 'هذه', 'هؤلاء', 'ذلك', 'تلك', 'أولئك',
        'الذي', 'التي', 'الذين', 'اللاتي', 'اللواتي', 'ما', 'ماذا', 'متى',
        'أين', 'كيف', 'لماذا', 'لم', 'لن', 'لا', 'ليس', 'إن', 'كان', 'كانت',
        'كانوا', 'يكون', 'تكون', 'يكونوا', 'و', 'أو', 'ثم', 'بل', 'لكن', 'حتى',
        'إذا', 'إلا', 'قد', 'مع', 'عند', 'عندما', 'كل', 'بعض', 'غير', 'مثل',
        'بين', 'فوق', 'تحت', 'أمام', 'خلف', 'حول', 'منذ', 'خلال', 'بعد', 'قبل',
        'أول', 'آخر', 'جديد', 'قديم', 'جيد', 'سيء', 'كبير', 'صغير'
    }
    
    return stopwords
