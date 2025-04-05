"""
Test script for the content generation pipeline.

This script demonstrates the usage of the content generation pipeline with detected trends.
"""

import os
import json
from src.content_generation import ArabicContentGenerator, ContentGenerationPipeline

def load_or_create_sample_trends():
    """Load existing trends or create sample trends for testing."""
    trends_path = 'arabic_marketing_generator/data/covid_trends.json'
    
    # Check if trends file exists
    if os.path.exists(trends_path):
        print(f"Loading existing trends from: {trends_path}")
        with open(trends_path, 'r', encoding='utf-8') as f:
            trends = json.load(f)
    else:
        print("Creating sample trends for testing")
        # Create sample trends
        trends = [
            {
                "topic": "تخفيضات رمضان",
                "keywords": ["تخفيضات", "رمضان", "عروض", "خصومات", "تسوق"],
                "hashtags": ["تخفيضات_رمضان", "عروض_خاصة", "رمضان_كريم"],
                "cluster_size": 25
            },
            {
                "topic": "كورونا",
                "keywords": ["كورونا", "فيروس", "وباء", "صحة", "لقاح"],
                "hashtags": ["كورونا", "كوفيد_19", "الصحة_العامة"],
                "cluster_size": 42
            },
            {
                "topic": "مباراة الهلال",
                "keywords": ["الهلال", "مباراة", "كرة_القدم", "دوري", "فوز"],
                "hashtags": ["مباراة_الهلال", "الدوري_السعودي", "كرة_القدم"],
                "cluster_size": 18
            }
        ]
        
        # Save sample trends
        os.makedirs(os.path.dirname(trends_path), exist_ok=True)
        with open(trends_path, 'w', encoding='utf-8') as f:
            json.dump(trends, f, ensure_ascii=False, indent=2)
    
    print(f"Loaded {len(trends)} trends")
    return trends, trends_path

def test_content_generator():
    """Test the content generator with a single trend."""
    print("\nTesting ArabicContentGenerator with a single trend...")
    
    # Create a sample trend
    trend = {
        "topic": "تخفيضات رمضان",
        "keywords": ["تخفيضات", "رمضان", "عروض", "خصومات", "تسوق"],
        "hashtags": ["تخفيضات_رمضان", "عروض_خاصة", "رمضان_كريم"],
        "cluster_size": 25
    }
    
    # Initialize content generator
    generator = ArabicContentGenerator()
    
    # Generate content
    content = generator.generate_content_for_trend(trend, num_captions=2, num_hashtags=3, num_ads=1)
    
    # Display results
    print(f"\nGenerated content for trend: {trend['topic']}")
    print("\nCaptions:")
    for caption in content['captions']:
        print(f"- {caption}")
    
    print("\nHashtags:")
    for hashtag in content['hashtags']:
        print(f"- {hashtag}")
    
    print("\nAd Script:")
    for script in content['ad_scripts']:
        print(f"- {script}")
    
    return content

def test_content_pipeline():
    """Test the complete content generation pipeline."""
    print("\nTesting ContentGenerationPipeline...")
    
    # Load or create trends
    trends, trends_path = load_or_create_sample_trends()
    
    # Initialize pipeline
    pipeline = ContentGenerationPipeline()
    
    # Set output path
    output_path = 'arabic_marketing_generator/data/generated_content.json'
    
    # Run pipeline
    content = pipeline.generate_content(trends, num_captions=2, num_hashtags=3, num_ads=1)
    
    # Save content
    saved_path = pipeline.save_content(output_path)
    print(f"Generated content saved to: {saved_path}")
    
    # Display results
    print(f"\nGenerated content for {len(content)} trends:")
    for i, item in enumerate(content, 1):
        print(f"\n{i}. Trend: {item['trend']}")
        print("   Captions:")
        for caption in item['captions'][:1]:  # Show only first caption
            print(f"   - {caption}")
        print("   Hashtags:")
        print(f"   - {', '.join(item['hashtags'][:3])}")
    
    # Filter content
    filtered_content = pipeline.filter_content()
    print(f"\nAfter filtering, {len(filtered_content)} trends remain")
    
    return content

def main():
    """Run the content generation tests."""
    # Test content generator
    test_content_generator()
    
    # Test content pipeline
    test_content_pipeline()
    
    print("\nContent generation tests completed successfully!")

if __name__ == "__main__":
    main()
