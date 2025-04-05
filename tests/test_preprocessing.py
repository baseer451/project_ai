"""
Test script for the preprocessing pipeline.

This script demonstrates the usage of the preprocessing pipeline with a sample dataset.
"""

import os
import pandas as pd
from src.preprocessing import DataLoader, ArabicTextPreprocessor, PreprocessingPipeline

def create_sample_data():
    """Create a sample dataset for testing."""
    # Create data directory if it doesn't exist
    os.makedirs('arabic_marketing_generator/data', exist_ok=True)
    
    # Sample Arabic tweets with various characteristics
    sample_data = {
        'id': range(1, 6),
        'text': [
            'تخفيضات رمضان تصل إلى ٥٠٪! 🛍️ #تخفيضات_رمضان #عروض_خاصة',
            'مباراة الهلال اليوم كانت رائعة! ⚽ #مباراة_الهلال https://example.com/match',
            'جربوا منتجاتنا الجديدة في رمضان 😊 @store #رمضان_كريم',
            'عروض خاصة بمناسبة شهر رمضان المبارك في متاجرنا! #رمضان #تخفيضات',
            'نتمنى لكم رمضان كريم وكل عام وأنتم بخير! 🌙 #رمضان_مبارك'
        ],
        'created_at': [
            '2025-03-01T10:00:00Z',
            '2025-03-02T15:30:00Z',
            '2025-03-03T18:45:00Z',
            '2025-03-04T09:15:00Z',
            '2025-03-05T20:20:00Z'
        ],
        'user': [
            'store1',
            'sports_fan',
            'brand_official',
            'market_place',
            'community_account'
        ]
    }
    
    # Create DataFrame
    df = pd.DataFrame(sample_data)
    
    # Save to CSV
    sample_path = 'arabic_marketing_generator/data/sample_tweets.csv'
    df.to_csv(sample_path, index=False, encoding='utf-8')
    
    return sample_path

def test_preprocessing_pipeline():
    """Test the preprocessing pipeline with sample data."""
    # Create sample data
    sample_path = create_sample_data()
    print(f"Created sample dataset at: {sample_path}")
    
    # Initialize pipeline
    pipeline = PreprocessingPipeline(text_column='text', timestamp_column='created_at')
    
    # Run pipeline
    processed_df = pipeline.run_pipeline(sample_path, new_column='processed_text')
    
    # Save processed data
    output_path = 'arabic_marketing_generator/data/processed_sample.csv'
    pipeline.save_processed_data(output_path)
    print(f"Saved processed data to: {output_path}")
    
    # Display results
    print("\nOriginal vs Processed Text:")
    for i, row in processed_df.iterrows():
        print(f"\nOriginal: {row['text']}")
        print(f"Processed: {row['processed_text']}")
    
    return processed_df

if __name__ == "__main__":
    test_preprocessing_pipeline()
