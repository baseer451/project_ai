"""
Test script for the trend detection pipeline using Arabic COVID dataset.

This script demonstrates the usage of the trend detection pipeline with a real Arabic COVID dataset.
"""

import os
import pandas as pd
from datasets import load_dataset
from src.preprocessing import PreprocessingPipeline
from src.trend_detection import TrendDetectionPipeline

def load_arabic_covid_dataset():
    """Load Arabic COVID dataset from Hugging Face."""
    print("Loading Arabic COVID dataset from Hugging Face...")
    
    # Load the Arabic COVID dataset
    dataset = load_dataset("msn59/arabic-covid19-twitter", split="train")
    
    # Convert to pandas DataFrame
    df = pd.DataFrame(dataset)
    
    # Create data directory if it doesn't exist
    os.makedirs('arabic_marketing_generator/data', exist_ok=True)
    
    # Save to CSV
    dataset_path = 'arabic_marketing_generator/data/arabic_covid_tweets.csv'
    df.to_csv(dataset_path, index=False, encoding='utf-8')
    
    print(f"Dataset loaded and saved to: {dataset_path}")
    print(f"Dataset shape: {df.shape}")
    print(f"Dataset columns: {df.columns.tolist()}")
    
    return df, dataset_path

def preprocess_dataset(df, text_column='text'):
    """Preprocess the dataset using the preprocessing pipeline."""
    print("\nPreprocessing the dataset...")
    
    # Initialize preprocessing pipeline
    preprocessing_pipeline = PreprocessingPipeline(text_column=text_column)
    
    # Preprocess the data
    df = preprocessing_pipeline.preprocessor.handle_missing_data(df, text_column)
    df = preprocessing_pipeline.preprocessor.preprocess_dataframe(df, text_column, 'processed_text')
    
    # Save processed data
    output_path = 'arabic_marketing_generator/data/processed_covid_tweets.csv'
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"Preprocessing completed. Processed data saved to: {output_path}")
    
    return df

def detect_trends(df, text_column='processed_text', raw_text_column='text'):
    """Detect trends in the preprocessed dataset."""
    print("\nDetecting trends in the dataset...")
    
    # Initialize trend detection pipeline
    trend_pipeline = TrendDetectionPipeline(min_cluster_size=10, max_clusters=30)
    
    # Run the pipeline
    df, trends = trend_pipeline.run_pipeline(
        df, 
        text_column=text_column,
        raw_text_column=raw_text_column,
        n_trends=10
    )
    
    # Save trends
    trends_path = 'arabic_marketing_generator/data/covid_trends.json'
    trend_pipeline.save_trends(trends_path)
    
    print(f"Trend detection completed. Trends saved to: {trends_path}")
    print("\nTop trends detected:")
    for i, trend in enumerate(trends, 1):
        print(f"{i}. Topic: {trend['topic']}")
        print(f"   Keywords: {', '.join(trend['keywords'][:5])}")
        if 'hashtags' in trend and trend['hashtags']:
            print(f"   Hashtags: {', '.join(trend['hashtags'][:3])}")
        print(f"   Cluster size: {trend['cluster_size']}")
        print()
    
    return df, trends

def main():
    """Run the complete pipeline."""
    # Load dataset
    df, _ = load_arabic_covid_dataset()
    
    # Take a subset for faster processing
    sample_size = min(5000, len(df))
    df_sample = df.sample(sample_size, random_state=42)
    print(f"\nUsing a sample of {sample_size} tweets for processing")
    
    # Preprocess dataset
    processed_df = preprocess_dataset(df_sample)
    
    # Detect trends
    clustered_df, trends = detect_trends(processed_df)
    
    print("\nPipeline completed successfully!")
    return clustered_df, trends

if __name__ == "__main__":
    main()
