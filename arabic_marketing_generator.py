"""
Command-line interface for Arabic Marketing Content Generator.

This module provides a CLI for using the Arabic Marketing Content Generator.
"""

import argparse
import os
import json
import pandas as pd
from typing import Dict, Any

from src.preprocessing import PreprocessingPipeline
from src.trend_detection import TrendDetectionPipeline
from src.content_generation import ContentGenerationPipeline


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Arabic Marketing Content Generator CLI',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Path to the input dataset (CSV or JSON)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./output',
        help='Directory to save output files'
    )
    
    parser.add_argument(
        '--text_column',
        type=str,
        default='text',
        help='Name of the column containing tweet text'
    )
    
    parser.add_argument(
        '--timestamp_column',
        type=str,
        default='created_at',
        help='Name of the column containing tweet timestamp'
    )
    
    parser.add_argument(
        '--num_trends',
        type=int,
        default=10,
        help='Number of top trends to detect'
    )
    
    parser.add_argument(
        '--num_captions',
        type=int,
        default=3,
        help='Number of captions to generate per trend'
    )
    
    parser.add_argument(
        '--num_hashtags',
        type=int,
        default=5,
        help='Number of hashtags to generate per trend'
    )
    
    parser.add_argument(
        '--num_ads',
        type=int,
        default=1,
        help='Number of ad scripts to generate per trend'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file (JSON)'
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a JSON file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


def run_pipeline(args):
    """Run the complete pipeline."""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configuration if provided
    config = {}
    if args.config and os.path.exists(args.config):
        config = load_config(args.config)
    
    # Set up file paths
    processed_path = os.path.join(args.output_dir, 'processed_data.csv')
    trends_path = os.path.join(args.output_dir, 'trends.json')
    content_path = os.path.join(args.output_dir, 'generated_content.json')
    
    print(f"Processing dataset: {args.dataset}")
    
    # Step 1: Preprocessing
    print("\n=== Step 1: Preprocessing ===")
    preprocessing_pipeline = PreprocessingPipeline(
        text_column=args.text_column,
        timestamp_column=args.timestamp_column
    )
    
    # Run preprocessing pipeline
    df = preprocessing_pipeline.run_pipeline(args.dataset, new_column='processed_text')
    preprocessing_pipeline.save_processed_data(processed_path)
    print(f"Processed data saved to: {processed_path}")
    
    # Step 2: Trend Detection
    print("\n=== Step 2: Trend Detection ===")
    trend_pipeline = TrendDetectionPipeline(
        min_cluster_size=config.get('min_cluster_size', 5),
        max_clusters=config.get('max_clusters', 20)
    )
    
    # Run trend detection pipeline
    df, trends = trend_pipeline.run_pipeline(
        df,
        text_column='processed_text',
        raw_text_column=args.text_column,
        n_trends=args.num_trends
    )
    
    # Save trends
    with open(trends_path, 'w', encoding='utf-8') as f:
        json.dump(trends, f, ensure_ascii=False, indent=2)
    print(f"Detected trends saved to: {trends_path}")
    
    # Display top trends
    print("\nTop trends detected:")
    for i, trend in enumerate(trends, 1):
        print(f"{i}. {trend['topic']} (Cluster size: {trend['cluster_size']})")
    
    # Step 3: Content Generation
    print("\n=== Step 3: Content Generation ===")
    content_pipeline = ContentGenerationPipeline(
        model_name=config.get('model_name', 'aubmindlab/aragpt2-medium')
    )
    
    # Generate content
    content = content_pipeline.generate_content(
        trends,
        num_captions=args.num_captions,
        num_hashtags=args.num_hashtags,
        num_ads=args.num_ads
    )
    
    # Filter content if specified in config
    if config.get('filter_content', True):
        sensitive_terms = config.get('sensitive_terms', None)
        content = content_pipeline.filter_content(sensitive_terms)
    
    # Save content
    content_pipeline.save_content(content_path)
    print(f"Generated content saved to: {content_path}")
    
    print("\nPipeline completed successfully!")
    return {
        'processed_data': processed_path,
        'trends': trends_path,
        'content': content_path
    }


def main():
    """Main entry point for the CLI."""
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
