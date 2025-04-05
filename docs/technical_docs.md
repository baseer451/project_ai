# Technical Documentation: Arabic Marketing Content Generator

## Architecture Overview

The Arabic Marketing Content Generator is built with a modular architecture consisting of three main components:

1. **Preprocessing Pipeline**: Handles data ingestion and Arabic text preprocessing
2. **Trend Detection System**: Identifies trending topics using NLP and clustering
3. **Content Generation Engine**: Creates marketing content based on detected trends

These components are designed to work together in a pipeline but can also be used independently for specific tasks.

## System Architecture Diagram

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  Preprocessing  │────▶│ Trend Detection │────▶│    Content      │
│    Pipeline     │     │     System      │     │   Generation    │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Processed Data │     │ Detected Trends │     │   Marketing     │
│                 │     │                 │     │    Content      │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## Component Details

### 1. Preprocessing Pipeline

#### Key Classes and Functions

- `DataLoader`: Loads Twitter datasets from CSV/JSON files
- `ArabicTextPreprocessor`: Cleans and normalizes Arabic text
- `PreprocessingPipeline`: Orchestrates the preprocessing workflow

#### Data Flow

1. Raw Twitter data (CSV/JSON) → `DataLoader`
2. Raw text → `ArabicTextPreprocessor` → Cleaned text
3. Preprocessed data saved to CSV for further processing

#### Arabic Text Processing

The system uses PyArabic and Farasapy for Arabic-specific text processing:

- Character normalization (e.g., combining different forms of Alef)
- Diacritics removal
- Tokenization and lemmatization
- Stopword removal
- Special character handling

### 2. Trend Detection System

#### Key Classes and Functions

- `ArabicFeatureExtractor`: Extracts embeddings and keywords using AraBERT
- `TrendDetector`: Implements K-means clustering with adaptive sizing
- `TrendDetectionPipeline`: Manages the trend detection workflow

#### Algorithm Details

1. **Feature Extraction**:
   - Text → AraBERT → Embeddings (768-dimensional vectors)
   - Text → Keyword extraction → Keywords and hashtags

2. **Clustering**:
   - Embeddings → K-means clustering → Topic clusters
   - Adaptive cluster sizing based on silhouette score
   - Optimal K selection using elbow method

3. **Trend Ranking**:
   - Clusters ranked by size and recency
   - Keywords extracted from cluster centroids
   - Hashtags aggregated from cluster members

### 3. Content Generation Engine

#### Key Classes and Functions

- `ArabicContentGenerator`: Generates content using AraGPT2
- `ContentGenerationPipeline`: Manages the content generation workflow

#### Generation Process

1. **Input Preparation**:
   - Trend topic and keywords → Prompt templates
   - Different templates for captions, hashtags, and ads

2. **Text Generation**:
   - Prompt → AraGPT2 → Raw generated text
   - Temperature and top-p sampling for creativity control

3. **Post-processing**:
   - Raw text → Cleaning and formatting → Final content
   - Content filtering for cultural sensitivity

## Technical Implementation Details

### Models and Libraries

- **AraBERT**: Arabic BERT model for embeddings and feature extraction
  - Model: `aubmindlab/bert-base-arabertv2`
  - Embedding size: 768 dimensions

- **AraGPT2**: Arabic GPT-2 model for text generation
  - Model: `aubmindlab/aragpt2-medium`
  - Parameters: 355M

- **Key Libraries**:
  - `transformers`: For working with AraBERT and AraGPT2
  - `scikit-learn`: For K-means clustering and other ML algorithms
  - `pyarabic`: For Arabic text processing
  - `farasapy`: For Arabic tokenization and lemmatization
  - `pandas`: For data manipulation
  - `streamlit`: For dashboard interface

### Performance Considerations

- **Memory Usage**:
  - AraBERT model: ~500MB
  - AraGPT2 model: ~1.5GB
  - Dataset processing: ~100MB per 10,000 tweets

- **Processing Time** (on standard hardware):
  - Preprocessing: ~5 minutes per 10,000 tweets
  - Trend detection: ~10 minutes per 10,000 tweets
  - Content generation: ~1 minute per trend

- **Optimization Techniques**:
  - Batch processing for embeddings
  - Caching of model outputs
  - Parallel processing where applicable

### Configuration System

The configuration system uses a hierarchical approach:

1. **Default configuration**: Hardcoded defaults
2. **Configuration file**: JSON file with user settings
3. **Command-line arguments**: Override specific settings

Configuration is managed by the `Config` class in `src/utils/config.py`.

## API Reference

### Preprocessing Module

```python
from arabic_marketing_generator.preprocessing import PreprocessingPipeline

# Initialize pipeline
pipeline = PreprocessingPipeline(text_column='text', timestamp_column='created_at')

# Process dataset
processed_df = pipeline.run_pipeline('tweets.csv', new_column='processed_text')

# Save processed data
pipeline.save_processed_data('processed_data.csv')
```

### Trend Detection Module

```python
from arabic_marketing_generator.trend_detection import TrendDetectionPipeline

# Initialize pipeline
pipeline = TrendDetectionPipeline(min_cluster_size=5, max_clusters=20)

# Detect trends
df, trends = pipeline.run_pipeline(
    df,
    text_column='processed_text',
    raw_text_column='text',
    n_trends=10
)

# Get top trends
top_trends = pipeline.get_top_trends(5)

# Save trends
pipeline.save_trends('trends.json')
```

### Content Generation Module

```python
from arabic_marketing_generator.content_generation import ContentGenerationPipeline

# Initialize pipeline
pipeline = ContentGenerationPipeline(model_name='aubmindlab/aragpt2-medium')

# Generate content
content = pipeline.generate_content(
    trends,
    num_captions=3,
    num_hashtags=5,
    num_ads=1
)

# Filter content
filtered_content = pipeline.filter_content()

# Save content
pipeline.save_content('generated_content.json')
```

## Extension Points

The system is designed to be extensible in several ways:

### Custom Preprocessing

Extend `ArabicTextPreprocessor` to implement custom preprocessing steps:

```python
from arabic_marketing_generator.preprocessing import ArabicTextPreprocessor

class CustomPreprocessor(ArabicTextPreprocessor):
    def preprocess_text(self, text):
        # Call parent method
        text = super().preprocess_text(text)
        
        # Add custom preprocessing
        # ...
        
        return text
```

### Alternative Models

Replace AraBERT or AraGPT2 with alternative models:

```python
from arabic_marketing_generator.trend_detection import ArabicFeatureExtractor

# Use alternative model
extractor = ArabicFeatureExtractor(model_name='alternative/arabic-model')
```

### Custom Content Filters

Add custom content filtering logic:

```python
from arabic_marketing_generator.content_generation import ContentGenerationPipeline

# Initialize pipeline
pipeline = ContentGenerationPipeline()

# Generate content
content = pipeline.generate_content(trends)

# Custom filtering
def custom_filter(content_list, criteria):
    # Implement custom filtering logic
    # ...
    return filtered_content

filtered_content = custom_filter(content, my_criteria)
```

## Testing

### Test Coverage

The system includes tests for all major components:

- Unit tests for individual classes and functions
- Integration tests for component interactions
- Validation tests for overall system performance

### Running Tests

```bash
# Run all tests
python -m unittest discover

# Run specific test module
python -m unittest tests.test_preprocessing

# Run validation tests
python tests/validate_solution.py
```

## Deployment

### Production Deployment Considerations

1. **Model Serving**:
   - Consider using model quantization for reduced size
   - Use a dedicated model server for production

2. **Scaling**:
   - Implement batch processing for large datasets
   - Consider distributed processing for very large datasets

3. **Monitoring**:
   - Log model inputs and outputs
   - Monitor performance metrics
   - Implement alerting for failures

### Docker Deployment

A Dockerfile is provided for containerized deployment:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY . .
RUN pip install -e .

EXPOSE 8501

CMD ["streamlit", "run", "dashboard/app.py"]
```

Build and run:

```bash
docker build -t arabic-marketing-generator .
docker run -p 8501:8501 arabic-marketing-generator
```

## Security Considerations

1. **Data Privacy**:
   - No user data is stored or transmitted
   - All processing happens locally

2. **Content Safety**:
   - Content filtering system prevents generation of sensitive content
   - Configurable sensitive terms list

3. **Model Security**:
   - Models are downloaded from trusted sources (Hugging Face)
   - No external API calls for core functionality

## Performance Benchmarks

| Operation | Dataset Size | Processing Time | Memory Usage |
|-----------|--------------|-----------------|--------------|
| Preprocessing | 10,000 tweets | ~5 minutes | ~100MB |
| Trend Detection | 10,000 tweets | ~10 minutes | ~600MB |
| Content Generation | 10 trends | ~1 minute | ~1.5GB |

*Note: Benchmarks performed on a system with 16GB RAM and 4-core CPU*

## Known Limitations

1. **Language Support**:
   - Optimized for Modern Standard Arabic (MSA)
   - Limited support for dialectal variations
   - No support for other languages

2. **Content Quality**:
   - Generated content requires human review
   - Quality varies based on input data quality
   - Limited understanding of brand-specific voice

3. **Resource Requirements**:
   - High memory usage for large datasets
   - GPU recommended for faster processing

## Future Development

Planned enhancements for future versions:

1. **Multilingual Support**:
   - Add support for other Arabic dialects
   - Implement code-switching detection

2. **Enhanced Generation**:
   - Fine-tuning options for specific industries
   - More customization for brand voice
   - Image suggestion capabilities

3. **Performance Improvements**:
   - Model quantization for reduced size
   - Streaming processing for large datasets
   - GPU acceleration
