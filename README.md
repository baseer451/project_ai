# Arabic Marketing Content Generator

## Overview

The Arabic Marketing Content Generator is an autonomous system that processes Arabic Twitter data, detects trending topics, and generates culturally relevant marketing content. This tool helps marketers create engaging social media content that resonates with Arabic-speaking audiences by leveraging natural language processing and machine learning techniques.

## Features

- **Data Ingestion**: Load and process Arabic Twitter datasets in CSV or JSON format
- **Text Preprocessing**: Clean and normalize Arabic text with dialect handling
- **Trend Detection**: Identify trending topics using AraBERT embeddings and K-means clustering
- **Content Generation**: Create marketing content using AraGPT2, including:
  - Social media captions
  - Relevant hashtags
  - Ad scripts
- **Cultural Relevance**: Filter content to ensure cultural sensitivity
- **User Interfaces**:
  - Command-line interface for batch processing
  - Interactive Streamlit dashboard with visualizations

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation Steps

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/arabic-marketing-generator.git
   cd arabic-marketing-generator
   ```

2. Install the package and dependencies:
   ```
   pip install -e .
   ```

   This will install all required dependencies including:
   - numpy, pandas, scikit-learn (for data processing)
   - pyarabic, farasapy (for Arabic text processing)
   - transformers, torch (for NLP models)
   - streamlit (for dashboard)

## Usage

### Command-Line Interface

The package provides a command-line interface for batch processing:

```bash
python arabic_marketing_generator.py --dataset path/to/tweets.csv --output_dir ./output --num_trends 10
```

#### CLI Options

- `--dataset`: Path to the input dataset (CSV or JSON) [required]
- `--output_dir`: Directory to save output files [default: ./output]
- `--text_column`: Name of the column containing tweet text [default: text]
- `--timestamp_column`: Name of the column containing tweet timestamp [default: created_at]
- `--num_trends`: Number of top trends to detect [default: 10]
- `--num_captions`: Number of captions to generate per trend [default: 3]
- `--num_hashtags`: Number of hashtags to generate per trend [default: 5]
- `--num_ads`: Number of ad scripts to generate per trend [default: 1]
- `--config`: Path to configuration file (JSON)

### Streamlit Dashboard

The package includes an interactive dashboard for visual exploration:

```bash
streamlit run dashboard/app.py
```

The dashboard provides:
- Dataset upload and preview
- Configuration options
- Trend visualization with word clouds
- Generated content display
- Export functionality

## Configuration

You can customize the behavior using a JSON configuration file:

```json
{
  "text_column": "text",
  "timestamp_column": "created_at",
  "min_cluster_size": 5,
  "max_clusters": 20,
  "num_trends": 10,
  "model_name": "aubmindlab/aragpt2-medium",
  "num_captions": 3,
  "num_hashtags": 5,
  "num_ads": 1,
  "filter_content": true,
  "sensitive_terms": [
    "سياسة", "دين", "طائفة", "مذهب", "حرب", "صراع", "خلاف",
    "عنصرية", "تمييز", "إرهاب", "متطرف", "عنف"
  ],
  "brand_voice": "neutral",
  "emoji_usage": "moderate",
  "content_length": "medium"
}
```

## Project Structure

```
arabic_marketing_generator/
├── data/                      # Data directory
├── src/                       # Source code
│   ├── preprocessing/         # Data preprocessing modules
│   ├── trend_detection/       # Trend detection modules
│   ├── content_generation/    # Content generation modules
│   └── utils/                 # Utility functions
├── tests/                     # Test scripts
├── dashboard/                 # Streamlit dashboard
├── docs/                      # Documentation
├── arabic_marketing_generator.py  # CLI entry point
├── setup.py                   # Package setup
└── README.md                  # This file
```

## Components

### Data Preprocessing

The preprocessing pipeline handles:
- Loading Twitter datasets in CSV/JSON formats
- Cleaning Arabic text (removing URLs, emojis, etc.)
- Normalizing Arabic characters
- Tokenization and lemmatization
- Handling dialect variations

### Trend Detection

The trend detection system:
- Extracts embeddings using AraBERT
- Identifies keywords and hashtags
- Clusters similar content using K-means
- Ranks trends by size and recency
- Adapts clustering thresholds automatically

### Content Generation

The content generation system:
- Uses AraGPT2 to generate culturally relevant content
- Creates social media captions with varied tones
- Generates appropriate hashtags
- Produces ad scripts for marketing campaigns
- Filters content for cultural sensitivity

## Customization

### Brand Voice

You can customize the brand voice in the configuration:
- `neutral`: Balanced, professional tone
- `formal`: More corporate and professional
- `casual`: Friendly and conversational
- `enthusiastic`: Energetic and exciting

### Content Length

Adjust content length preferences:
- `short`: Brief, concise content
- `medium`: Standard length content
- `long`: More detailed content

### Emoji Usage

Control emoji density in generated content:
- `none`: No emojis
- `minimal`: Very few emojis
- `moderate`: Balanced emoji usage
- `heavy`: Abundant emoji usage

## Troubleshooting

### Common Issues

#### Import Errors

If you encounter import errors:
```
ModuleNotFoundError: No module named 'src'
```

Make sure you've installed the package in development mode:
```
pip install -e .
```

#### Memory Issues

For large datasets, you may encounter memory errors. Try:
- Processing a subset of the data
- Increasing system swap space
- Using a machine with more RAM

#### Arabic Text Display Issues

If Arabic text appears incorrectly:
- Ensure your terminal supports RTL languages
- Use a font that supports Arabic characters
- Set appropriate encoding in your environment

### Model Loading Issues

If you encounter errors loading the AraBERT or AraGPT2 models:
- Check your internet connection
- Verify you have sufficient disk space
- Try downloading the models manually using the Hugging Face CLI

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [AraBERT](https://github.com/aub-mind/arabert) for Arabic language embeddings
- [AraGPT2](https://github.com/aub-mind/aragpt2) for Arabic text generation
- [PyArabic](https://github.com/linuxscout/pyarabic) for Arabic text processing
- [Farasa](https://farasa.qcri.org/) for Arabic NLP tools
# project_ai
