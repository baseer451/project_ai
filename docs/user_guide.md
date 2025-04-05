# User Guide: Arabic Marketing Content Generator

## Introduction

The Arabic Marketing Content Generator is a powerful tool designed to help marketers create culturally relevant content for Arabic-speaking audiences. This guide will walk you through how to use the system effectively, from data preparation to content generation.

## Getting Started

### System Requirements

- **Operating System**: Windows 10+, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **RAM**: Minimum 8GB, recommended 16GB for larger datasets
- **Disk Space**: At least 5GB free space
- **Python**: Version 3.8 or higher

### Installation

1. **Install Python**: If not already installed, download and install Python from [python.org](https://python.org)

2. **Install the package**:
   ```bash
   pip install arabic-marketing-generator
   ```
   
   Or install from source:
   ```bash
   git clone https://github.com/yourusername/arabic-marketing-generator.git
   cd arabic-marketing-generator
   pip install -e .
   ```

3. **Verify installation**:
   ```bash
   python -c "import arabic_marketing_generator; print('Installation successful!')"
   ```

## Data Preparation

### Supported Data Formats

The system accepts Twitter data in the following formats:
- CSV files with tweet text and optional metadata
- JSON files with tweet objects

### Required Data Fields

Your dataset should include at least:
- **Text column**: Contains the Arabic tweet text
- **Timestamp column** (optional): Contains the creation date/time

### Sample Data Format

CSV example:
```
id,text,created_at
1,تخفيضات كبيرة بمناسبة رمضان,2023-03-15T12:30:45Z
2,أفضل المنتجات الجديدة لهذا الموسم,2023-03-16T08:15:22Z
```

JSON example:
```json
[
  {
    "id": 1,
    "text": "تخفيضات كبيرة بمناسبة رمضان",
    "created_at": "2023-03-15T12:30:45Z"
  },
  {
    "id": 2,
    "text": "أفضل المنتجات الجديدة لهذا الموسم",
    "created_at": "2023-03-16T08:15:22Z"
  }
]
```

### Data Cleaning Recommendations

For best results:
- Remove duplicate tweets
- Ensure text is in Arabic (UTF-8 encoding)
- Include recent data for more relevant trends
- Aim for at least 1,000 tweets for meaningful trend detection

## Using the Command-Line Interface

### Basic Usage

Process a dataset and generate marketing content:

```bash
python -m arabic_marketing_generator --dataset path/to/tweets.csv --output_dir ./output
```

### Advanced Options

Customize the processing with additional options:

```bash
python -m arabic_marketing_generator \
  --dataset path/to/tweets.csv \
  --output_dir ./output \
  --text_column tweet_text \
  --timestamp_column tweet_time \
  --num_trends 15 \
  --num_captions 5 \
  --num_hashtags 8 \
  --num_ads 2 \
  --config path/to/config.json
```

### Output Files

The CLI generates the following output files:
- `processed_data.csv`: Preprocessed tweet data
- `trends.json`: Detected trends with keywords and metadata
- `generated_content.json`: Generated marketing content

## Using the Streamlit Dashboard

### Starting the Dashboard

Launch the interactive dashboard:

```bash
streamlit run -m arabic_marketing_generator.dashboard.app
```

### Dashboard Features

1. **Data Upload**: Upload your Twitter dataset (CSV/JSON)
2. **Configuration**: Set processing options
3. **Processing**: Generate trends and content
4. **Visualization**: Explore trends with interactive visualizations
5. **Content Review**: Browse and edit generated content
6. **Export**: Download results in various formats

### Dashboard Walkthrough

1. **Upload Data**:
   - Click "Upload Twitter Dataset" and select your file
   - Review the dataset preview to ensure correct loading

2. **Configure Options**:
   - Set column names for text and timestamp
   - Adjust advanced options if needed

3. **Generate Content**:
   - Click "Generate Marketing Content" to start processing
   - Monitor the progress bar

4. **Explore Results**:
   - View detected trends in the "Trends" tab
   - Explore visualizations in the "Visualizations" tab
   - Review generated content in the "Generated Content" tab

5. **Download Results**:
   - Use the download links to save processed data, trends, and content

## Customizing Content Generation

### Configuration File

Create a JSON configuration file to customize the system:

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

### Brand Voice Options

Customize the tone of generated content:

- **neutral**: Balanced, professional tone suitable for most brands
- **formal**: Corporate and professional, ideal for B2B or financial services
- **casual**: Friendly and conversational, good for consumer brands
- **enthusiastic**: Energetic and exciting, perfect for promotions or events

### Content Length Options

Control the verbosity of generated content:

- **short**: Brief captions (1-2 sentences) and concise ad scripts
- **medium**: Standard length content (2-3 sentences)
- **long**: More detailed content with additional context

### Emoji Usage Options

Adjust emoji density in generated content:

- **none**: No emojis in generated content
- **minimal**: Very few, strategically placed emojis
- **moderate**: Balanced emoji usage for emphasis
- **heavy**: Abundant emoji usage for expressive content

## Best Practices

### For Optimal Trend Detection

- Use datasets with at least 1,000 tweets
- Include recent data (within the last month)
- Focus on a specific market segment or topic area
- Include hashtags in the original data if possible

### For Quality Content Generation

- Review and edit generated content before publishing
- Use the filtering system to ensure cultural appropriateness
- Customize brand voice to match your brand identity
- Combine multiple generated captions for best results

### For Performance Optimization

- Clean your dataset before processing
- Start with a smaller sample to test settings
- Increase `min_cluster_size` for larger datasets
- Use CSV format for faster processing

## Troubleshooting

### Common Issues and Solutions

#### Slow Processing

**Issue**: Processing takes too long
**Solution**: 
- Reduce dataset size
- Increase `min_cluster_size`
- Use a machine with more RAM

#### Poor Quality Trends

**Issue**: Detected trends are not meaningful
**Solution**:
- Use a larger dataset
- Decrease `min_cluster_size`
- Ensure dataset is focused on relevant topics

#### Arabic Text Display Problems

**Issue**: Arabic text appears as boxes or question marks
**Solution**:
- Ensure your terminal/editor supports RTL languages
- Use a font that supports Arabic characters
- Check file encoding (should be UTF-8)

#### Out of Memory Errors

**Issue**: System crashes with memory errors
**Solution**:
- Process a smaller dataset
- Close other applications
- Increase system swap space

## Getting Help

If you encounter issues not covered in this guide:

- Check the [GitHub repository](https://github.com/yourusername/arabic-marketing-generator) for updates
- Submit an issue on GitHub with details about your problem
- Contact support at support@example.com

## Examples

### Example 1: E-commerce Promotions

```bash
python -m arabic_marketing_generator \
  --dataset ecommerce_tweets.csv \
  --config ecommerce_config.json
```

With `ecommerce_config.json`:
```json
{
  "brand_voice": "enthusiastic",
  "emoji_usage": "heavy",
  "content_length": "short"
}
```

### Example 2: Corporate Communications

```bash
python -m arabic_marketing_generator \
  --dataset corporate_tweets.csv \
  --config corporate_config.json
```

With `corporate_config.json`:
```json
{
  "brand_voice": "formal",
  "emoji_usage": "none",
  "content_length": "medium"
}
```

## Conclusion

The Arabic Marketing Content Generator streamlines the process of creating culturally relevant marketing content for Arabic-speaking audiences. By following this guide, you can effectively leverage the power of AI to enhance your marketing campaigns while ensuring cultural sensitivity and relevance.
