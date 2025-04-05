# Quick Start Guide: Arabic Marketing Content Generator

This quick start guide will help you get up and running with the Arabic Marketing Content Generator in just a few minutes.

## Installation

```bash
# Install the package
pip install arabic-marketing-generator

# Or install from source
git clone https://github.com/yourusername/arabic-marketing-generator.git
cd arabic-marketing-generator
pip install -e .
```

## Basic Usage

### Command Line

```bash
# Process a dataset and generate marketing content
python -m arabic_marketing_generator --dataset your_tweets.csv --output_dir ./output
```

### Dashboard

```bash
# Launch the interactive dashboard
streamlit run -m arabic_marketing_generator.dashboard.app
```

## 5-Minute Tutorial

1. **Prepare your data**: Ensure you have a CSV file with Arabic tweets in a column named "text"

2. **Process the data**:
   ```bash
   python -m arabic_marketing_generator --dataset your_tweets.csv --output_dir ./output
   ```

3. **Review the results**:
   - Check `output/trends.json` for detected trends
   - Check `output/generated_content.json` for marketing content

4. **Use the content**:
   - Copy captions for social media posts
   - Use hashtags to increase visibility
   - Adapt ad scripts for your marketing campaigns

## Next Steps

- Read the [User Guide](user_guide.md) for detailed instructions
- Explore [Technical Documentation](technical_docs.md) for advanced usage
- Customize the configuration to match your brand voice

## Need Help?

- Check the [Troubleshooting](user_guide.md#troubleshooting) section
- Visit our GitHub repository for updates
- Contact support at support@example.com
