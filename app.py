import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import tempfile
from PIL import Image
import io
import base64

# Set page configuration
st.set_page_config(
    page_title="Arabic Marketing Content Generator",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .highlight {
        background-color: #f1f8e9;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .arabic-text {
        font-family: 'Arial', sans-serif;
        font-size: 18px;
        direction: rtl;
        text-align: right;
    }
</style>
""", unsafe_allow_html=True)

# Load sample data
def load_sample_data():
    # Sample tweets
    sample_tweets = pd.read_csv('sample_tweets.csv')
    
    # Sample trends
    with open('sample_trends.json', 'r', encoding='utf-8') as f:
        sample_trends = json.load(f)
    
    # Sample content
    with open('sample_generated_content.json', 'r', encoding='utf-8') as f:
        sample_content = json.load(f)
    
    return sample_tweets, sample_trends, sample_content

# Create word cloud
def create_wordcloud(text, title="Word Cloud", max_words=100):
    # Create word cloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        max_words=max_words,
        contour_width=1,
        contour_color='steelblue'
    ).generate(text)
    
    # Display word cloud
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(title, fontsize=16)
    ax.axis('off')
    
    return fig

# Display trend visualizations
def display_trend_visualizations(trends):
    st.subheader("Trend Visualizations")
    
    # Create tabs for different visualizations
    tab1, tab2 = st.tabs(["Word Cloud", "Top Keywords"])
    
    with tab1:
        # Word cloud of all keywords
        all_keywords = []
        for trend in trends:
            all_keywords.extend(trend.get('keywords', []))
        
        if all_keywords:
            keyword_text = ' '.join(all_keywords)
            fig = create_wordcloud(keyword_text, title="Keywords Word Cloud")
            st.pyplot(fig)
        else:
            st.info("No keywords available for word cloud.")
    
    with tab2:
        # Top keywords bar chart
        keyword_counts = {}
        for trend in trends:
            for keyword in trend.get('keywords', [])[:5]:  # Take top 5 keywords from each trend
                if keyword in keyword_counts:
                    keyword_counts[keyword] += 1
                else:
                    keyword_counts[keyword] = 1
        
        if keyword_counts:
            # Sort by count and take top 15
            sorted_keywords = dict(sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:15])
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=list(sorted_keywords.values()), y=list(sorted_keywords.keys()), ax=ax)
            ax.set_title('Top Keywords Across Trends')
            ax.set_xlabel('Frequency')
            
            st.pyplot(fig)
        else:
            st.info("No keywords available for visualization.")

# Display content
def display_content(content):
    st.subheader("Generated Marketing Content")
    
    for i, item in enumerate(content, 1):
        with st.expander(f"Trend {i}: {item['trend']}", expanded=i==1):
            # Captions
            st.markdown("#### Social Media Captions")
            for j, caption in enumerate(item['captions'], 1):
                st.markdown(f"""
                <div class="highlight arabic-text">
                    {caption}
                </div>
                """, unsafe_allow_html=True)
            
            # Hashtags
            st.markdown("#### Hashtags")
            hashtag_html = " ".join([f'<span style="background-color: #e3f2fd; padding: 5px; margin: 5px; border-radius: 5px;">{hashtag}</span>' for hashtag in item['hashtags']])
            st.markdown(f"""
            <div class="arabic-text">
                {hashtag_html}
            </div>
            """, unsafe_allow_html=True)
            
            # Ad Scripts
            st.markdown("#### Ad Scripts")
            for j, script in enumerate(item['ad_scripts'], 1):
                st.markdown(f"""
                <div class="highlight arabic-text">
                    {script}
                </div>
                """, unsafe_allow_html=True)

# Main function
def main():
    st.title("Arabic Marketing Content Generator")
    st.markdown("Generate culturally relevant marketing content from Arabic Twitter data")
    
    # Load sample data
    try:
        sample_tweets, sample_trends, sample_content = load_sample_data()
        data_loaded = True
    except Exception as e:
        st.error(f"Error loading sample data: {str(e)}")
        data_loaded = False
    
    # Sidebar
    st.sidebar.header("About")
    st.sidebar.markdown("""
    This tool helps marketers create engaging social media content that resonates with Arabic-speaking audiences by leveraging natural language processing and machine learning techniques.
    
    **Features:**
    - Data Ingestion: Load and process Arabic Twitter datasets
    - Trend Detection: Identify trending topics using AraBERT
    - Content Generation: Create marketing content using AraGPT2
    - Cultural Relevance: Filter content to ensure cultural sensitivity
    """)
    
    st.sidebar.header("Demo Options")
    demo_option = st.sidebar.radio(
        "Choose a demo option:",
        ["View Sample Trends", "View Generated Content", "About the Project"]
    )
    
    # Main content area
    if demo_option == "View Sample Trends" and data_loaded:
        st.header("Detected Trends")
        
        # Display sample tweets
        with st.expander("Sample Twitter Data"):
            st.dataframe(sample_tweets)
        
        # Display trends
        for i, trend in enumerate(sample_trends, 1):
            st.markdown(f"""
            <div class="highlight">
                <h3>{i}. {trend['topic']}</h3>
                <p><strong>Keywords:</strong> {', '.join(trend['keywords'][:5])}</p>
                <p><strong>Hashtags:</strong> {', '.join(trend['hashtags'])}</p>
                <p><strong>Cluster Size:</strong> {trend['cluster_size']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Display visualizations
        display_trend_visualizations(sample_trends)
    
    elif demo_option == "View Generated Content" and data_loaded:
        st.header("Generated Marketing Content")
        
        # Display content
        display_content(sample_content)
    
    elif demo_option == "About the Project":
        st.header("About the Project")
        
        st.markdown("""
        ## Overview

        The Arabic Marketing Content Generator is an autonomous system that processes Arabic Twitter data, detects trending topics, and generates culturally relevant marketing content. This tool helps marketers create engaging social media content that resonates with Arabic-speaking audiences by leveraging natural language processing and machine learning techniques.

        ## How It Works

        1. **Data Ingestion**: Load and process Arabic Twitter datasets in CSV or JSON format
        2. **Text Preprocessing**: Clean and normalize Arabic text with dialect handling
        3. **Trend Detection**: Identify trending topics using AraBERT embeddings and K-means clustering
        4. **Content Generation**: Create marketing content using AraGPT2, including:
           - Social media captions
           - Relevant hashtags
           - Ad scripts
        5. **Cultural Relevance**: Filter content to ensure cultural sensitivity

        ## Technologies Used

        - **AraBERT**: For Arabic language embeddings and feature extraction
        - **AraGPT2**: For Arabic text generation
        - **PyArabic**: For Arabic text processing
        - **Streamlit**: For this interactive dashboard
        
        ## Contact
        
        For more information or to request a custom implementation, please contact us.
        """)
    
    else:
        st.warning("Sample data could not be loaded. Please check the file paths.")

if __name__ == "__main__":
    main()
