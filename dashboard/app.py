"""
Streamlit dashboard for Arabic Marketing Content Generator.

This module provides a web interface for using the Arabic Marketing Content Generator.
"""

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

from src.preprocessing import PreprocessingPipeline
from src.trend_detection import TrendDetectionPipeline
from src.content_generation import ContentGenerationPipeline


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


def create_wordcloud(text, title="Word Cloud", max_words=100):
    """Create and display a word cloud from text."""
    # Create word cloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        max_words=max_words,
        font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',  # Font that supports Arabic
        contour_width=1,
        contour_color='steelblue'
    ).generate(text)
    
    # Display word cloud
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(title, fontsize=16)
    ax.axis('off')
    
    return fig


def display_trend_visualizations(df, trends):
    """Display visualizations for detected trends."""
    st.subheader("Trend Visualizations")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Trend Distribution", "Word Cloud", "Top Keywords"])
    
    with tab1:
        # Trend distribution chart
        if 'cluster' in df.columns:
            cluster_counts = df['cluster'].value_counts().sort_index()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            cluster_counts.plot(kind='bar', ax=ax)
            ax.set_title('Cluster Size Distribution')
            ax.set_xlabel('Cluster ID')
            ax.set_ylabel('Number of Tweets')
            
            st.pyplot(fig)
    
    with tab2:
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
    
    with tab3:
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


def display_content(content):
    """Display generated marketing content."""
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


def get_download_link(file_path, link_text):
    """Generate a download link for a file."""
    with open(file_path, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{os.path.basename(file_path)}">{link_text}</a>'
    return href


def main():
    """Main function for the Streamlit dashboard."""
    st.title("Arabic Marketing Content Generator")
    st.markdown("Generate culturally relevant marketing content from Arabic Twitter data")
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload Twitter Dataset (CSV/JSON)", type=["csv", "json"])
    
    # Configuration options
    text_column = st.sidebar.text_input("Text Column Name", "text")
    timestamp_column = st.sidebar.text_input("Timestamp Column Name", "created_at")
    
    # Advanced options in expander
    with st.sidebar.expander("Advanced Options"):
        num_trends = st.number_input("Number of Trends", min_value=1, max_value=20, value=10)
        num_captions = st.number_input("Captions per Trend", min_value=1, max_value=5, value=3)
        num_hashtags = st.number_input("Hashtags per Trend", min_value=1, max_value=10, value=5)
        num_ads = st.number_input("Ad Scripts per Trend", min_value=1, max_value=3, value=1)
        filter_content = st.checkbox("Filter Sensitive Content", value=True)
    
    # Main content area
    if uploaded_file is not None:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            dataset_path = tmp_file.name
        
        # Create output directory
        output_dir = tempfile.mkdtemp()
        processed_path = os.path.join(output_dir, 'processed_data.csv')
        trends_path = os.path.join(output_dir, 'trends.json')
        content_path = os.path.join(output_dir, 'generated_content.json')
        
        # Display dataset info
        if dataset_path.endswith('.csv'):
            df = pd.read_csv(dataset_path)
        else:
            df = pd.read_json(dataset_path)
        
        st.markdown(f"### Dataset Information")
        st.write(f"Number of rows: {len(df)}")
        st.write(f"Columns: {', '.join(df.columns)}")
        
        # Sample data
        with st.expander("Preview Dataset"):
            st.dataframe(df.head())
        
        # Process button
        if st.button("Generate Marketing Content"):
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Preprocessing
                status_text.text("Step 1/3: Preprocessing data...")
                preprocessing_pipeline = PreprocessingPipeline(
                    text_column=text_column,
                    timestamp_column=timestamp_column
                )
                
                # Run preprocessing pipeline
                df = preprocessing_pipeline.run_pipeline(dataset_path, new_column='processed_text')
                preprocessing_pipeline.save_processed_data(processed_path)
                progress_bar.progress(33)
                
                # Step 2: Trend Detection
                status_text.text("Step 2/3: Detecting trends...")
                trend_pipeline = TrendDetectionPipeline(
                    min_cluster_size=5,
                    max_clusters=20
                )
                
                # Run trend detection pipeline
                df, trends = trend_pipeline.run_pipeline(
                    df,
                    text_column='processed_text',
                    raw_text_column=text_column,
                    n_trends=num_trends
                )
                
                # Save trends
                with open(trends_path, 'w', encoding='utf-8') as f:
                    json.dump(trends, f, ensure_ascii=False, indent=2)
                progress_bar.progress(66)
                
                # Step 3: Content Generation
                status_text.text("Step 3/3: Generating content...")
                content_pipeline = ContentGenerationPipeline()
                
                # Generate content
                content = content_pipeline.generate_content(
                    trends,
                    num_captions=num_captions,
                    num_hashtags=num_hashtags,
                    num_ads=num_ads
                )
                
                # Filter content if specified
                if filter_content:
                    content = content_pipeline.filter_content()
                
                # Save content
                content_pipeline.save_content(content_path)
                progress_bar.progress(100)
                status_text.text("Processing completed!")
                
                # Display results
                st.markdown("## Results")
                
                # Tabs for different views
                tab1, tab2, tab3 = st.tabs(["Trends", "Visualizations", "Generated Content"])
                
                with tab1:
                    st.subheader("Detected Trends")
                    for i, trend in enumerate(trends, 1):
                        st.markdown(f"""
                        <div class="highlight">
                            <h3>{i}. {trend['topic']}</h3>
                            <p><strong>Keywords:</strong> {', '.join(trend['keywords'][:5])}</p>
                            <p><strong>Cluster Size:</strong> {trend['cluster_size']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with tab2:
                    display_trend_visualizations(df, trends)
                
                with tab3:
                    display_content(content)
                
                # Download section
                st.markdown("## Download Results")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(get_download_link(processed_path, "Download Processed Data (CSV)"), unsafe_allow_html=True)
                
                with col2:
                    st.markdown(get_download_link(trends_path, "Download Trends (JSON)"), unsafe_allow_html=True)
                
                with col3:
                    st.markdown(get_download_link(content_path, "Download Generated Content (JSON)"), unsafe_allow_html=True)
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.exception(e)
    
    else:
        # Display instructions when no file is uploaded
        st.info("Please upload a Twitter dataset (CSV or JSON) to get started.")
        
        # Example section
        with st.expander("How to Use This Tool"):
            st.markdown("""
            ### Instructions
            
            1. **Upload Dataset**: Use the file uploader in the sidebar to upload your Arabic Twitter dataset in CSV or JSON format.
            2. **Configure Options**: Set the column names and advanced options in the sidebar.
            3. **Generate Content**: Click the "Generate Marketing Content" button to start the process.
            4. **View Results**: Explore the detected trends, visualizations, and generated marketing content.
            5. **Download**: Download the processed data, trends, and generated content for your use.
            
            ### Dataset Format
            
            Your dataset should contain at least these columns:
            - Text column: Contains the tweet text in Arabic
            - Timestamp column (optional): Contains the tweet creation date/time
            
            ### Example Use Cases
            
            - Generate marketing content for Ramadan promotions
            - Create social media posts for trending topics
            - Develop ad campaigns based on current interests
            """)


if __name__ == "__main__":
    main()
