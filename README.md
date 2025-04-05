# Arabic Marketing Content Generator - Web Deployment

This directory contains the web deployment version of the Arabic Marketing Content Generator.

## Files

- `app.py`: Streamlit application for the web interface
- `environment.yml`: Conda environment file for dependencies
- `sample_*.json/csv`: Sample data files for demonstration
- `static/`: Directory for static assets

## Deployment Instructions

1. Clone this repository
2. Install dependencies using Conda:
   ```
   conda env create -f environment.yml
   conda activate arabic-marketing
   ```
3. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

## Deployment to Streamlit Cloud

This application can be deployed to Streamlit Cloud for permanent hosting:

1. Push this code to a GitHub repository
2. Sign up for [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your GitHub repository
4. Select the `app.py` file and deploy

## Deployment to Heroku

Alternatively, you can deploy to Heroku:

1. Create a `Procfile` with:
   ```
   web: streamlit run app.py --server.port=$PORT
   ```
2. Push to Heroku:
   ```
   heroku create
   git push heroku main
   ```

## Features

- View sample Arabic Twitter trends
- Explore generated marketing content
- Visualize trend data with word clouds
- Learn about the project methodology
