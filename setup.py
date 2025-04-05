from setuptools import setup, find_packages

setup(
    name="arabic_marketing_generator",
    version="0.1.0",
    description="A tool for generating culturally relevant Arabic marketing content from Twitter data",
    author="",
    author_email="",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "streamlit>=1.10.0",
        "transformers>=4.15.0",
        "torch>=1.10.0",
        "pyarabic>=0.6.15",
        "farasapy>=0.0.14",
        "wordcloud>=1.8.0"
    ],
    entry_points={
        'console_scripts': [
            'arabic-marketing-generator=arabic_marketing_generator:main',
        ],
    },
    python_requires='>=3.8',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
