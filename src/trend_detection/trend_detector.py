"""
Trend Detector Module for Arabic Marketing Content Generator

This module handles detecting trends from Arabic text using K-means clustering.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import Counter
from typing import List, Dict, Any, Optional, Union, Tuple
import datetime


class TrendDetector:
    """
    Class for detecting trends in Arabic text data using K-means clustering.
    """
    
    def __init__(self, min_cluster_size: int = 5, max_clusters: int = 20):
        """
        Initialize the TrendDetector.
        
        Args:
            min_cluster_size: Minimum number of items to consider a cluster as a trend
            max_clusters: Maximum number of clusters to try
        """
        self.min_cluster_size = min_cluster_size
        self.max_clusters = max_clusters
        self.kmeans = None
        self.cluster_keywords = {}
        self.cluster_hashtags = {}
        self.cluster_sizes = {}
        self.cluster_recency = {}
    
    def find_optimal_clusters(self, embeddings: np.ndarray, min_clusters: int = 2) -> int:
        """
        Find the optimal number of clusters using silhouette score.
        
        Args:
            embeddings: Array of text embeddings
            min_clusters: Minimum number of clusters to try
            
        Returns:
            Optimal number of clusters
        """
        # Limit max clusters to number of samples - 1 or max_clusters, whichever is smaller
        max_k = min(len(embeddings) - 1, self.max_clusters)
        
        # If we have too few samples, return min_clusters
        if max_k <= min_clusters:
            return min_clusters
        
        best_score = -1
        best_k = min_clusters
        
        # Try different numbers of clusters
        for k in range(min_clusters, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)
            
            # Calculate silhouette score
            try:
                score = silhouette_score(embeddings, labels)
                
                if score > best_score:
                    best_score = score
                    best_k = k
            except:
                # If silhouette score fails (e.g., single-item clusters), skip this k
                continue
        
        return best_k
    
    def detect_trends(self, df: pd.DataFrame, 
                     embedding_column: str = 'embeddings',
                     keywords_column: str = 'keywords',
                     hashtags_column: Optional[str] = 'hashtags',
                     timestamp_column: Optional[str] = 'created_at',
                     cluster_column: str = 'cluster') -> pd.DataFrame:
        """
        Detect trends using K-means clustering on embeddings.
        
        Args:
            df: DataFrame containing text embeddings and keywords
            embedding_column: Name of column containing embeddings
            keywords_column: Name of column containing keywords
            hashtags_column: Name of column containing hashtags (optional)
            timestamp_column: Name of column containing timestamps (optional)
            cluster_column: Name of column to store cluster assignments
            
        Returns:
            DataFrame with cluster assignments
        """
        # Extract embeddings
        embeddings = np.vstack(df[embedding_column].tolist())
        
        # Find optimal number of clusters
        n_clusters = self.find_optimal_clusters(embeddings)
        
        # Apply K-means clustering
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df[cluster_column] = self.kmeans.fit_predict(embeddings)
        
        # Extract cluster information
        self._extract_cluster_info(df, keywords_column, hashtags_column, timestamp_column, cluster_column)
        
        return df
    
    def _extract_cluster_info(self, df: pd.DataFrame,
                             keywords_column: str,
                             hashtags_column: Optional[str],
                             timestamp_column: Optional[str],
                             cluster_column: str):
        """
        Extract information about each cluster.
        
        Args:
            df: DataFrame with cluster assignments
            keywords_column: Name of column containing keywords
            hashtags_column: Name of column containing hashtags
            timestamp_column: Name of column containing timestamps
            cluster_column: Name of column containing cluster assignments
        """
        # Reset cluster info
        self.cluster_keywords = {}
        self.cluster_hashtags = {}
        self.cluster_sizes = {}
        self.cluster_recency = {}
        
        # Get unique clusters
        clusters = df[cluster_column].unique()
        
        for cluster in clusters:
            # Get items in this cluster
            cluster_items = df[df[cluster_column] == cluster]
            
            # Store cluster size
            self.cluster_sizes[cluster] = len(cluster_items)
            
            # Extract keywords
            all_keywords = [kw for keywords in cluster_items[keywords_column] for kw in keywords]
            keyword_counts = Counter(all_keywords)
            self.cluster_keywords[cluster] = keyword_counts.most_common(10)
            
            # Extract hashtags if available
            if hashtags_column and hashtags_column in df.columns:
                all_hashtags = [ht for hashtags in cluster_items[hashtags_column] for ht in hashtags]
                hashtag_counts = Counter(all_hashtags)
                self.cluster_hashtags[cluster] = hashtag_counts.most_common(10)
            
            # Calculate recency if timestamp available
            if timestamp_column and timestamp_column in df.columns:
                try:
                    # Convert timestamps to datetime if they're strings
                    if isinstance(cluster_items[timestamp_column].iloc[0], str):
                        timestamps = pd.to_datetime(cluster_items[timestamp_column])
                    else:
                        timestamps = cluster_items[timestamp_column]
                    
                    # Calculate average timestamp (recency)
                    avg_timestamp = timestamps.mean()
                    self.cluster_recency[cluster] = avg_timestamp
                except:
                    # If timestamp conversion fails, use cluster size as recency
                    self.cluster_recency[cluster] = len(cluster_items)
    
    def get_top_trends(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        Get the top N trends based on cluster size and recency.
        
        Args:
            n: Number of top trends to return
            
        Returns:
            List of trend dictionaries with topic, keywords, hashtags, and cluster_size
        """
        if not self.cluster_sizes:
            raise ValueError("No trends detected. Run detect_trends() first.")
        
        # Filter clusters by minimum size
        valid_clusters = {c: size for c, size in self.cluster_sizes.items() 
                         if size >= self.min_cluster_size}
        
        if not valid_clusters:
            return []
        
        # Rank clusters by a combination of size and recency
        if self.cluster_recency:
            # Normalize recency to 0-1 range
            recency_values = list(self.cluster_recency.values())
            if isinstance(recency_values[0], datetime.datetime):
                # For datetime values, newer is better
                min_time = min(recency_values)
                max_time = max(recency_values)
                time_range = (max_time - min_time).total_seconds()
                if time_range == 0:
                    normalized_recency = {c: 1.0 for c in self.cluster_recency}
                else:
                    normalized_recency = {
                        c: (t - min_time).total_seconds() / time_range 
                        for c, t in self.cluster_recency.items()
                    }
            else:
                # For other values (like cluster size), higher is better
                min_val = min(recency_values)
                max_val = max(recency_values)
                val_range = max_val - min_val
                if val_range == 0:
                    normalized_recency = {c: 1.0 for c in self.cluster_recency}
                else:
                    normalized_recency = {
                        c: (t - min_val) / val_range 
                        for c, t in self.cluster_recency.items()
                    }
            
            # Normalize size to 0-1 range
            size_values = list(valid_clusters.values())
            min_size = min(size_values)
            max_size = max(size_values)
            size_range = max_size - min_size
            if size_range == 0:
                normalized_size = {c: 1.0 for c in valid_clusters}
            else:
                normalized_size = {
                    c: (s - min_size) / size_range 
                    for c, s in valid_clusters.items()
                }
            
            # Combine size and recency (0.7 * size + 0.3 * recency)
            cluster_scores = {
                c: 0.7 * normalized_size[c] + 0.3 * normalized_recency[c]
                for c in valid_clusters
            }
            
            # Sort clusters by score
            sorted_clusters = sorted(cluster_scores.items(), key=lambda x: x[1], reverse=True)
            top_clusters = [c for c, _ in sorted_clusters[:n]]
        else:
            # If no recency information, sort by size only
            sorted_clusters = sorted(valid_clusters.items(), key=lambda x: x[1], reverse=True)
            top_clusters = [c for c, _ in sorted_clusters[:n]]
        
        # Create trend dictionaries
        trends = []
        for cluster in top_clusters:
            trend = {
                "topic": self._get_cluster_topic(cluster),
                "keywords": [kw for kw, _ in self.cluster_keywords.get(cluster, [])],
                "cluster_size": self.cluster_sizes[cluster]
            }
            
            # Add hashtags if available
            if self.cluster_hashtags:
                trend["hashtags"] = [ht for ht, _ in self.cluster_hashtags.get(cluster, [])]
            
            trends.append(trend)
        
        return trends
    
    def _get_cluster_topic(self, cluster: int) -> str:
        """
        Get a representative topic name for a cluster.
        
        Args:
            cluster: Cluster ID
            
        Returns:
            Topic name based on top keywords or hashtags
        """
        # Try to use hashtags first
        if cluster in self.cluster_hashtags and self.cluster_hashtags[cluster]:
            top_hashtag, _ = self.cluster_hashtags[cluster][0]
            return top_hashtag
        
        # Fall back to keywords
        if cluster in self.cluster_keywords and self.cluster_keywords[cluster]:
            # Use top 2 keywords if available
            if len(self.cluster_keywords[cluster]) >= 2:
                kw1, _ = self.cluster_keywords[cluster][0]
                kw2, _ = self.cluster_keywords[cluster][1]
                return f"{kw1} {kw2}"
            else:
                kw, _ = self.cluster_keywords[cluster][0]
                return kw
        
        # If no keywords or hashtags, use cluster ID
        return f"Cluster {cluster}"
    
    def adjust_clustering(self, embeddings: np.ndarray, current_n_clusters: int, 
                         increase: bool = True) -> Tuple[KMeans, np.ndarray]:
        """
        Adjust clustering by increasing or decreasing the number of clusters.
        
        Args:
            embeddings: Array of text embeddings
            current_n_clusters: Current number of clusters
            increase: Whether to increase (True) or decrease (False) the number of clusters
            
        Returns:
            Tuple of (new KMeans model, new cluster labels)
        """
        if increase:
            new_n_clusters = min(current_n_clusters + 2, len(embeddings) - 1, self.max_clusters)
        else:
            new_n_clusters = max(current_n_clusters - 2, 2)
        
        # If no change, return current model
        if new_n_clusters == current_n_clusters:
            return self.kmeans, self.kmeans.labels_
        
        # Create new model
        new_kmeans = KMeans(n_clusters=new_n_clusters, random_state=42, n_init=10)
        new_labels = new_kmeans.fit_predict(embeddings)
        
        return new_kmeans, new_labels
