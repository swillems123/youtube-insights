#!/usr/bin/env python
"""
Hook Emotion Analyzer: Analyzes emotions in YouTube video hooks and compares with performance metrics
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import logging
import argparse
from nrclex import NRCLex # Import NRCLex

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure NLTK resources are available
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    logger.info("Downloading NLTK resources for sentiment analysis")
    nltk.download('vader_lexicon')

def detect_emotion(text):
    """
    Detect primary emotion in text using NRCLex.
    Returns the dominant emotion (e.g., joy, anger, sadness) or 'neutral'/'unknown'.
    """
    if not isinstance(text, str) or not text.strip():
        return "unknown"

    try:
        emotion = NRCLex(text)
        # Get the scores for each emotion
        scores = emotion.affect_frequencies
        
        # Filter out positive/negative polarity scores if present, focus on emotions
        emotion_scores = {k: v for k, v in scores.items() if k not in ['positive', 'negative'] and v > 0}
        
        if not emotion_scores: # No specific emotion detected
            # Fallback to simple polarity if needed, or just return neutral
            # polarity = TextBlob(text).sentiment.polarity
            # if polarity > 0.1: return 'positive_overall'
            # if polarity < -0.1: return 'negative_overall'
            return "neutral"
            
        # Find the emotion with the highest score
        dominant_emotion = max(emotion_scores, key=emotion_scores.get)
        return dominant_emotion
        
    except Exception as e:
        # logger.warning(f"Could not process text for emotion: '{text[:50]}...' Error: {e}")
        return "unknown" # Return unknown if NRCLex fails

def analyze_hook_sentiment(csv_path):
    """
    Analyze sentiment in hook column and compare with view/like metrics
    """
    logger.info(f"Analyzing hook sentiment in {csv_path}")
    
    # Read CSV file
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Successfully loaded {len(df)} rows from {csv_path}")
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        return None
    
    # Check if hook column exists
    if 'hook' not in df.columns:
        logger.error("No 'hook' column found in CSV")
        return None
    
    # Ensure numeric columns are properly formatted
    for col in ['viewCount', 'likeCount']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Apply sentiment analysis to hook column
    logger.info("Analyzing sentiment in hooks")
    df['hook_sentiment'] = df['hook'].apply(detect_emotion)
    
    # Calculate additional metrics
    df['hook_length'] = df['hook'].astype(str).apply(len)
    
    # Generate results
    results = {
        'sentiment_counts': df['hook_sentiment'].value_counts().to_dict(),
        'sentiment_performance': {}
    }
    
    # Calculate performance metrics by sentiment group
    sentiment_groups = df.groupby('hook_sentiment')
    
    # For each sentiment category, calculate performance metrics
    for sentiment, group in sentiment_groups:
        results['sentiment_performance'][sentiment] = {
            'count': len(group),
            'avg_views': group['viewCount'].mean(),
            'median_views': group['viewCount'].median(),
            'total_views': group['viewCount'].sum(),
            'avg_likes': group['likeCount'].mean(),
            'median_likes': group['likeCount'].median(),
            'avg_hook_length': group['hook_length'].mean()
        }
    
    # Create visualizations
    create_visualizations(df, 'hook_sentiment_analysis')
    
    return results

def create_visualizations(df, output_dir='hook_sentiment_analysis'):
    """Create visualizations for hook emotion analysis"""
    logger.info(f"Creating visualizations in {output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define a color palette suitable for multiple emotions
    # Using a qualitative palette like 'tab10' or 'Set3'
    unique_emotions = df['hook_sentiment'].unique()
    # Exclude 'unknown' and 'neutral' if they exist for specific coloring
    emotion_categories = [e for e in unique_emotions if e not in ['unknown', 'neutral']]
    base_palette = plt.get_cmap('tab10', len(emotion_categories))
    emotion_colors = {emotion: base_palette(i) for i, emotion in enumerate(emotion_categories)}
    emotion_colors['neutral'] = '#D3D3D3' # Light grey for neutral
    emotion_colors['unknown'] = '#808080' # Darker grey for unknown
    
    # 1. Bar chart of average views by emotion
    plt.figure(figsize=(12, 7))
    emotion_view_data = df.groupby('hook_sentiment')['viewCount'].mean().sort_values(ascending=False)
    
    # Map colors, defaulting to grey if an emotion isn't in the palette
    colors = [emotion_colors.get(s, '#A9A9A9') for s in emotion_view_data.index]
    ax = emotion_view_data.plot(kind='bar', color=colors)
    
    plt.title('Average Views by Hook Emotion', fontsize=16)
    plt.ylabel('Average Views', fontsize=14)
    plt.xlabel('Hook Emotion', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/avg_views_by_emotion.png", dpi=300)
    plt.close()
    
    # 2. Box plot of view distribution by emotion
    plt.figure(figsize=(14, 8))
    # Create a sorted order for the x-axis
    order = sorted([e for e in df['hook_sentiment'].unique() if e != 'unknown'] + ['unknown'])
    sns.boxplot(x='hook_sentiment', y='viewCount', data=df, order=order, palette=emotion_colors, showfliers=False) # Hide outliers for clarity
    plt.title('View Count Distribution by Hook Emotion (excluding outliers)', fontsize=16)
    plt.ylabel('View Count', fontsize=14)
    plt.xlabel('Hook Emotion', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/view_distribution_by_emotion.png", dpi=300)
    plt.close()
    
    # 3. Engagement ratio (likes/views) by emotion 
    df['engagement_ratio'] = (df['likeCount'] / df['viewCount'].replace(0, np.nan)) * 100
    
    plt.figure(figsize=(12, 7))
    engagement_data = df.groupby('hook_sentiment')['engagement_ratio'].mean().sort_values(ascending=False)
    
    colors = [emotion_colors.get(s, '#A9A9A9') for s in engagement_data.index]
    engagement_data.plot(kind='bar', color=colors)
    
    plt.title('Likes-to-Views Ratio by Hook Emotion (%)', fontsize=16)
    plt.ylabel('Likes/Views Ratio (%)', fontsize=14)
    plt.xlabel('Hook Emotion', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/engagement_by_emotion.png", dpi=300)
    plt.close()
    
    # 4. Scatter plot of hook length vs views, colored by emotion
    plt.figure(figsize=(14, 9))
    # Use Seaborn for easier legend handling with many categories
    sns.scatterplot(data=df, x='hook_length', y='viewCount', hue='hook_sentiment', 
                    palette=emotion_colors, alpha=0.6, s=50) # s is marker size
    
    plt.title('Hook Length vs Views by Emotion', fontsize=16)
    plt.xlabel('Hook Length (characters)', fontsize=14)
    plt.ylabel('View Count', fontsize=14)
    plt.legend(title='Emotion', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
    plt.savefig(f"{output_dir}/hook_length_vs_views.png", dpi=300)
    plt.close()
    
    # 5. Pie chart showing distribution of emotions
    plt.figure(figsize=(10, 10))
    emotion_counts = df['hook_sentiment'].value_counts()
    colors = [emotion_colors.get(s, '#A9A9A9') for s in emotion_counts.index]
    
    plt.pie(emotion_counts, labels=emotion_counts.index, autopct='%1.1f%%', 
            startangle=90, colors=colors, wedgeprops={'edgecolor': 'white'})
    plt.title('Distribution of Hook Emotions', fontsize=16)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/emotion_distribution.png", dpi=300)
    plt.close()
    
    logger.info(f"Visualizations created successfully in {output_dir}")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze hook sentiment in YouTube CSV')
    parser.add_argument('csv_file', help='Path to YouTube CSV file')
    args = parser.parse_args()
    
    try:
        results = analyze_hook_sentiment(args.csv_file)
        if results:
            # Print summary of results
            print("\n===== HOOK EMOTION ANALYSIS SUMMARY =====\n") # Updated title
            
            print("EMOTION DISTRIBUTION:") # Updated title
            # Sort emotions for consistent printing
            sorted_emotions = sorted(results['sentiment_counts'].items(), key=lambda item: item[1], reverse=True)
            for emotion, count in sorted_emotions:
                print(f"  {emotion}: {count} videos")
            
            print("\nPERFORMANCE BY EMOTION:") # Updated title
            # Sort performance results for consistent printing
            sorted_performance = sorted(results['sentiment_performance'].items(), key=lambda item: item[1]['avg_views'], reverse=True)
            for emotion, metrics in sorted_performance:
                print(f"\n  {emotion.upper()}:")
                print(f"    Count: {metrics['count']} videos")
                print(f"    Avg Views: {metrics['avg_views']:.1f}")
                print(f"    Avg Likes: {metrics['avg_likes']:.1f}")
                print(f"    Avg Hook Length: {metrics['avg_hook_length']:.1f} chars")
            
            print("\nVisualizations created in 'hook_sentiment_analysis' folder")
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
