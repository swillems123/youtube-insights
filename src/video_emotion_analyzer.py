import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nrclex import NRCLex
import logging
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_video_emotion_performance(csv_path, text_column='transcript'):
    """
    Analyze which emotional videos perform best by view count
    
    Args:
        csv_path: Path to CSV file with video data
        text_column: Column containing video text (transcript, description, etc.)
    """
    logger.info(f"Analyzing video emotional performance in {csv_path}")
    
    # Read CSV file
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} videos")
    except FileNotFoundError:
        logger.error(f"CSV file not found at: {csv_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        raise
    
    # Ensure numeric columns
    if 'viewCount' not in df.columns:
        logger.error("Missing required column: viewCount")
        raise ValueError("Missing required column: viewCount")
    df['viewCount'] = pd.to_numeric(df['viewCount'], errors='coerce').fillna(0)
    
    # Check if text column exists
    if text_column not in df.columns:
        logger.error(f"No '{text_column}' column found in CSV")
        raise ValueError(f"Missing required column: {text_column}")
    
    # Function to extract the dominant emotion from text
    def get_dominant_emotion(text):
        if not isinstance(text, str) or not text.strip():
            return "unknown"
        
        try:
            # Process with NRCLex
            emotion_analyzer = NRCLex(text)
            # Get raw counts of emotion words (not just frequencies)
            emotion_counts = emotion_analyzer.raw_emotion_scores
            # Remove positive/negative categories
            emotion_counts = {k: v for k, v in emotion_counts.items() 
                             if k not in ['positive', 'negative'] and v > 0}
            
            if not emotion_counts:
                return "neutral"
                
            # Return emotion with most word occurrences
            return max(emotion_counts, key=emotion_counts.get)
        except Exception as e:
            # Log specific error for debugging
            logger.warning(f"Error processing text chunk: {str(e)}. Text starts with: '{text[:50]}...'")
            return "unknown"
    
    # Apply emotion extraction to each video
    logger.info(f"Categorizing videos by dominant emotion using column '{text_column}'...")
    df['dominant_emotion'] = df[text_column].apply(get_dominant_emotion)
    logger.info("Finished categorizing videos.")
    
    # Create output directory
    output_dir = 'video_emotion_analysis'
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output will be saved to '{output_dir}'")
    
    # Calculate metrics by emotion
    logger.info("Calculating performance metrics by emotion...")
    # Handle potential empty groups if some emotions aren't found
    if df['dominant_emotion'].nunique() > 0:
        emotion_performance = df.groupby('dominant_emotion').agg(
            avg_views=('viewCount', 'mean'),
            median_views=('viewCount', 'median'),
            total_views=('viewCount', 'sum'),
            video_count=('viewCount', 'count') # Use a column guaranteed to exist
        ).reset_index()
    else:
        logger.warning("No dominant emotions found to group by.")
        emotion_performance = pd.DataFrame(columns=['dominant_emotion', 'avg_views', 'median_views', 'total_views', 'video_count'])

    # Sort emotions by average view count
    top_emotions = emotion_performance.sort_values(by='avg_views', ascending=False)
    
    # Create visualizations
    logger.info("Creating visualizations...")
    
    # Define a color palette
    unique_emotions = df['dominant_emotion'].unique()
    emotion_categories = [e for e in unique_emotions if e not in ['unknown', 'neutral']]
    if emotion_categories:
        base_palette = plt.get_cmap('tab10', len(emotion_categories))
        emotion_colors = {emotion: base_palette(i) for i, emotion in enumerate(emotion_categories)}
    else:
        emotion_colors = {}
    emotion_colors['neutral'] = '#D3D3D3'
    emotion_colors['unknown'] = '#808080'
    
    # Map colors for plotting, handling missing emotions
    plot_palette = {e: emotion_colors.get(e, '#A9A9A9') for e in top_emotions['dominant_emotion']}

    # 1. Bar chart of average views by emotion
    plt.figure(figsize=(12, 8))
    sns.barplot(x='dominant_emotion', y='avg_views', data=top_emotions, 
                palette=plot_palette, order=top_emotions['dominant_emotion'])
    plt.title('Average Views by Dominant Video Emotion', fontsize=16)
    plt.ylabel('Average Views', fontsize=14)
    plt.xlabel('Dominant Emotion', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/avg_views_by_emotion.png", dpi=300)
    plt.close()
    
    # 2. Box plot for distribution
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='dominant_emotion', y='viewCount', data=df,
                order=top_emotions['dominant_emotion'], palette=plot_palette, showfliers=False)
    plt.title('View Count Distribution by Dominant Emotion (Outliers Hidden)', fontsize=16)
    plt.ylabel('View Count', fontsize=14)
    plt.xlabel('Dominant Emotion', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/view_distribution_by_emotion.png", dpi=300)
    plt.close()
    
    logger.info("Visualizations created.")
    
    # Return results
    return {
        'emotion_counts': df['dominant_emotion'].value_counts().to_dict(),
        'emotion_performance': top_emotions.set_index('dominant_emotion').to_dict('index'),
        'output_dir': output_dir
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze which emotional videos perform best based on text analysis.')
    parser.add_argument('csv_file', help='Path to CSV file with video data (including view counts and text column).')
    parser.add_argument('--text-column', default='transcript', 
                      help='Column name containing the video text (e.g., transcript, description) to analyze (default: transcript)')
    args = parser.parse_args()
    
    try:
        results = analyze_video_emotion_performance(args.csv_file, args.text_column)
        
        # Print summary
        print("\n===== VIDEO EMOTION PERFORMANCE SUMMARY =====\n")
        print("VIDEOS BY DOMINANT EMOTION:")
        # Sort counts for printing
        sorted_counts = sorted(results['emotion_counts'].items(), key=lambda x: x[1], reverse=True)
        for emotion, count in sorted_counts:
            print(f"  {emotion}: {count} videos")
        
        print("\nPERFORMANCE BY DOMINANT EMOTION (Sorted by Avg Views):")
        performance = results['emotion_performance']
        # Results are already sorted
        for emotion, metrics in performance.items():
            print(f"\n  {emotion.upper()}:")
            print(f"    Avg Views: {metrics['avg_views']:.1f}")
            print(f"    Median Views: {metrics['median_views']:.1f}")
            print(f"    Total Views: {metrics['total_views']:,}")
            print(f"    Video Count: {metrics['video_count']}")
            
        print(f"\nVisualizations saved in '{results['output_dir']}' folder.")
        
    except FileNotFoundError:
        print(f"Error: Input CSV file not found at '{args.csv_file}'", file=sys.stderr)
        sys.exit(1)
    except ValueError as ve:
        print(f"Error: {ve}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
