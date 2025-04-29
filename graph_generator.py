#!/usr/bin/env python
"""
YouTube Graph Generator: Creates visualization-ready data points from YouTube insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import json
import os
import sys
import logging
import traceback
from wordcloud import WordCloud
import matplotlib.cm as cm
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("graph_generator.log"), logging.StreamHandler()])
logger = logging.getLogger(__name__)

# Fix import path for local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Try to import local modules with better error handling
try:
    from src.analyzer import analyze_videos
    from src.per_video_analyzer import analyze_videos_per_video
    from src.topic_deep_analyzer import analyze_topic_deeply
    from src.utils.metrics import calculate_engagement_rate, calculate_average_views, calculate_like_to_view_ratio
    logger.info("Successfully imported local modules")
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    traceback.print_exc()
    sys.exit(1)

def load_topic_labels():
    """Load topic labels from JSON file"""
    try:
        with open('topic_labels.json', 'r') as f:
            data = json.load(f)
            return data.get('topic_labels', {})
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Could not load topic labels: {e}")
        return {}

def create_time_series_analysis(df):
    """Create time series analysis of key metrics"""
    logger.info("Creating time series analysis")
    
    # Ensure publishedAt is in datetime format
    if 'publishedAt' in df.columns:
        df['publishedAt'] = pd.to_datetime(df['publishedAt'], errors='coerce')
        df = df.sort_values('publishedAt')
        df = df.dropna(subset=['publishedAt'])  # Drop rows with NaN publishedAt
        
        # Create dates directly from publishedAt
        df['date'] = df['publishedAt'].dt.date
        df['month'] = df['publishedAt'].dt.to_period('M')
        df['week'] = df['publishedAt'].dt.to_period('W')
        
        # Create time-based metrics - monthly
        monthly_metrics = df.groupby(df['month'].astype(str)).agg({
            'viewCount': 'mean',
            'likeCount': 'mean',
            'commentCount': 'mean',
            'engagementRate': 'mean'
        }).reset_index()
        monthly_metrics['date'] = pd.to_datetime(monthly_metrics['month'] + '-01')
        
        # Create time-based metrics - weekly - Fix the date parsing
        weekly_metrics = df.groupby(df['week'].astype(str)).agg({
            'viewCount': 'mean',
            'likeCount': 'mean',
            'commentCount': 'mean',
            'engagementRate': 'mean'
        }).reset_index()
        
        # Use a safer approach to extract dates from period objects
        try:
            # Extract the first day of each week directly from the period
            weekly_metrics['week_start'] = weekly_metrics['week'].apply(
                lambda w: pd.Period(w).start_time.date())
            weekly_metrics['date'] = pd.to_datetime(weekly_metrics['week_start'])
        except Exception as e:
            logger.warning(f"Could not parse week dates: {e}")
            # Fallback to using the month data timestamps
            weekly_metrics['date'] = pd.to_datetime('2000-01-01')  # Default date
        
        return {
            'weekly_metrics': weekly_metrics.to_dict('records'),
            'monthly_metrics': monthly_metrics.to_dict('records')
        }
    
    logger.warning("No publishedAt column found for time series analysis")
    return {}

def create_video_performance_comparison(df):
    """Compare performance metrics across videos"""
    logger.info("Creating video performance comparison")
    
    # Ensure numeric columns are actually numeric
    for col in ['viewCount', 'likeCount', 'commentCount', 'engagementRate']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Get top and bottom performing videos by different metrics
    top_views = df.nlargest(10, 'viewCount')[['title', 'viewCount']].fillna({'title': 'Unknown'})
    
    if 'engagementRate' in df.columns:
        top_engagement = df.nlargest(10, 'engagementRate')[['title', 'engagementRate', 'viewCount']].fillna({'title': 'Unknown'})
    else:
        top_engagement = pd.DataFrame(columns=['title', 'engagementRate', 'viewCount'])
    
    top_likes = df.nlargest(10, 'likeCount')[['title', 'likeCount', 'viewCount']].fillna({'title': 'Unknown'})
    
    # Get length distribution of titles
    if 'title' in df.columns:
        df['title_length'] = df['title'].astype(str).apply(len)
        title_vs_performance = {
            'title_length': df['title_length'].tolist(),
            'views': df['viewCount'].tolist(),
            'engagement': df['engagementRate'].tolist() if 'engagementRate' in df.columns else []
        }
    else:
        title_vs_performance = {}
    
    # Get correlations between metrics - select only numeric columns
    numeric_cols = ['viewCount', 'likeCount', 'commentCount']
    if 'engagementRate' in df.columns and pd.api.types.is_numeric_dtype(df['engagementRate']):
        numeric_cols.append('engagementRate')
    
    correlation_matrix = df[numeric_cols].corr().to_dict()
    
    return {
        'top_views': top_views.to_dict('records'),
        'top_engagement': top_engagement.to_dict('records'),
        'top_likes': top_likes.to_dict('records'),
        'title_vs_performance': title_vs_performance,
        'correlation_matrix': correlation_matrix
    }

def create_topic_analysis(df, topic_labels):
    """Analyze performance by topic"""
    logger.info("Creating topic analysis")
    
    if 'topic' not in df.columns:
        logger.warning("No topic column found for topic analysis")
        return {}
    
    # Ensure numeric columns are actually numeric
    for col in ['viewCount', 'likeCount', 'commentCount', 'engagementRate']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Get performance metrics by topic
    try:
        topic_metrics = df.groupby('topic').agg({
            'viewCount': ['mean', 'sum', 'count'],
            'likeCount': ['mean', 'sum'],
            'commentCount': ['mean', 'sum'],
            'engagementRate': 'mean'
        }).reset_index()
        
        # Flatten the multi-level columns
        topic_metrics.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in topic_metrics.columns.values]
        
        # Add topic labels
        topic_metrics['topic_label'] = topic_metrics['topic'].map(
            lambda x: topic_labels.get(str(x), f"Topic {x}")
        )
        
        # Calculate normalized metrics for visual comparison
        numeric_cols = ['viewCount_mean', 'likeCount_mean', 'commentCount_mean', 'engagementRate_mean']
        available_cols = [col for col in numeric_cols if col in topic_metrics.columns]
        
        if not topic_metrics.empty and len(available_cols) > 0:
            # Handle missing columns
            for col in numeric_cols:
                if col not in topic_metrics.columns:
                    topic_metrics[col] = 0
            
            # Apply normalization for each column separately
            for col in numeric_cols:
                max_val = topic_metrics[col].max()
                min_val = topic_metrics[col].min()
                if max_val > min_val:
                    topic_metrics[f"{col}_normalized"] = (topic_metrics[col] - min_val) / (max_val - min_val)
                else:
                    topic_metrics[f"{col}_normalized"] = 0
    except Exception as e:
        logger.error(f"Error in topic grouping: {e}")
        traceback.print_exc()
        topic_metrics = pd.DataFrame()
    
    return {
        'topic_metrics': topic_metrics.to_dict('records') if not topic_metrics.empty else [],
        'topic_distribution': df['topic'].value_counts().to_dict()
    }

def create_text_content_analysis(df):
    """Analyze text content of videos"""
    logger.info("Creating text content analysis")
    
    text_analysis = {}
    
    # Ensure engagementRate is numeric
    if 'engagementRate' in df.columns:
        df['engagementRate'] = pd.to_numeric(df['engagementRate'], errors='coerce')
    
    # Title word count vs. performance
    if 'title' in df.columns:
        df['title_word_count'] = df['title'].astype(str).apply(lambda x: len(x.split()))
        text_analysis['title_word_counts'] = {
            'word_counts': df['title_word_count'].tolist(),
            'engagement': df['engagementRate'].tolist() if 'engagementRate' in df.columns else [],
            'views': df['viewCount'].tolist()
        }
    
    # Extract most common words in titles of top performing videos
    if 'title' in df.columns and 'engagementRate' in df.columns and df['engagementRate'].notna().any():
        # Get top and bottom performers with valid engagement rates
        df_valid = df.dropna(subset=['engagementRate'])
        if len(df_valid) > 0:
            top_count = min(20, len(df_valid))
            top_performing = df_valid.nlargest(top_count, 'engagementRate')
            all_words = ' '.join(top_performing['title'].astype(str)).lower()
            text_analysis['top_performing_words'] = all_words
            
            bottom_performing = df_valid.nsmallest(top_count, 'engagementRate')
            all_words_bottom = ' '.join(bottom_performing['title'].astype(str)).lower()
            text_analysis['bottom_performing_words'] = all_words_bottom
    
    # Sentiment vs. performance if available
    if 'sentiment' in df.columns:
        text_analysis['sentiment_vs_performance'] = {
            'sentiment': df['sentiment'].tolist(),
            'engagement': df['engagementRate'].tolist() if 'engagementRate' in df.columns else [],
            'views': df['viewCount'].tolist()
        }
    
    return text_analysis

def create_audience_engagement_analysis(df):
    """Analyze audience engagement patterns"""
    logger.info("Creating audience engagement analysis")
    
    engagement_analysis = {}
    
    # Ensure numeric columns are actually numeric
    for col in ['viewCount', 'likeCount', 'commentCount', 'engagementRate']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Calculate like to view ratio, comment to view ratio
    df['like_view_ratio'] = df['likeCount'] / df['viewCount'].replace(0, np.nan)
    df['comment_view_ratio'] = df['commentCount'] / df['viewCount'].replace(0, np.nan)
    
    engagement_analysis['ratios'] = {
        'like_view_ratio': df['like_view_ratio'].fillna(0).tolist(),
        'comment_view_ratio': df['comment_view_ratio'].fillna(0).tolist(),
        'engagement_rate': df['engagementRate'].fillna(0).tolist() if 'engagementRate' in df.columns else []
    }
    
    # Engagement by day of week if publishedAt is available
    if 'publishedAt' in df.columns:
        df['publishedAt'] = pd.to_datetime(df['publishedAt'], errors='coerce')
        df_valid = df.dropna(subset=['publishedAt'])
        
        if not df_valid.empty:
            df_valid['day_of_week'] = df_valid['publishedAt'].dt.day_name()
            engagement_by_day = df_valid.groupby('day_of_week').agg({
                'engagementRate': 'mean',
                'viewCount': 'mean'
            }).reset_index()
            
            # Reorder days of week
            days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            engagement_by_day['day_of_week'] = pd.Categorical(
                engagement_by_day['day_of_week'], 
                categories=days_order, 
                ordered=True
            )
            engagement_by_day = engagement_by_day.sort_values('day_of_week')
            
            engagement_analysis['engagement_by_day'] = engagement_by_day.to_dict('records')
    
    # If video duration is available, analyze engagement by video length
    if 'duration' in df.columns:
        try:
            df['duration_seconds'] = pd.to_numeric(df['duration'], errors='coerce')
            df_valid = df.dropna(subset=['duration_seconds'])
            
            if not df_valid.empty and df_valid['duration_seconds'].max() > 0:
                # Create duration buckets with valid ranges
                max_duration = df_valid['duration_seconds'].max()
                bins = [0, 60, 300, 600, 900, 1800]
                
                # Ensure max duration is greater than the largest bin
                if max_duration <= 1800:
                    bins = [b for b in bins if b < max_duration]
                    bins.append(max_duration)
                else:
                    bins.append(3600)
                    bins.append(max_duration)
                
                # Sort bins to ensure monotonic increase
                bins = sorted(list(set(bins)))
                
                # Create labels based on bins
                labels = []
                for i in range(len(bins)-1):
                    if i == 0:
                        labels.append(f"< {bins[i+1]/60:.1f} min")
                    else:
                        labels.append(f"{bins[i]/60:.1f}-{bins[i+1]/60:.1f} min")
                
                df_valid['duration_bucket'] = pd.cut(
                    df_valid['duration_seconds'],
                    bins=bins,
                    labels=labels
                )
                
                engagement_by_duration = df_valid.groupby('duration_bucket').agg({
                    'engagementRate': 'mean',
                    'viewCount': 'mean',
                    'title': 'count'  # Count of videos
                }).reset_index()
                
                engagement_analysis['engagement_by_duration'] = engagement_by_duration.to_dict('records')
        except Exception as e:
            logger.error(f"Error in duration analysis: {e}")
            traceback.print_exc()
    
    return engagement_analysis

def create_graphs_data(csv_file_path):
    """Main function to generate data for graphs"""
    logger.info(f"Starting graph data generation for {csv_file_path}")
    
    try:
        # Check if file exists
        if not os.path.exists(csv_file_path):
            logger.error(f"CSV file not found: {csv_file_path}")
            return {}
            
        # Load and process the data using direct function calls to avoid import issues
        logger.info("Loading and cleaning data directly")
        
        # Import locally to avoid circular imports
        from src.utils.csv_reader import read_csv
        from src.utils.data_cleaner import clean_data
        
        raw_data = read_csv(csv_file_path)
        cleaned_data = clean_data(raw_data)
        df = pd.DataFrame(cleaned_data)
        
        # Ensure numeric columns are actually numeric
        for col in ['viewCount', 'likeCount', 'commentCount']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Calculate engagement rate if not present
        if 'engagementRate' not in df.columns:
            logger.info("Calculating engagement rate")
            df['engagementRate'] = (df['likeCount'] + df['commentCount']) / df['viewCount'].replace(0, np.nan) * 100
        else:
            df['engagementRate'] = pd.to_numeric(df['engagementRate'], errors='coerce')
        
        # Load topic labels
        topic_labels = load_topic_labels()
        
        # Print DataFrame info for debugging
        logger.info(f"DataFrame shape: {df.shape}")
        logger.info(f"DataFrame columns: {df.columns.tolist()}")
        
        # Generate data for various graphs
        results = {}
        
        # Generate each section with try/except to prevent total failure
        try:
            results['time_series'] = create_time_series_analysis(df)
        except Exception as e:
            logger.error(f"Error in time series analysis: {e}")
            traceback.print_exc()
            results['time_series'] = {}
            
        try:
            results['video_performance'] = create_video_performance_comparison(df)
        except Exception as e:
            logger.error(f"Error in video performance comparison: {e}")
            traceback.print_exc()
            results['video_performance'] = {}
            
        try:
            results['topic_analysis'] = create_topic_analysis(df, topic_labels)
        except Exception as e:
            logger.error(f"Error in topic analysis: {e}")
            traceback.print_exc()
            results['topic_analysis'] = {}
            
        try:
            results['text_content'] = create_text_content_analysis(df)
        except Exception as e:
            logger.error(f"Error in text content analysis: {e}")
            traceback.print_exc()
            results['text_content'] = {}
            
        try:
            results['audience_engagement'] = create_audience_engagement_analysis(df)
        except Exception as e:
            logger.error(f"Error in audience engagement analysis: {e}")
            traceback.print_exc()
            results['audience_engagement'] = {}
        
        # Add overall metrics
        try:
            # Calculate overall metrics directly instead of using analyze_videos
            total_videos = len(df)
            total_views = df['viewCount'].sum()
            avg_views = df['viewCount'].mean()
            avg_engagement = df['engagementRate'].mean()
            
            results['overall_metrics'] = {
                'total_videos': total_videos,
                'total_views': int(total_views),
                'average_views': int(avg_views),
                'average_engagement_rate': float(avg_engagement)
            }
        except Exception as e:
            logger.error(f"Error in overall metrics calculation: {e}")
            traceback.print_exc()
            results['overall_metrics'] = {}
        
        # Save results to JSON for later visualization
        output_path = 'graph_data_points.json'
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Graph data points saved to {output_path}")
        return results
    except Exception as e:
        logger.error(f"Unexpected error in create_graphs_data: {e}")
        traceback.print_exc()
        return {}

def create_visualizations(graph_data, output_dir='visualizations'):
    """Create visualizations from graph data"""
    logger.info(f"Creating visualizations in {output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Time series visualizations (REMOVED)
    # if graph_data.get('time_series'):
    #     time_series = graph_data['time_series']
    #     
    #     if 'monthly_metrics' in time_series and time_series['monthly_metrics']:
    #         monthly_df = pd.DataFrame(time_series['monthly_metrics'])
    #         if not monthly_df.empty and 'date' in monthly_df.columns:
    #             # Create time series plot with Plotly (REMOVED)
    #             # ... code for monthly_performance.html ...
    #             pass # Placeholder
    
    # Video performance comparison
    if graph_data.get('video_performance'):
        video_perf = graph_data['video_performance']
        
        if 'correlation_matrix' in video_perf:
            # Create correlation heatmap (KEEP)
            corr_df = pd.DataFrame(video_perf['correlation_matrix'])
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_df, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
            plt.title('Correlation Between Performance Metrics')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/correlation_heatmap.png", dpi=300)
            plt.close()
        
        # Title length vs. performance scatter plot (REMOVED)
        # if 'title_vs_performance' in video_perf and video_perf['title_vs_performance']:
        #     # ... code for title_length_performance.html ...
        #     pass # Placeholder
    
    # Topic analysis (REMOVED)
    # if graph_data.get('topic_analysis') and 'topic_metrics' in graph_data['topic_analysis']:
    #     topic_data = pd.DataFrame(graph_data['topic_analysis']['topic_metrics'])
    #     
    #     if not topic_data.empty and 'topic_label' in topic_data.columns:
    #         # Create radar chart for topic comparison (REMOVED)
    #         # ... code for topic_radar_chart.html ...
    #         
    #         # Create bar chart for video count by topic (REMOVED)
    #         # ... code for topic_video_count.html ...
    #         pass # Placeholder
    
    # Text content analysis
    if graph_data.get('text_content'):
        text_data = graph_data['text_content']
        
        # Create word clouds for top performing videos (KEEP)
        if 'top_performing_words' in text_data:
            wordcloud = WordCloud(width=800, height=400, 
                                  background_color='white', 
                                  colormap='viridis',
                                  max_words=100).generate(text_data['top_performing_words'])
            
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Words in Titles of Top Performing Videos')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/top_performing_wordcloud.png", dpi=300)
            plt.close()
            
            # Also create word cloud for bottom performing videos (KEEP)
            if 'bottom_performing_words' in text_data:
                wordcloud = WordCloud(width=800, height=400, 
                                    background_color='white', 
                                    colormap='magma',
                                    max_words=100).generate(text_data['bottom_performing_words'])
                
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title('Words in Titles of Bottom Performing Videos')
                plt.tight_layout()
                plt.savefig(f"{output_dir}/bottom_performing_wordcloud.png", dpi=300)
                plt.close()
    
    # Audience engagement analysis
    if graph_data.get('audience_engagement'):
        engagement_data = graph_data['audience_engagement']
        
        # Create day of week engagement plot (KEEP)
        if 'engagement_by_day' in engagement_data:
            day_df = pd.DataFrame(engagement_data['engagement_by_day'])
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig.add_trace(
                go.Bar(x=day_df['day_of_week'], y=day_df['engagementRate'], 
                        name="Engagement Rate (%)"),
                secondary_y=False,
            )
            
            fig.add_trace(
                go.Scatter(x=day_df['day_of_week'], y=day_df['viewCount'], 
                            name="Average Views", line=dict(color='darkblue')),
                secondary_y=True,
            )
            
            fig.update_layout(
                title="Performance by Day of Week",
                xaxis_title="Day of Week",
                barmode='group'
            )
            
            fig.update_yaxes(title_text="Engagement Rate (%)", secondary_y=False)
            fig.update_yaxes(title_text="Average Views", secondary_y=True)
            
            fig.write_html(f"{output_dir}/day_of_week_performance.html")
        
        # Create video duration vs. engagement plot (REMOVED)
        # if 'engagement_by_duration' in engagement_data:
        #     # ... code for duration_performance.html ...
        #     pass # Placeholder
    
    logger.info(f"Visualizations created successfully in {output_dir}")
    return True

def create_views_correlation_visualizations(df, output_dir='visualizations'):
    """Create visualizations focusing on correlations with views"""
    logger.info(f"Creating views-focused correlation visualizations in {output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure numeric columns are actually numeric
    for col in ['viewCount', 'likeCount', 'commentCount', 'engagementRate', 'duration']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 1. Create a network graph showing correlations with views at the center (REMOVED)
    # try:
    #     # ... code for views_correlation_network.html ...
    #     pass # Placeholder
    # except Exception as e:
    #     logger.error(f"Error creating correlation network: {e}")
    #     traceback.print_exc()
    
    # 2. Create a sunburst chart showing views by topic and video duration (REMOVED)
    # try:
    #     # ... code for topic_duration_views_sunburst.html ...
    #     pass # Placeholder
    # except Exception as e:
    #     logger.error(f"Error creating sunburst chart: {e}")
    #     traceback.print_exc()
    
    # 3. Create a bubble chart showing relationship between views, engagement, and publish day (KEEP)
    try:
        if all(col in df.columns for col in ['publishedAt', 'viewCount', 'engagementRate']):
            # Extract day of week
            df['publishedAt'] = pd.to_datetime(df['publishedAt'], errors='coerce')
            df['day_of_week'] = df['publishedAt'].dt.day_name()
            
            # Group by day of week
            day_metrics = df.groupby('day_of_week').agg({
                'viewCount': 'mean',
                'engagementRate': 'mean',
                'video_id': 'count'  # Count of videos
            }).reset_index()
            
            # Create bubble chart
            fig = px.scatter(
                day_metrics,
                x='engagementRate',
                y='viewCount',
                size='video_id',  # Bubble size based on number of videos
                color='day_of_week',
                text='day_of_week',
                title='Relationship Between Views, Engagement Rate, and Day of Week',
                labels={
                    'viewCount': 'Average Views',
                    'engagementRate': 'Average Engagement Rate (%)',
                    'video_id': 'Number of Videos',
                    'day_of_week': 'Day of Week'
                }
            )
            
            fig.update_layout(width=900, height=700)
            fig.write_html(f"{output_dir}/day_views_engagement_bubble.html")
            logger.info("Created day-views-engagement bubble chart")
    except Exception as e:
        logger.error(f"Error creating bubble chart: {e}")
        traceback.print_exc()
    
    # 4. Create a heatmap of views by time of day and day of week (REMOVED - using digestible version instead)
    # try:
    #     # ... code for time_day_views_heatmap.png ...
    #     pass # Placeholder
    # except Exception as e:
    #     logger.error(f"Error creating time-day heatmap: {e}")
    #     traceback.print_exc()
    
    # 5. Create a radar chart comparing video metrics by category (REMOVED)
    # try:
    #     # ... code for category_performance_radar.html ...
    #     pass # Placeholder
    # except Exception as e:
    #     logger.error(f"Error creating category radar chart: {e}")
    #     traceback.print_exc()
    
    logger.info("Completed views-focused visualizations")
    return True

def create_digestable_time_day_heatmap(df, output_dir='visualizations'):
    """Create a digestible heatmap of views by time block and day of week"""
    logger.info("Creating digestible time-day views heatmap")
    try:
        if 'publishedAt' in df.columns:
            df['publishedAt'] = pd.to_datetime(df['publishedAt'], errors='coerce')
            df['hour_of_day'] = df['publishedAt'].dt.hour
            df['day_of_week'] = df['publishedAt'].dt.day_name()
            df['viewCount'] = pd.to_numeric(df['viewCount'], errors='coerce')
            # Group hours into blocks
            hour_bins = [0, 6, 10, 14, 18, 22, 24]
            hour_labels = ['12am-6am', '6am-10am', '10am-2pm', '2pm-6pm', '6pm-10pm', '10pm-12am']
            df['hour_block'] = pd.cut(df['hour_of_day'], bins=hour_bins, labels=hour_labels, right=False, include_lowest=True)
            # Create pivot table
            time_day_views = df.pivot_table(
                values='viewCount',
                index='day_of_week',
                columns='hour_block',
                aggfunc='mean'
            )
            days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            if all(day in time_day_views.index for day in days_order):
                time_day_views = time_day_views.reindex(days_order)
            # Create heatmap
            import matplotlib.pyplot as plt
            import seaborn as sns
            import numpy as np
            plt.figure(figsize=(12, 8))
            cmap = sns.color_palette("YlGnBu", as_cmap=True)
            ax = sns.heatmap(
                time_day_views,
                cmap=cmap,
                annot=True,
                fmt='.0f',
                linewidths=0.5,
                cbar_kws={'label': 'Average Views'},
                annot_kws={"size": 11, "weight": "bold"}
            )
            plt.xticks(rotation=0)
            # Highlight top 25% cells
            flat_data = time_day_views.to_numpy().flatten()
            flat_data = flat_data[~np.isnan(flat_data)]
            if len(flat_data) > 0:
                threshold = np.percentile(flat_data, 75)
                for i in range(time_day_views.shape[0]):
                    for j in range(time_day_views.shape[1]):
                        val = time_day_views.iloc[i, j]
                        if not np.isnan(val) and val >= threshold:
                            ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='red', lw=2, clip_on=False))
            plt.title('Digestible: Best Times to Publish (Avg Views by Day & Time Block)', fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('Hour Block', fontsize=14, labelpad=10)
            plt.ylabel('Day of Week', fontsize=14, labelpad=10)
            plt.figtext(0.5, 0.01, "Red boxes highlight the best performing time slots (top 25%)", ha="center", fontsize=12, fontstyle='italic')
            plt.tight_layout()
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(f"{output_dir}/digestable_time_day_views_heatmap.png", dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Created digestible time-day views heatmap")
            return True
    except Exception as e:
        logger.error(f"Error creating digestible time-day heatmap: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate graphable data points from YouTube insights")
    parser.add_argument('csv_file', help="Path to CSV file with YouTube data")
    parser.add_argument('--visualize', action='store_true', help="Create visualizations from the data")
    parser.add_argument('--output-dir', default='visualizations', help="Directory to save visualizations")
    
    args = parser.parse_args()
    
    try:
        # Generate data for graphs
        logger.info(f"Processing file: {args.csv_file}")
        graph_data = create_graphs_data(args.csv_file)
        
        # Create visualizations if requested
        if args.visualize and graph_data:
            logger.info(f"Creating visualizations in {args.output_dir}")
            
            # First create the regular visualizations
            create_visualizations(graph_data, args.output_dir)
            
            # Now create our new views-focused correlation visualizations
            # We need to get the DataFrame again
            try:
                # Import locally to avoid circular imports
                from src.utils.csv_reader import read_csv
                from src.utils.data_cleaner import clean_data
                
                raw_data = read_csv(args.csv_file)
                cleaned_data = clean_data(raw_data)
                df = pd.DataFrame(cleaned_data)
                
                # Ensure numeric columns are actually numeric
                for col in ['viewCount', 'likeCount', 'commentCount']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                
                # Calculate engagement rate if not present
                if 'engagementRate' not in df.columns:
                    df['engagementRate'] = (df['likeCount'] + df['commentCount']) / df['viewCount'].replace(0, np.nan) * 100
                else:
                    df['engagementRate'] = pd.to_numeric(df['engagementRate'], errors='coerce')
                
                # Create the views correlation visualizations
                create_views_correlation_visualizations(df, args.output_dir)
                logger.info("Created views-focused correlation visualizations")
                
                # *** ADD CALL TO THE NEW DIGESTIBLE HEATMAP FUNCTION HERE ***
                create_digestable_time_day_heatmap(df, args.output_dir)
                logger.info("Created digestible time-day heatmap visualization")
                
            except Exception as e:
                logger.error(f"Error creating additional visualizations: {e}") # Updated error message
                traceback.print_exc()
                
        elif args.visualize:
            logger.error("No data available for visualization")
    except Exception as e:
        logger.error(f"Unexpected error in main: {e}")
        traceback.print_exc()