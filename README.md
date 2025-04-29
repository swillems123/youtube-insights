# YouTube Insights

A comprehensive analytics toolkit for YouTube content creators to analyze video performance, audience engagement patterns, and content optimization opportunities.

## Features

- **Hook Sentiment Analysis**: Analyze emotional content in video hooks and correlate with performance metrics
- **Video Performance Metrics**: Compare metrics across videos to identify top performers
- **Time Series Analysis**: Track performance trends over time
- **Topic Analysis**: Analyze performance by content topic
- **Audience Engagement Analysis**: Identify patterns in viewer engagement
- **Visualization Generation**: Create charts and graphs for easier pattern recognition

## Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd youtube-insights
   ```

2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Dependencies

The project requires the following Python packages:

```
pandas
numpy
matplotlib
seaborn
plotly
wordcloud
nltk
textblob
nrclex
scikit-learn>=1.0.2
```

## Usage

### Preparing Your Data

This toolkit expects a CSV file containing YouTube video data with at least the following columns:
- `title`: Video title
- `hook`: The hook/intro text of the video (for hook sentiment analysis)
- `viewCount`: Number of views
- `likeCount`: Number of likes
- `commentCount`: Number of comments
- `publishedAt`: Publish date/time (for time series analysis)

Other optional columns:
- `topic`: Video topic/category
- `duration`: Video duration
- `description`: Video description

### Running Hook Sentiment Analysis

Analyze emotions in video hooks and correlate with performance metrics:

```
python src/hook_sentiment_analyzer.py path/to/your-data.csv
```

This will:
1. Detect emotions in the hook text using NRCLex
2. Calculate performance metrics by emotion
3. Create visualizations in the `hook_sentiment_analysis` folder
4. Display a summary of results in the terminal

### Generating Performance Graphs

Create comprehensive performance visualizations:

```
python graph_generator.py path/to/your-data.csv --visualize
```

Optional arguments:
- `--output-dir DIRECTORY`: Specify custom output directory (default: "visualizations")

This will create a variety of visualizations analyzing different aspects of your YouTube performance data.

## Visualization Outputs

### Hook Sentiment Analysis

- `avg_views_by_emotion.png`: Bar chart showing average views by hook emotion
- `view_distribution_by_emotion.png`: Box plot of view distribution by emotion
- `engagement_by_emotion.png`: Engagement ratio (likes/views) by emotion
- `hook_length_vs_views.png`: Scatter plot of hook length vs views by emotion
- `emotion_distribution.png`: Pie chart showing distribution of emotions in hooks

### Graph Generator

The `graph_generator.py` script produces a variety of visualizations in the specified output directory:
- Time series analysis of key metrics
- Video performance comparisons
- Topic analysis visualizations
- Text content analysis
- Audience engagement patterns

## Topic Labels

For topic analysis, the system attempts to load topic labels from a `topic_labels.json` file. This file should contain a mapping of topic IDs to human-readable labels:

```json
{
  "topic_labels": {
    "1": "Technology",
    "2": "Gaming",
    "3": "Education",
    ...
  }
}
```

## Advanced Usage

### Custom Analysis Modules

The project is organized to allow for custom analysis modules in the `src` directory:
- `src/analyzer.py`: General video analysis
- `src/hook_sentiment_analyzer.py`: Hook emotion analysis
- `src/video_emotion_analyzer.py`: Overall video emotion analysis
- `src/topic_deep_analyzer.py`: In-depth topic analysis

### Utility Modules

Reusable components are located in the `src/utils` directory:
- `csv_reader.py`: CSV data loading utilities
- `data_cleaner.py`: Data preprocessing functions
- `metrics.py`: Metric calculation functions

## License

[Specify your license information here]

## Contributing

[Contribution guidelines if applicable]
