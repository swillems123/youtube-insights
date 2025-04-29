#!/usr/bin/env python
"""
Topic Deep Analyzer: Provides in-depth analysis of specific YouTube video topics
"""

import pandas as pd
import logging
import argparse
import json
import re
import numpy as np
from collections import Counter
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, ENGLISH_STOP_WORDS

# Local imports
from .utils.csv_reader import read_csv
from .utils.data_cleaner import clean_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Additional stopwords beyond sklearn's defaults
ADDITIONAL_STOPWORDS = [
    'video', 'youtube', 'subscribe', 'channel', 'like', 'comment', 'share',
    'watch', 'click', 'http', 'https', 'www', 'com', 'org', 'net', 'link',
    'follow', 'check', 'website', 'today', 'new', 'going', 'make', 'know',
    'please', 'thanks', 'thank', 'hello', 'hey', 'hi', 'right', 'guys',
    'yeah', 'yes', 'no', 'okay', 'ok', 'amp', 'get', 'one', 'two', 'three',
    'next', 'last', 'first', 'second', 'third', 'let', 'got', 'put', 'way',
    'really', 'say', 'said', 'go', 'going', 'gone', 'gonna', 'wanna'
]

# Words to explicitly filter out
FILTER_WORDS = [
    'nbsp', 'music', 'audio', 'sound', 'applause', 'laughter', 'background',
    'um', 'uh', 'ah', 'oh', 'hmm', 'eh', 'mm', 'er', 'err', 'huh', 'foreign'
]

def clean_text(text):
    """Clean text by removing special characters and extra whitespace."""
    if not isinstance(text, str):
        return ""
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove special characters except letters and spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Filter out specific words
    for word in FILTER_WORDS:
        text = re.sub(r'\b' + re.escape(word) + r'\b', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def analyze_topic_deeply(df, topic_id):
    """
    Provide deep analysis of a specific topic.
    
    Args:
        df: Pandas DataFrame containing YouTube data
        topic_id: Topic ID to analyze
    
    Returns:
        Dictionary containing the deep analysis
    """
    if 'topic' not in df.columns:
        return {"error": "No topic column found in the dataset"}
    
    # Filter data for this topic
    topic_data = df[df['topic'] == topic_id]
    
    if len(topic_data) < 2:
        return {"error": f"Not enough data for topic {topic_id}"}
    
    # Prepare data
    if 'text' in df.columns:
        topic_data['cleaned_text'] = topic_data['text'].astype(str).apply(clean_text)
    else:
        topic_data['cleaned_text'] = ""
    
    if 'description' in df.columns:
        topic_data['cleaned_desc'] = topic_data['description'].astype(str).apply(clean_text)
    else:
        topic_data['cleaned_desc'] = ""
    
    if 'title' in df.columns:
        topic_data['cleaned_title'] = topic_data['title'].astype(str).apply(clean_text)
    else:
        topic_data['cleaned_title'] = ""
    
    # 1. Get key phrases (3-4 words) from text
    all_stopwords = list(ENGLISH_STOP_WORDS) + ADDITIONAL_STOPWORDS
    
    combined_text = ' '.join(topic_data['cleaned_text'].tolist())
    combined_desc = ' '.join(topic_data['cleaned_desc'].tolist())
    combined_title = ' '.join(topic_data['cleaned_title'].tolist())
    
    # Analysis results
    analysis = {
        "topic_id": str(topic_id),
        "video_count": len(topic_data),
        "key_phrases": [],
        "key_words": [],
        "title_themes": [],
        "description_themes": [],
        "sentiment": None,
        "performance_metrics": {},
        "summary": ""
    }
    
    # 2. Extract key phrases
    try:
        # Get phrases from text
        phrase_vectorizer = CountVectorizer(
            ngram_range=(3, 4),
            stop_words=all_stopwords,
            min_df=2,
            max_features=15
        )
        text_corpus = topic_data['cleaned_text'].tolist()
        if len(combined_text) > 100:  # Only if there's enough text
            phrase_matrix = phrase_vectorizer.fit_transform(text_corpus)
            phrases = phrase_vectorizer.get_feature_names_out()
            phrase_freq = phrase_matrix.sum(axis=0).A1
            top_phrases = sorted(zip(phrases, phrase_freq), key=lambda x: x[1], reverse=True)
            analysis["key_phrases"] = [phrase for phrase, freq in top_phrases[:5]]
    except Exception as e:
        logger.warning(f"Error extracting key phrases: {e}")
    
    # 3. Extract key words
    try:
        # Get distinctive words
        word_vectorizer = TfidfVectorizer(
            ngram_range=(1, 1),
            stop_words=all_stopwords,
            min_df=2,
            max_features=30
        )
        if len(combined_text) > 50:
            word_matrix = word_vectorizer.fit_transform(text_corpus)
            words = word_vectorizer.get_feature_names_out()
            word_scores = word_matrix.mean(axis=0).A1
            top_words = sorted(zip(words, word_scores), key=lambda x: x[1], reverse=True)
            analysis["key_words"] = [word for word, score in top_words[:15]]
    except Exception as e:
        logger.warning(f"Error extracting key words: {e}")
    
    # 4. Extract title themes
    try:
        title_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words=all_stopwords,
            min_df=1,
            max_features=20
        )
        title_corpus = topic_data['cleaned_title'].tolist()
        if len(combined_title) > 20:
            title_matrix = title_vectorizer.fit_transform(title_corpus)
            title_terms = title_vectorizer.get_feature_names_out()
            title_scores = title_matrix.mean(axis=0).A1
            top_title_terms = sorted(zip(title_terms, title_scores), key=lambda x: x[1], reverse=True)
            analysis["title_themes"] = [term for term, score in top_title_terms[:8]]
    except Exception as e:
        logger.warning(f"Error extracting title themes: {e}")
    
    # 5. Calculate sentiment
    try:
        sentiments = []
        for text in topic_data['cleaned_text']:
            if len(text) > 20:
                sentiments.append(TextBlob(text).sentiment.polarity)
        
        if sentiments:
            avg_sentiment = sum(sentiments) / len(sentiments)
            analysis["sentiment"] = {
                "polarity": avg_sentiment,
                "description": "positive" if avg_sentiment > 0.1 else "negative" if avg_sentiment < -0.1 else "neutral"
            }
    except Exception as e:
        logger.warning(f"Error calculating sentiment: {e}")
    
    # 6. Performance metrics
    try:
        metrics = {}
        if 'viewCount' in topic_data.columns:
            metrics["avg_views"] = float(topic_data['viewCount'].mean())
            
        if 'likeCount' in topic_data.columns:
            metrics["avg_likes"] = float(topic_data['likeCount'].mean())
            
        if 'commentCount' in topic_data.columns:
            metrics["avg_comments"] = float(topic_data['commentCount'].mean())
            
        if 'engagementRate' in topic_data.columns:
            metrics["avg_engagement"] = float(topic_data['engagementRate'].mean())
            
        if 'duration' in topic_data.columns:
            metrics["avg_duration"] = topic_data['duration'].mean()
            
        analysis["performance_metrics"] = metrics
    except Exception as e:
        logger.warning(f"Error calculating performance metrics: {e}")
    
    # 7. Generate summary
    try:
        summary = f"Topic {topic_id} contains {len(topic_data)} videos. "
        
        # Add theme information
        if analysis["title_themes"]:
            summary += f"Common themes include {', '.join(analysis['title_themes'][:3])}. "
            
        # Add key words if available
        if analysis["key_words"]:
            summary += f"Frequently used keywords: {', '.join(analysis['key_words'][:5])}. "
            
        # Add performance info
        if "avg_views" in analysis["performance_metrics"]:
            summary += f"Videos average {int(analysis['performance_metrics']['avg_views'])} views. "
            
        # Add sentiment
        if analysis["sentiment"]:
            summary += f"Content has overall {analysis['sentiment']['description']} sentiment. "
            
        # Add distinctive phrases if available
        if analysis["key_phrases"]:
            summary += f"Representative phrases include: \"{analysis['key_phrases'][0]}\""
            
        analysis["summary"] = summary
    except Exception as e:
        logger.warning(f"Error generating summary: {e}")
        analysis["summary"] = f"Topic {topic_id} contains {len(topic_data)} videos."
    
    return analysis

def main():
    """Main function to parse arguments and run the deep topic analysis."""
    parser = argparse.ArgumentParser(description='Deeply analyze specific YouTube video topics')
    parser.add_argument('csv_file', help='Path to YouTube CSV data file')
    parser.add_argument('--topics', '-t', nargs='+', required=True, help='Topic IDs to analyze deeply')
    parser.add_argument('--output', '-o', help='Output JSON file path (optional)')
    
    args = parser.parse_args()
    
    # Convert topic IDs to appropriate type
    try:
        topic_ids = [int(t) if t.isdigit() else t for t in args.topics]
    except:
        topic_ids = args.topics
    
    # Read and process the CSV file
    logger.info(f"Reading data from {args.csv_file}")
    raw_data = read_csv(args.csv_file)
    cleaned_data = clean_data(raw_data)
    df = pd.DataFrame(cleaned_data)
    
    # Generate deep analysis for each requested topic
    logger.info(f"Analyzing topics: {', '.join(map(str, topic_ids))}")
    
    deep_analysis = {}
    for topic_id in topic_ids:
        logger.info(f"Analyzing topic {topic_id}")
        deep_analysis[str(topic_id)] = analyze_topic_deeply(df, topic_id)
    
    # Print the results
    if deep_analysis:
        logger.info(f"Generated deep analysis for {len(deep_analysis)} topics")
        result = {"topic_deep_analysis": deep_analysis}
        
        if args.output:
            # Save to JSON file
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"Results saved to {args.output}")
        else:
            # Print to console
            print(json.dumps(result, indent=2))
    else:
        logger.warning("No topic analysis generated")

if __name__ == "__main__":
    main()