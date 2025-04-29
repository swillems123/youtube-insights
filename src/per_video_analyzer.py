import logging
import pandas as pd
import numpy as np
from .utils.csv_reader import read_csv
from .utils.data_cleaner import clean_data
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
import re
import warnings
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import functools

# Suppress warnings about empty slices
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")

# Compile regex patterns once for efficiency
TIMESTAMP_PATTERN = re.compile(r'\[\d+:\d+:\d+\]|\[\d+:\d+\]')
SPECIAL_CHARS_PATTERN = re.compile(r'[^\w\s\']')
WHITESPACE_PATTERN = re.compile(r'\s+')
BRACKET_CONTENT_PATTERN = re.compile(r'[\[\(][^\]\)]*(?:music|applause|laughter|singing).*?[\]\)]', re.IGNORECASE)
NUMBERS_PATTERN = re.compile(r'\b\d+\b|\b\d+(?:st|nd|rd|th)\b')
SPEAKER_PATTERN = re.compile(r'speaker \d+|person \d+|interviewer|interviewee', re.IGNORECASE)
TIME_INDICATOR_PATTERN = re.compile(r'\d+:\d+|timestamp|timecode', re.IGNORECASE)
NON_SPEECH_PATTERN = re.compile(r'\b(?:cough|sneeze|breath|sigh|pause)\b', re.IGNORECASE)

# Music-related terms to filter out
MUSIC_TERMS = [
    'music', 'audio', 'sound', 'instrumental', 'melody', 'rhythm', 'beat', 'song', 
    'tune', 'soundtrack', 'intro music', 'outro music', 'theme music', 'background music',
    'jingle', 'applause', 'clapping', 'cheering', 'singing', 'acoustic', 'vocal', 
    'musical', 'audio', 'sound effect', 'sfx', 'drumming', 'guitar', 'piano', 'violin',
    'orchestra', 'band', 'chorus', 'choir', 'vocalist', 'singer', 'rap', 'bass'
]

# Common filler words
FILLER_WORDS = [
    'um', 'uh', 'like', 'you know', 'sort of', 'kind of', 'i mean', 'actually', 
    'basically', 'literally', 'so', 'anyway', 'well', 'right', 'just', 
    'okay', 'so yeah', 'yeah', 'erm', 'hmm', 'ah', 'oh'
]

# Common linking words
COMMON_WORDS = ['the', 'and', 'that', 'this', 'with', 'for', 'from', 'have', 'what', 
               'when', 'where', 'who', 'why', 'how', 'it', 'is', 'was', 'are', 'be']


def analyze_videos_per_video(csv_file_path):
    logger = logging.getLogger(__name__)
    logger.info(f"Loading and cleaning data from {csv_file_path}")
    raw = read_csv(csv_file_path)
    cleaned = clean_data(raw)
    df = pd.DataFrame(cleaned)

    # Ensure numeric types
    df['viewCount'] = pd.to_numeric(df.get('viewCount', 0), errors='coerce').fillna(0).astype(int)
    df['likeCount'] = pd.to_numeric(df.get('likeCount', 0), errors='coerce').fillna(0).astype(int)
    df['commentCount'] = pd.to_numeric(df.get('commentCount', 0), errors='coerce').fillna(0).astype(int)

    # Per-video metrics
    df['engagementRate'] = (df['likeCount'] + df['commentCount']) / df['viewCount'].replace(0, pd.NA) * 100
    df['likeRate'] = df['likeCount'] / df['viewCount'].replace(0, pd.NA) * 100
    df['commentRate'] = df['commentCount'] / df['viewCount'].replace(0, pd.NA) * 100

    # Text sentiment
    titles = df.get('title', '').astype(str)
    df['sentiment'] = titles.apply(lambda t: TextBlob(t).sentiment.polarity)

    return df


def compute_quartiles(df, column):
    return df[column].quantile([0.25, 0.5, 0.75]).to_dict()


def extract_top_ngrams(df, text_column, ngram_range=(2,3), top_n=10, additional_stop_words=None):
    """
    Extract top n-grams from a text column in the DataFrame.
    
    Args:
        df: DataFrame with text data
        text_column: Column name containing text to analyze
        ngram_range: Tuple defining the range of n-grams (default: (2,3))
        top_n: Number of top n-grams to return (default: 10)
        additional_stop_words: List of additional words to filter out (default: None)
        
    Returns:
        List of (ngram, frequency) tuples sorted by frequency
    """
    # Default stop words plus any additional ones provided
    stop_words = 'english'
    if additional_stop_words:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        stop_words = list(ENGLISH_STOP_WORDS) + additional_stop_words
    
    # Add music-related terms to always filter out
    if isinstance(stop_words, list):
        stop_words.extend(MUSIC_TERMS)
    else:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        stop_words = list(ENGLISH_STOP_WORDS) + MUSIC_TERMS + (additional_stop_words or [])
    
    # Clean text data using parallel processing for efficiency
    texts = df[text_column].astype(str).tolist()
    cleaned_texts = process_texts_in_parallel(texts)
    
    logging.info(f"Processing {len(cleaned_texts)} documents for {text_column} n-gram extraction")
    
    # Use efficient memory layout with sparse matrices
    vec = CountVectorizer(ngram_range=ngram_range, stop_words=stop_words, min_df=5)
    X = vec.fit_transform(cleaned_texts)
    freqs = X.sum(axis=0).A1
    terms = vec.get_feature_names_out()
    
    # Filter out n-grams that don't contain at least one meaningful word (longer than 3 chars)
    meaningful_ngrams = []
    for term, freq in zip(terms, freqs):
        words = term.split()
        # Check if it contains at least one meaningful word of 4+ chars
        if any(len(word) > 3 for word in words):
            # Skip if all words are very common linking words
            # Skip if any word is music-related
            contains_music = any(music_word in term for music_word in MUSIC_TERMS)
            if not all(word in COMMON_WORDS for word in words) and not contains_music:
                meaningful_ngrams.append((term, freq))
    
    # Eliminate redundant n-grams (subsets of other n-grams)
    final_ngrams = []
    sorted_by_length = sorted(meaningful_ngrams, key=lambda x: len(x[0]), reverse=True)
    
    for i, (term1, freq1) in enumerate(sorted_by_length):
        # Check if this n-gram is a subset of any longer n-gram we're keeping
        is_subset = False
        for j, (term2, _) in enumerate(sorted_by_length[:i]):
            if term1 in term2:
                is_subset = True
                break
        
        if not is_subset:
            final_ngrams.append((term1, freq1))
            
        # Limit to top_n
        if len(final_ngrams) >= top_n:
            break
    
    # If we don't have enough final n-grams, use the original meaningful_ngrams
    if len(final_ngrams) < top_n:
        # Sort by frequency and take top_n
        top = sorted(meaningful_ngrams, key=lambda x: x[1], reverse=True)[:top_n]
        return top
    
    return final_ngrams


def correlate_ngrams_with_metrics(df, ngrams, metric_column='engagementRate', text_column='title'):
    """
    Correlate n-grams in video text with a specific performance metric.
    
    Args:
        df: DataFrame with video data
        ngrams: List of (ngram, frequency) tuples
        metric_column: Performance metric to correlate with (default: engagementRate)
        text_column: Column to search for n-grams (default: title)
        
    Returns:
        Dictionary with n-grams as keys and dictionaries of stats as values
    """
    results = {}
    
    # Clean text more efficiently using cached function and parallel processing
    texts = df[text_column].astype(str).tolist()
    cleaned_texts = process_texts_in_parallel(texts)
    
    # Create a temporary column with cleaned text
    df = df.copy()
    df['cleaned_' + text_column] = cleaned_texts
    
    # Filter out n-grams that contain music-related terms (more efficiently)
    filtered_ngrams = [(ngram, freq) for ngram, freq in ngrams 
                       if not any(music_term in ngram.lower() for music_term in MUSIC_TERMS)]
    
    # Process all n-grams at once using vectorized operations where possible
    for ngram, freq in filtered_ngrams:
        # Filter videos containing this n-gram in the cleaned text column
        mask = df['cleaned_' + text_column].str.contains(ngram, case=False, na=False)
        videos_with_ngram = df[mask]
        
        if len(videos_with_ngram) > 0:
            # Calculate statistics
            stats = {
                'frequency': freq,
                'video_count': len(videos_with_ngram),
                'avg_metric': float(videos_with_ngram[metric_column].mean()),
                'median_metric': float(videos_with_ngram[metric_column].median()),
                'max_metric': float(videos_with_ngram[metric_column].max()),
                'avg_views': float(videos_with_ngram['viewCount'].mean())
            }
            
            # Add additional metrics if they exist in the dataframe (using vectorized operations)
            numeric_columns = {
                'channel_subscribers': 'avg_subscribers',
                'likeRate': 'avg_like_rate',
                'commentRate': 'avg_comment_rate',
                'word_count': 'avg_word_count'
            }
            
            for col, stat_name in numeric_columns.items():
                if col in videos_with_ngram.columns:
                    try:
                        # Convert to numeric, coerce errors to NaN, and then calculate mean
                        numeric_values = pd.to_numeric(videos_with_ngram[col], errors='coerce')
                        if not numeric_values.isna().all():  # Only calculate if we have valid numbers
                            stats[stat_name] = float(numeric_values.mean())
                    except Exception as e:
                        logging.warning(f"Could not calculate {stat_name}: {e}")
            
            results[ngram] = stats
    
    # Sort by average metric value (descending)
    sorted_results = dict(sorted(results.items(), 
                                key=lambda x: x[1]['avg_metric'], 
                                reverse=True))
    return sorted_results


@functools.lru_cache(maxsize=1000)
def clean_text(text):
    """Clean text of transcription artifacts and non-meaningful content (cached for efficiency)"""
    
    # Replace common transcription artifacts
    for artifact in MUSIC_TERMS:
        text = re.sub(r'\b' + re.escape(artifact) + r'\b', ' ', text, flags=re.IGNORECASE)
    
    # Apply regex patterns
    text = BRACKET_CONTENT_PATTERN.sub(' ', text)
    text = TIMESTAMP_PATTERN.sub(' ', text)
    text = SPECIAL_CHARS_PATTERN.sub(' ', text)
    text = NUMBERS_PATTERN.sub(' ', text)
    text = SPEAKER_PATTERN.sub(' ', text)
    text = TIME_INDICATOR_PATTERN.sub(' ', text) 
    text = NON_SPEECH_PATTERN.sub(' ', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = WHITESPACE_PATTERN.sub(' ', text).strip()
    
    return text

def process_texts_in_parallel(texts, max_workers=None):
    """Process a list of texts in parallel using clean_text function"""
    # Use a safer approach with fewer workers and disable parallel processing for small datasets
    if len(texts) < 100:
        return [clean_text(text) for text in texts]  # Sequential processing for small datasets
        
    if max_workers is None:
        max_workers = max(1, min(4, multiprocessing.cpu_count() - 1))  # Limit to a safer number
    
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(clean_text, texts))
    except Exception as e:
        logging.warning(f"Parallel processing failed: {e}. Falling back to sequential processing.")
        return [clean_text(text) for text in texts]  # Fallback to sequential


if __name__ == '__main__':
    import argparse, json
    
    # Define a custom JSON encoder to handle NumPy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
    
    parser = argparse.ArgumentParser(description='Per-video insights with text analysis')
    parser.add_argument('csv_file', help='Path to YouTube CSV')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

    df = analyze_videos_per_video(args.csv_file)
    quartiles = compute_quartiles(df, 'engagementRate')
    
    # Get n-grams from titles
    title_ngrams = extract_top_ngrams(df, 'title', ngram_range=(2,3))
    
    # Get n-grams from text column with filler words filtered out and 3-4 word phrases
    text_ngrams = extract_top_ngrams(df, 'text', ngram_range=(3,4), top_n=20, additional_stop_words=FILLER_WORDS)
    
    # Analyze descriptions for insights (1-2 word phrases for better tag extraction)
    description_ngrams = []
    if 'description' in df.columns:
        description_ngrams = extract_top_ngrams(df, 'description', ngram_range=(1,2), top_n=15)
    
    # Analyze hooks if present
    hook_ngrams = []
    if 'hook' in df.columns:
        hook_ngrams = extract_top_ngrams(df, 'hook', ngram_range=(3,5), top_n=15)
    
    # Extract topics if present
    topics = []
    if 'topic' in df.columns:
        topics = df['topic'].value_counts().head(10).to_dict()
    
    # Correlate n-grams with engagement rate
    title_ngram_correlations = correlate_ngrams_with_metrics(df, title_ngrams, 'engagementRate', 'title')
    text_ngram_correlations = correlate_ngrams_with_metrics(df, text_ngrams, 'engagementRate', 'text')
    
    # Also correlate with views to see which phrases drive more traffic
    title_ngram_view_correlations = correlate_ngrams_with_metrics(df, title_ngrams, 'viewCount', 'title')
    text_ngram_view_correlations = correlate_ngrams_with_metrics(df, text_ngrams, 'viewCount', 'text')

    # Correlate description phrases with metrics if present
    description_correlations = {}
    if description_ngrams:
        description_correlations = correlate_ngrams_with_metrics(df, description_ngrams, 'engagementRate', 'description')
    
    # Correlate hooks with metrics if present
    hook_correlations = {}
    if hook_ngrams:
        hook_correlations = correlate_ngrams_with_metrics(df, hook_ngrams, 'engagementRate', 'hook')
    
    # Advanced topic analysis: Find which topics get most engagement
    topic_engagement = {}
    if 'topic' in df.columns:
        topic_groups = df.groupby('topic')
        for topic, group in topic_groups:
            if len(group) >= 3:  # Only include topics with enough videos
                topic_engagement[topic] = {
                    'video_count': len(group),
                    'avg_engagement': float(group['engagementRate'].mean()),
                    'avg_views': float(group['viewCount'].mean()),
                    'avg_sentiment': float(group['sentiment'].mean()) if 'sentiment' in group.columns else 0
                }
        # Sort by engagement
        topic_engagement = dict(sorted(topic_engagement.items(), key=lambda x: x[1]['avg_engagement'], reverse=True))

    # Convert DataFrame to a dictionary with native Python types before serializing
    output = {
        'quartiles_engagement': quartiles,
        'title_ngrams': title_ngrams,
        'text_ngrams': text_ngrams,
        'title_ngram_engagement_analysis': title_ngram_correlations,
        'text_ngram_engagement_analysis': text_ngram_correlations,
        'title_ngram_view_analysis': title_ngram_view_correlations,
        'text_ngram_view_analysis': text_ngram_view_correlations
    }
    
    # Add additional analyses if data is available
    if description_ngrams:
        output['description_ngrams'] = description_ngrams
        output['description_engagement_analysis'] = description_correlations
    
    if hook_ngrams:
        output['hook_ngrams'] = hook_ngrams
        output['hook_engagement_analysis'] = hook_correlations
    
    if topics:
        output['top_topics'] = topics
        output['topic_engagement_analysis'] = topic_engagement

    # We're using the custom NumpyEncoder to handle NumPy types
    print(json.dumps(output, cls=NumpyEncoder, indent=2))
