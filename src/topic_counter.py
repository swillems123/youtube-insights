#!/usr/bin/env python
"""
Topic Counter: Lists all topic IDs and their counts in the dataset
"""

import pandas as pd
import logging
import argparse
import json

# Local imports
from .utils.csv_reader import read_csv
from .utils.data_cleaner import clean_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def count_topics(df):
    """
    Count occurrences of each topic ID in the dataset.
    
    Args:
        df: Pandas DataFrame containing the YouTube data
    
    Returns:
        Dictionary with topic IDs as keys and counts as values
    """
    if 'topic' not in df.columns:
        logger.error("No 'topic' column found in the dataset")
        return {}
    
    # Count topics
    topic_counts = df['topic'].value_counts().to_dict()
    
    # Convert keys to strings for JSON compatibility
    return {str(k): v for k, v in topic_counts.items()}

def main():
    """Main function to parse arguments and run the topic counter."""
    parser = argparse.ArgumentParser(description='Count topic IDs in YouTube data')
    parser.add_argument('csv_file', help='Path to YouTube CSV data file')
    parser.add_argument('--min-count', type=int, default=1, help='Minimum count to include in results')
    parser.add_argument('--output', '-o', help='Output JSON file path (optional)')
    
    args = parser.parse_args()
    
    # Read and process the CSV file
    logger.info(f"Reading data from {args.csv_file}")
    raw_data = read_csv(args.csv_file)
    cleaned_data = clean_data(raw_data)
    df = pd.DataFrame(cleaned_data)
    
    # Count topics
    logger.info("Counting topics")
    topic_counts = count_topics(df)
    
    # Filter by minimum count if specified
    if args.min_count > 1:
        topic_counts = {k: v for k, v in topic_counts.items() if v >= args.min_count}
    
    # Print the results
    if topic_counts:
        # Sort by count (descending)
        sorted_counts = dict(sorted(topic_counts.items(), key=lambda x: x[1], reverse=True))
        logger.info(f"Found {len(sorted_counts)} different topics")
        result = {"topic_counts": sorted_counts}
        
        if args.output:
            # Save to JSON file
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"Results saved to {args.output}")
        else:
            # Print to console
            print(json.dumps(result, indent=2))
    else:
        logger.warning("No topics found")

if __name__ == "__main__":
    main()