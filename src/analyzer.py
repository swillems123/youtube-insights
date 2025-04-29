import logging

def analyze_videos(csv_file_path):
    logger = logging.getLogger(__name__)
    logger.info(f"Starting analysis for {csv_file_path}")
    import pandas as pd
    from .utils.csv_reader import read_csv
    from .utils.data_cleaner import clean_data
    from .utils.metrics import calculate_engagement_rate, calculate_average_views

    # Read CSV file or all CSVs in a directory
    import os, glob
    raw_data = []
    if os.path.isdir(csv_file_path):
        logger.info(f"Directory detected, scanning for *.csv files in {csv_file_path}")
        csv_files = glob.glob(os.path.join(csv_file_path, '*.csv'))
        for f in csv_files:
            raw_data.extend(read_csv(f))
        logger.info(f"Read {len(raw_data)} records from {len(csv_files)} CSV files")
    else:
        raw_data = read_csv(csv_file_path)
        logger.info(f"Read {len(raw_data)} records from CSV")

    # Clean the data
    cleaned_data = clean_data(raw_data)
    logger.info(f"Cleaned data, {len(cleaned_data)} unique records after cleaning")

    # Convert cleaned data to DataFrame for analysis
    df = pd.DataFrame(cleaned_data)
    logger.debug(f"DataFrame sample: {df.head().to_dict()}")

    # Extract numeric columns and calculate metrics
    views = pd.to_numeric(df['viewCount'], errors='coerce').fillna(0).astype(int).tolist()
    likes = pd.to_numeric(df['likeCount'], errors='coerce').fillna(0).astype(int).sum()
    comments = pd.to_numeric(df['commentCount'], errors='coerce').fillna(0).astype(int).sum()
    total_views = sum(views)
    logger.info(f"Total views: {total_views}, Likes: {likes}, Comments: {comments}")

    insights = {
        "total_videos": len(df),
        "average_views": calculate_average_views(views),
        "engagement_rate_percent": calculate_engagement_rate(likes, comments, total_views),
    }
    logger.info(f"Computed insights: {insights}")

    return insights


if __name__ == '__main__':
    import argparse, json

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(description='Analyze YouTube CSV for marketing insights')
    parser.add_argument('csv_file', help='Path to YouTube CSV file')
    args = parser.parse_args()

    logger.info(f"Analyzing file: {args.csv_file}")
    result = analyze_videos(args.csv_file)
    logger.info("Analysis complete")
    print(json.dumps(result, indent=2))