import logging
logger = logging.getLogger(__name__)

def read_csv(file_path):
    logger.info(f"Attempting to read file: {file_path}")
    # Support Parquet files natively
    if file_path.lower().endswith('.parquet'):
        import pandas as pd
        try:
            logger.debug(f"Reading Parquet file: {file_path}")
            df = pd.read_parquet(file_path)
            logger.info(f"Successfully read Parquet, records: {len(df)}")
            return df.to_dict('records')
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
        except Exception as e:
            logger.error(f"Error reading Parquet file {file_path}: {e}")
        return []
    import csv
    import sys
    # allow large CSV fields, fallback if system max is too large
    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        csv.field_size_limit(2**31 - 1)
    data = []
    try:
        with open(file_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append(row)
        logger.info(f"Successfully read CSV file, records: {len(data)}")
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
    except Exception as e:
        logger.error(f"Error reading CSV file {file_path}: {e}")
    return data

# Optional: export data to Parquet format
def write_parquet(data, file_path):
    import pandas as pd
    logger.info(f"Attempting to write Parquet file: {file_path}, records: {len(data)}")
    df = pd.DataFrame(data)
    try:
        df.to_parquet(file_path)
        logger.info(f"Successfully wrote Parquet file: {file_path}")
    except Exception as e:
        logger.error(f"Error writing Parquet file {file_path}: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("csv_reader module executed as script")