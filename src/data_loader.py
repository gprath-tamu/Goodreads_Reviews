import gzip
import json
import pandas as pd


def load_goodreads_data(file_path):
    """
    Load Goodreads JSON.gz dataset into a pandas DataFrame.

    Args:
        file_path (str): Path to the JSON.gz file

    Returns:
        pd.DataFrame: Loaded dataset
    """
    print("Loading dataset...")

    rows = []
    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    df = pd.DataFrame(rows)

    print("Data loaded successfully.\n")
    return df


def dataset_summary(df):
    """
    Print key dataset statistics.

    Args:
        df (pd.DataFrame): Loaded dataset
    """
    print("========== DATASET SUMMARY ==========\n")

    print(f"Total rows (reviews): {len(df):,}")
    print(f"Total columns: {len(df.columns)}")

    print(df.info())

    print("\nUnique counts:")
    print(f"- Unique users: {df['user_id'].nunique():,}")
    print(f"- Unique books: {df['book_id'].nunique():,}")

    print("\n=====================================")


if __name__ == "__main__":
    DATA_PATH = "data/raw/goodreads_reviews_comics_graphic.json.gz"

    df = load_goodreads_data(DATA_PATH)
    dataset_summary(df)
