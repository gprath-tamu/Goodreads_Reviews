import pandas as pd
import pickle
import os
import warnings
warnings.filterwarnings("ignore")


def handle_missing_and_types(df):
    """
    Handle missing values and convert data types.
    """

    print("Handling missing values and converting data types...")

    # Convert empty strings to missing values
    df = df.replace("", pd.NA)

    # Convert datetime columns with UTC (fix warnings)
    time_cols = ["date_added", "date_updated", "read_at", "started_at"]
    for col in time_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    # Convert numeric columns
    numeric_cols = ["rating", "n_votes", "n_comments"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    print("Missing values and data types handled.")
    return df


def clean_core_columns(df):
    """
    Remove invalid rows for user–book interaction analysis.
    """

    print("Cleaning core identifier columns...")

    before = len(df)

    # Drop rows missing essential identifiers
    df = df.dropna(subset=["user_id", "book_id"])

    # Keep valid ratings only
    if "rating" in df.columns:
        df = df[df["rating"].between(1, 5, inclusive="both")]

    after = len(df)

    print(f"Removed {before - after:,} rows due to missing identifiers or invalid ratings.")
    return df


def prepare_text_columns(df):
    """
    Clean text and create review length feature.
    """

    print("Preparing review text columns...")

    if "review_text" in df.columns:
        df["review_text"] = df["review_text"].fillna("").astype(str).str.strip()
        df["review_len_tokens"] = df["review_text"].str.split().str.len()

    print("Text cleaned and review_len_tokens created.")
    return df


def check_duplicates(df):
    """
    Check for duplicate rows.
    """

    duplicate_count = df.duplicated().sum()
    print(f"Duplicate rows detected: {duplicate_count:,}")
    return duplicate_count


def create_baskets(df):
    """
    Create user -> distinct list of books (transactions).
    """

    print("Creating user baskets...")

    baskets = (
        df.groupby("user_id")["book_id"]
        .apply(lambda x: sorted(set(x)))
    )

    print(f"Total users before filtering: {len(baskets):,}")
    return baskets


def filter_baskets(baskets, min_books=2):
    """
    Remove users with fewer than min_books interactions.
    """

    print(f"Filtering baskets with fewer than {min_books} books...")

    before = len(baskets)

    baskets = baskets[baskets.apply(len) >= min_books]

    after = len(baskets)

    print(f"Removed {before - after:,} users.")
    print(f"Remaining users: {after:,}")

    return baskets


def save_outputs(df, baskets, base_path="."):
    """
    Save cleaned data and baskets.
    """

    processed_dir = os.path.join(base_path, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    cleaned_path = os.path.join(processed_dir, "cleaned_data.csv")
    baskets_path = os.path.join(processed_dir, "baskets.pkl")

    # Save cleaned dataframe
    df.to_csv(cleaned_path, index=False)

    # Save baskets
    with open(baskets_path, "wb") as f:
        pickle.dump(baskets, f)

    print("Saved outputs:")
    print(f" - {cleaned_path}")
    print(f" - {baskets_path}")


def preprocess_pipeline(df, base_path="."):
    """
    Full preprocessing pipeline.
    """

    df = handle_missing_and_types(df)
    df = clean_core_columns(df)
    df = prepare_text_columns(df)

    check_duplicates(df)

    baskets = create_baskets(df)
    baskets = filter_baskets(baskets, min_books=2)

    save_outputs(df, baskets, base_path)

    print("Preprocessing complete.")
    return df, baskets


# Run as script (optional)
if __name__ == "__main__":
    from data_loader import load_goodreads_data

    DATA_PATH = "data/raw/goodreads_reviews_comics_graphic.json.gz"

    df = load_goodreads_data(DATA_PATH)

    df, baskets = preprocess_pipeline(df)