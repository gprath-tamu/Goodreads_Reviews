import os
import gzip
import json
import pandas as pd

RAW_PATH = os.path.join("data", "raw", "goodreads_books_comics_graphic.json.gz")
PROCESSED_PATH = os.path.join("data", "processed", "books.csv")


def _normalize_title(title):
    """
    Normalize title for matching
    """
    return " ".join(str(title).strip().lower().split())


def load_books(base_path="."):
    """
    Load books dataset.

    Behavior:
    1. If processed CSV exists -> load it directly (fast)
    2. Else -> process .gz file, save books.csv, and return DataFrame

    Returns DataFrame with columns:
    - book_id
    - original_title
    - norm_title
    """

    csv_path = os.path.join(base_path, PROCESSED_PATH)

    # Fast path: already processed
    if os.path.exists(csv_path):
        print("✅ Loading books from processed CSV")
        df = pd.read_csv(csv_path, dtype={"book_id": str})
        return df

    print("⚙️ Processing books from .gz (only first time)")

    gz_path = os.path.join(base_path, RAW_PATH)

    if not os.path.exists(gz_path):
        raise FileNotFoundError(f"Books file not found: {gz_path}")

    records = []

    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)

                book_id = obj.get("book_id")
                title = obj.get("title")

                if book_id is None or title is None:
                    continue

                title = str(title).strip()
                if title == "":
                    continue

                records.append({
                    "book_id": str(book_id).strip(),
                    "original_title": title,
                    "norm_title": _normalize_title(title)
                })

            except Exception:
                continue

    df = pd.DataFrame(records)

    if df.empty:
        raise ValueError("Books dataset is empty after processing.")

    # Ensure type is string
    df["book_id"] = df["book_id"].astype(str)

    # Remove duplicates
    df = df.drop_duplicates(subset=["book_id", "norm_title"]).reset_index(drop=True)

    # Save processed CSV
    processed_dir = os.path.join(base_path, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    df.to_csv(csv_path, index=False)

    print(f"✅ Saved processed books → {csv_path}")
    print(f"✅ Total books: {len(df)}")

    return df


def find_book_matches(book_name, books_df, top_n=10):
    """
    Find likely matches for a user-entered book name.

    Priority:
    1. Exact normalized match
    2. Substring match
    """
    query = _normalize_title(book_name)

    exact = books_df[books_df["norm_title"] == query]
    if not exact.empty:
        return exact[["book_id", "original_title", "norm_title"]].head(top_n).reset_index(drop=True)

    contains = books_df[books_df["norm_title"].str.contains(query, na=False, regex=False)]
    return contains[["book_id", "original_title", "norm_title"]].head(top_n).reset_index(drop=True)


def build_id_to_title_map(books_df):
    """
    Build book_id -> title dictionary
    """
    return dict(zip(books_df["book_id"].astype(str), books_df["original_title"]))


def get_book_title(book_id, books_df):
    """
    Get book title for a given book_id
    """
    book_id = str(book_id)

    match = books_df[books_df["book_id"].astype(str) == book_id]

    if match.empty:
        return None

    return match.iloc[0]["original_title"]