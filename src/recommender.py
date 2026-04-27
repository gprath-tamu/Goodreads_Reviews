import os
import ast
import pandas as pd

from books_dataloader import load_books, find_book_matches, build_id_to_title_map

DEFAULT_SUPPORT = 0.01


def _support_to_suffix(support=DEFAULT_SUPPORT):
    return str(support).replace(".", "")


def _parse_set(val):
    """
    Convert saved rule strings into Python sets of strings.

    Handles:
    - frozenset({'15704307'})
    - (15704307,)
    - ('15704307', '17131869')
    - 15704307
    """
    if pd.isna(val):
        return set()

    s = str(val).strip()

    if s.startswith("frozenset(") and s.endswith(")"):
        s = s[len("frozenset("):-1]

    try:
        parsed = ast.literal_eval(s)

        if isinstance(parsed, (set, list, tuple)):
            return set(map(str, parsed))

        return {str(parsed)}
    except Exception:
        return {s}


def load_rules(base_path=".", support=DEFAULT_SUPPORT):
    """
    Load rules for a given support
    """
    suffix = _support_to_suffix(support)

    path = os.path.join(
        base_path,
        "outputs",
        "itemsets",
        f"rules_sup_{suffix}.csv"
    )

    if not os.path.exists(path):
        raise FileNotFoundError(f"Rules file not found: {path}")

    return pd.read_csv(path)


def recommend_from_book_name(book_name, top_n=5, base_path=".", support=DEFAULT_SUPPORT):
    """
    Recommender:
    Input:
      - book_name
      - top_n
    Output:
      - recommended book names

    Uses:
      - processed books.csv via books_dataloader
      - rules_sup_001.csv by default (support = 0.01)
    """

    # Load books and rules
    books_df = load_books(base_path)
    rules_df = load_rules(base_path, support=support)

    # Build fast mapping
    id_to_title = build_id_to_title_map(books_df)

    # Find matching book from user input
    matches = find_book_matches(book_name, books_df)

    if matches.empty:
        return {
            "status": "book_not_found",
            "input_book": book_name,
            "matched_book_id": None,
            "matched_book_title": None,
            "recommendations": [],
            "candidate_matches": []
        }

    # Use top match
    matched_book_id = str(matches.iloc[0]["book_id"])
    matched_book_title = str(matches.iloc[0]["original_title"])

    # Parse rule columns
    rules = rules_df.copy()
    rules["antecedent_set"] = rules["antecedents"].apply(_parse_set)
    rules["consequent_set"] = rules["consequents"].apply(_parse_set)

    # Keep rules where selected book is in antecedent
    filtered = rules[rules["antecedent_set"].apply(lambda x: matched_book_id in x)]

    if filtered.empty:
        return {
            "status": "no_rules_found",
            "input_book": book_name,
            "matched_book_id": matched_book_id,
            "matched_book_title": matched_book_title,
            "recommendations": [],
            "candidate_matches": matches.to_dict(orient="records")
        }

    # Rank by strongest rules
    sort_cols = [c for c in ["lift", "confidence", "support"] if c in filtered.columns]
    filtered = filtered.sort_values(sort_cols, ascending=False)

    recommendations = []

    for _, row in filtered.iterrows():
        for rec_id in row["consequent_set"]:
            if rec_id == matched_book_id:
                continue

            rec_title = id_to_title.get(rec_id, rec_id)

            if rec_title not in recommendations:
                recommendations.append(rec_title)

            if len(recommendations) >= top_n:
                return {
                    "status": "ok",
                    "input_book": book_name,
                    "matched_book_id": matched_book_id,
                    "matched_book_title": matched_book_title,
                    "recommendations": recommendations[:top_n],
                    "candidate_matches": matches.to_dict(orient="records")
                }

    return {
        "status": "ok",
        "input_book": book_name,
        "matched_book_id": matched_book_id,
        "matched_book_title": matched_book_title,
        "recommendations": recommendations[:top_n],
        "candidate_matches": matches.to_dict(orient="records")
    }