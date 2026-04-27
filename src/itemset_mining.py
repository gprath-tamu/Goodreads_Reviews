import os
import gc
import pickle
from collections import Counter

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules


def load_baskets(base_path="."):
    """
    Load preprocessed baskets from:
    data/processed/baskets.pkl
    """
    baskets_path = os.path.join(base_path, "data", "processed", "baskets.pkl")

    print("Loading baskets...")
    with open(baskets_path, "rb") as f:
        baskets = pickle.load(f)

    print(f"Loaded baskets for {len(baskets):,} users.")
    return baskets


def filter_items_by_user_frequency(baskets, min_user_freq=10):
    """
    Keep only books that appear in at least `min_user_freq` user baskets.

    Why:
    - reduces dimensionality
    - reduces RAM usage
    - removes extremely rare books that do not help frequent itemset mining
    """
    print(f"Filtering books with user frequency < {min_user_freq}...")

    item_counts = Counter()

    # Count unique book appearances across user baskets
    for books in baskets:
        item_counts.update(set(books))

    keep_items = {item for item, cnt in item_counts.items() if cnt >= min_user_freq}

    # Filter each basket
    filtered_baskets = baskets.apply(lambda books: [b for b in books if b in keep_items])

    # Keep only baskets with >=2 books after filtering
    filtered_baskets = filtered_baskets[filtered_baskets.apply(len) >= 2]

    print(f"Books kept after filtering: {len(keep_items):,}")
    print(f"Users remaining after filtering: {len(filtered_baskets):,}")

    return filtered_baskets, keep_items


def create_transaction_matrix(baskets):
    """
    Convert baskets into one-hot encoded transaction matrix required by mlxtend.
    """
    print("Creating transaction matrix...")

    te = TransactionEncoder()
    te_array = te.fit(baskets.tolist()).transform(baskets.tolist())
    df_trans = pd.DataFrame(te_array, columns=te.columns_)

    print(f"Transaction matrix shape: {df_trans.shape}")
    return df_trans


def run_fp_growth(df_trans, min_support=0.01, max_len=2):
    """
    Run FP-Growth with a single support threshold.
    """
    print(f"Running FP-Growth with min_support={min_support}, max_len={max_len}...")

    freq_itemsets = fpgrowth(
        df_trans,
        min_support=min_support,
        use_colnames=True,
        max_len=max_len
    )

    print(f"Frequent itemsets found: {len(freq_itemsets):,}")
    return freq_itemsets


def generate_rules(freq_itemsets, min_lift=1.0):
    """
    Generate association rules using lift threshold.
    """
    print(f"Generating rules with min_lift={min_lift}...")

    if freq_itemsets.empty:
        print("No frequent itemsets found. Skipping rule generation.")
        return pd.DataFrame()

    rules = association_rules(
        freq_itemsets,
        metric="lift",
        min_threshold=min_lift
    )

    print(f"Rules generated: {len(rules):,}")
    return rules


def add_helper_columns(freq_itemsets, rules):
    """
    Add a few helper columns to make analysis easier later.
    """
    if not freq_itemsets.empty:
        freq_itemsets = freq_itemsets.copy()
        freq_itemsets["itemset_size"] = freq_itemsets["itemsets"].apply(len)

    if not rules.empty:
        rules = rules.copy()
        rules["antecedent_len"] = rules["antecedents"].apply(len)
        rules["consequent_len"] = rules["consequents"].apply(len)
        rules["rule_len"] = rules["antecedent_len"] + rules["consequent_len"]

    return freq_itemsets, rules


def save_results(freq_itemsets, rules, support, base_path="."):
    """
    Save itemsets and rules to outputs/itemsets/
    """
    output_dir = os.path.join(base_path, "outputs", "itemsets")
    os.makedirs(output_dir, exist_ok=True)

    support_str = str(support).replace(".", "")
    itemsets_path = os.path.join(output_dir, f"itemsets_sup_{support_str}.csv")
    rules_path = os.path.join(output_dir, f"rules_sup_{support_str}.csv")

    freq_itemsets.to_csv(itemsets_path, index=False)
    rules.to_csv(rules_path, index=False)

    print(f"Saved: {itemsets_path}")
    print(f"Saved: {rules_path}")


def summarize_run(freq_itemsets, rules, support, min_user_freq, max_len):
    """
    Return small summary dict for quick tracking.
    """
    return {
        "support": support,
        "min_user_freq": min_user_freq,
        "max_len": max_len,
        "num_itemsets": len(freq_itemsets),
        "num_rules": len(rules),
    }


def run_pipeline(base_path=".", support=0.01, min_user_freq=10, max_len=2, min_lift=1.0):
    """
    Full pipeline for a single support value.

    Parameters:
    - base_path: project root
    - support: single minimum support threshold
    - min_user_freq: remove books appearing in fewer than this many user baskets
    - max_len: maximum size of itemsets
    - min_lift: minimum lift threshold for rules
    """

    print(f"Running pipeline for support = {support}")

    baskets = load_baskets(base_path)

    baskets_filtered, keep_items = filter_items_by_user_frequency(
        baskets,
        min_user_freq=min_user_freq
    )

    df_trans = create_transaction_matrix(baskets_filtered)

    freq_itemsets = run_fp_growth(
        df_trans,
        min_support=support,
        max_len=max_len
    )

    rules = generate_rules(
        freq_itemsets,
        min_lift=min_lift
    )

    freq_itemsets, rules = add_helper_columns(freq_itemsets, rules)

    save_results(freq_itemsets, rules, support, base_path)

    summary = summarize_run(
        freq_itemsets=freq_itemsets,
        rules=rules,
        support=support,
        min_user_freq=min_user_freq,
        max_len=max_len
    )

    print(f"\nItemset mining complete for support = {support}.")

    # free memory
    del df_trans
    gc.collect()

    return {
        "summary": summary,
        "itemsets": freq_itemsets,
        "rules": rules,
        "books_kept": len(keep_items),
        "users_kept": len(baskets_filtered)
    }


if __name__ == "__main__":
    BASE_PATH = "."
    results = run_pipeline(
        base_path=BASE_PATH,
        support=0.01,
        min_user_freq=5,
        max_len=4,
        min_lift=1.0
    )
    print(results["summary"])