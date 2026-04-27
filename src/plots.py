import os
import ast
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


def _support_to_suffix(support):
    """
    Convert support float to filename suffix used in saved itemset/rule files.
    Example:
        0.005 -> '0005'
        0.01  -> '001'
        0.02  -> '002'
    """
    return str(support).replace(".", "")


def _safe_parse_frozenset(val):
    """
    Parse strings like 'frozenset({...})' into Python lists.
    """
    if pd.isna(val):
        return []

    s = str(val)

    if s.startswith("frozenset(") and s.endswith(")"):
        inner = s[len("frozenset("):-1]
        try:
            parsed = ast.literal_eval(inner)
            return [str(x) for x in parsed]
        except Exception:
            return [s]

    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (set, list, tuple)):
            return [str(x) for x in parsed]
    except Exception:
        pass

    return [s]


def _format_rule(row):
    antecedent = ", ".join(_safe_parse_frozenset(row["antecedents"]))
    consequent = ", ".join(_safe_parse_frozenset(row["consequents"]))
    return f"{{{antecedent}}} → {{{consequent}}}"


def load_results(base_path=".", support=0.01):
    """
    Load itemsets and rules for a single support value.
    """
    suffix = _support_to_suffix(support)

    itemsets_path = os.path.join(base_path, "outputs", "itemsets", f"itemsets_sup_{suffix}.csv")
    rules_path = os.path.join(base_path, "outputs", "itemsets", f"rules_sup_{suffix}.csv")

    if not os.path.exists(itemsets_path):
        raise FileNotFoundError(f"Itemsets file not found: {itemsets_path}")

    if not os.path.exists(rules_path):
        raise FileNotFoundError(f"Rules file not found: {rules_path}")

    itemsets = pd.read_csv(itemsets_path)
    rules = pd.read_csv(rules_path)

    return itemsets, rules


def create_output_dir(base_path=".", support=0.01, min_user_freq=10, max_len=2):
    """
    Create a parameter-specific output folder for figures.
    """
    folder_name = f"sup_{_support_to_suffix(support)}_freq_{min_user_freq}_len_{max_len}"
    output_dir = os.path.join(base_path, "outputs", "figures", folder_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def create_itemset_size_distribution(itemsets, output_dir, support, min_user_freq, max_len):
    itemsets = itemsets.copy()
    itemsets["itemset_size"] = itemsets["itemsets"].apply(lambda x: len(_safe_parse_frozenset(x)))

    plt.figure(figsize=(7, 4))
    sns.countplot(data=itemsets, x="itemset_size", color="#7a0019")
    plt.title(f"Itemset Size Distribution\nsupport={support}, min_user_freq={min_user_freq}, max_len={max_len}")
    plt.xlabel("Itemset Size")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "itemset_size_distribution.png"), dpi=300, bbox_inches="tight")
    plt.close()


def create_top_itemsets_plot(itemsets, output_dir, support, min_user_freq, max_len, top_n=10):
    top_itemsets = itemsets.sort_values("support", ascending=False).head(top_n).copy()
    top_itemsets["itemset_label"] = top_itemsets["itemsets"].apply(
        lambda x: "{" + ", ".join(_safe_parse_frozenset(x)) + "}"
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_itemsets, y="itemset_label", x="support", color="#008272")
    plt.title(f"Top {top_n} Frequent Itemsets by Support\nsupport={support}, min_user_freq={min_user_freq}, max_len={max_len}")
    plt.xlabel("Support")
    plt.ylabel("Itemset")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top_itemsets_by_support.png"), dpi=300, bbox_inches="tight")
    plt.close()

    top_itemsets.to_csv(os.path.join(output_dir, "top_itemsets_by_support.csv"), index=False)


def create_lift_distribution(rules, output_dir, support, min_user_freq, max_len):
    if "lift" not in rules.columns or rules.empty:
        return

    plt.figure(figsize=(7, 4))
    sns.histplot(rules["lift"], bins=30, color="#7a0019", edgecolor="white")
    plt.title(f"Lift Distribution\nsupport={support}, min_user_freq={min_user_freq}, max_len={max_len}")
    plt.xlabel("Lift")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lift_distribution.png"), dpi=300, bbox_inches="tight")
    plt.close()


def create_confidence_distribution(rules, output_dir, support, min_user_freq, max_len):
    if "confidence" not in rules.columns or rules.empty:
        return

    plt.figure(figsize=(7, 4))
    sns.histplot(rules["confidence"], bins=30, color="#003c71", edgecolor="white")
    plt.title(f"Confidence Distribution\nsupport={support}, min_user_freq={min_user_freq}, max_len={max_len}")
    plt.xlabel("Confidence")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confidence_distribution.png"), dpi=300, bbox_inches="tight")
    plt.close()


def create_top_rules_by_lift(rules, output_dir, support, min_user_freq, max_len, top_n=10):
    if rules.empty or "lift" not in rules.columns:
        return

    top_rules = rules.sort_values("lift", ascending=False).head(top_n).copy()
    top_rules["rule_label"] = top_rules.apply(_format_rule, axis=1)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_rules, y="rule_label", x="lift", color="#7a0019")
    plt.title(f"Top {top_n} Rules by Lift\nsupport={support}, min_user_freq={min_user_freq}, max_len={max_len}")
    plt.xlabel("Lift")
    plt.ylabel("Rule")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top_rules_by_lift.png"), dpi=300, bbox_inches="tight")
    plt.close()

    top_rules.to_csv(os.path.join(output_dir, "top_rules_by_lift.csv"), index=False)


def create_top_rules_by_confidence(rules, output_dir, support, min_user_freq, max_len, top_n=10):
    if rules.empty or "confidence" not in rules.columns:
        return

    top_rules = rules.sort_values("confidence", ascending=False).head(top_n).copy()
    top_rules["rule_label"] = top_rules.apply(_format_rule, axis=1)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_rules, y="rule_label", x="confidence", color="#003c71")
    plt.title(f"Top {top_n} Rules by Confidence\nsupport={support}, min_user_freq={min_user_freq}, max_len={max_len}")
    plt.xlabel("Confidence")
    plt.ylabel("Rule")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top_rules_by_confidence.png"), dpi=300, bbox_inches="tight")
    plt.close()

    top_rules.to_csv(os.path.join(output_dir, "top_rules_by_confidence.csv"), index=False)


def create_support_vs_lift_scatter(rules, output_dir, support, min_user_freq, max_len):
    if rules.empty or not {"support", "lift"}.issubset(rules.columns):
        return

    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=rules, x="support", y="lift", alpha=0.7, s=40, color="#7a0019")
    plt.title(f"Rule Support vs Lift\nsupport={support}, min_user_freq={min_user_freq}, max_len={max_len}")
    plt.xlabel("Rule Support")
    plt.ylabel("Lift")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "support_vs_lift_scatter.png"), dpi=300, bbox_inches="tight")
    plt.close()


def create_support_vs_confidence_scatter(rules, output_dir, support, min_user_freq, max_len):
    if rules.empty or not {"support", "confidence"}.issubset(rules.columns):
        return

    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=rules, x="support", y="confidence", alpha=0.7, s=40, color="#008272")
    plt.title(f"Rule Support vs Confidence\nsupport={support}, min_user_freq={min_user_freq}, max_len={max_len}")
    plt.xlabel("Rule Support")
    plt.ylabel("Confidence")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "support_vs_confidence_scatter.png"), dpi=300, bbox_inches="tight")
    plt.close()


def create_rule_length_distribution(rules, output_dir, support, min_user_freq, max_len):
    if rules.empty:
        return

    rules = rules.copy()
    rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(_safe_parse_frozenset(x)))
    rules["consequent_len"] = rules["consequents"].apply(lambda x: len(_safe_parse_frozenset(x)))
    rules["rule_len"] = rules["antecedent_len"] + rules["consequent_len"]

    plt.figure(figsize=(7, 4))
    sns.countplot(data=rules, x="rule_len", color="#ff8c00")
    plt.title(f"Rule Length Distribution\nsupport={support}, min_user_freq={min_user_freq}, max_len={max_len}")
    plt.xlabel("Total Rule Length (LHS + RHS)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rule_length_distribution.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_rq1_results(base_path=".", support=0.01, min_user_freq=10, max_len=2):
    """
    Main plotting pipeline.

    Generates:
    1. Per-support detailed plots
    2. Summary plots across multiple support values
    """
    print(f"Generating plots for support={support}, min_user_freq={min_user_freq}, max_len={max_len}")

    itemsets, rules = load_results(base_path=base_path, support=support)
    output_dir = create_output_dir(
        base_path=base_path,
        support=support,
        min_user_freq=min_user_freq,
        max_len=max_len
    )

    # detailed plots for chosen support
    create_itemset_size_distribution(itemsets, output_dir, support, min_user_freq, max_len)
    create_top_itemsets_plot(itemsets, output_dir, support, min_user_freq, max_len)
    create_lift_distribution(rules, output_dir, support, min_user_freq, max_len)
    create_confidence_distribution(rules, output_dir, support, min_user_freq, max_len)
    create_top_rules_by_lift(rules, output_dir, support, min_user_freq, max_len)
    create_top_rules_by_confidence(rules, output_dir, support, min_user_freq, max_len)
    create_support_vs_lift_scatter(rules, output_dir, support, min_user_freq, max_len)
    create_support_vs_confidence_scatter(rules, output_dir, support, min_user_freq, max_len)
    create_rule_length_distribution(rules, output_dir, support, min_user_freq, max_len)


    print(f"Plots saved to: {output_dir}")

    return {
        "output_dir": output_dir,
        "num_itemsets": len(itemsets),
        "num_rules": len(rules)
    }


if __name__ == "__main__":
    results = plot_rq1_results(
        base_path=".",
        support=0.01,
        min_user_freq=10,
        max_len=3
    )
    print(results)