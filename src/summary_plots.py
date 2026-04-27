import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


def support_to_suffix(support):
    """
    Convert support value to filename format
    Example:
        0.005 -> '0005'
        0.01  -> '001'
        0.02  -> '002'
    """
    return str(support).replace(".", "")


def load_summary_data(base_path=".", supports=None):
    """
    Load itemset and rule counts for given supports
    """
    summary = []

    for support in supports:
        suffix = support_to_suffix(support)

        itemsets_path = os.path.join(
            base_path, "outputs", "itemsets", f"itemsets_sup_{suffix}.csv"
        )
        rules_path = os.path.join(
            base_path, "outputs", "itemsets", f"rules_sup_{suffix}.csv"
        )

        if not os.path.exists(itemsets_path):
            print(f"Missing: {itemsets_path}")
            continue

        if not os.path.exists(rules_path):
            print(f"Missing: {rules_path}")
            continue

        itemsets = pd.read_csv(itemsets_path)
        rules = pd.read_csv(rules_path)

        summary.append({
            "support": support,
            "num_itemsets": len(itemsets),
            "num_rules": len(rules)
        })

    summary_df = pd.DataFrame(summary).sort_values("support")

    return summary_df


def plot_itemsets_vs_support(summary_df, output_dir):
    plt.figure(figsize=(7, 4))
    sns.lineplot(
        data=summary_df,
        x="support",
        y="num_itemsets",
        marker="o",
        color="#7a0019"
    )

    plt.title("Support vs Number of Frequent Itemsets")
    plt.xlabel("Minimum Support")
    plt.ylabel("Number of Itemsets")
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, "support_vs_itemsets.png"), dpi=300)
    plt.show()


def plot_rules_vs_support(summary_df, output_dir):
    plt.figure(figsize=(7, 4))

    sns.lineplot(
        data=summary_df,
        x="support",
        y="num_rules",
        marker="o",
        color="#003c71"
    )

    plt.title("Support vs Number of Rules")
    plt.xlabel("Minimum Support")
    plt.ylabel("Number of Rules")
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, "support_vs_rules.png"), dpi=300)
    plt.show()


def run_summary_plots(base_path=".", supports=None):
    """
    Main function for summary plots
    """

    if supports is None:
        supports = [0.005, 0.01, 0.02]

    print("Generating summary plots for supports:", supports)

    output_dir = os.path.join(base_path, "outputs", "figures", "summary")
    os.makedirs(output_dir, exist_ok=True)

    summary_df = load_summary_data(base_path, supports)

    if summary_df.empty:
        print("No data found — check your paths")
        return None

    # save summary table
    summary_df.to_csv(os.path.join(output_dir, "summary.csv"), index=False)

    # plots
    plot_itemsets_vs_support(summary_df, output_dir)
    plot_rules_vs_support(summary_df, output_dir)

    print("Saved summary plots at:", output_dir)

    return summary_df