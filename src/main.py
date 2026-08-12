"""
Orchestrates the full experiment: trains all 9 algorithms under identical
setups, evaluates, generates all required comparisons, and saves results.
"""
import os
import pandas as pd
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress INFO/WARNING logs too

from data_loader import load_data, get_activity_labels
from optimizers_config import OPTIMIZER_CONFIGS, EPOCHS
from train import train_and_evaluate
from visualize import (plot_single_history, plot_confusion_matrix,
                        plot_group_comparison, plot_final_ranking)

os.makedirs("results/history", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)


def main():
    print("Loading data...")
    X_train, X_val, X_test, y_train, y_val, y_test, y_test_raw = load_data()
    class_names = get_activity_labels()

    all_results = []

    for name, cfg in OPTIMIZER_CONFIGS.items():
        print(f"\n=== Training {name} (Part {cfg['part']}, batch={cfg['batch_size']}) ===")
        result, cm, history = train_and_evaluate(
            name, cfg["build"], cfg["batch_size"],
            X_train, y_train, X_val, y_val,
            X_test, y_test, y_test_raw,
            epochs=EPOCHS
        )
        result["part"] = cfg["part"]
        all_results.append(result)

        plot_single_history(name, history.history)
        plot_confusion_matrix(name, cm, class_names)
        print(result)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv("results/metrics.csv", index=False)
    print("\nAll results saved to results/metrics.csv")

    # ---- Comparison (a) Part I ----
    part1 = [n for n in OPTIMIZER_CONFIGS if OPTIMIZER_CONFIGS[n]["part"] == "I"]
    plot_group_comparison(results_df, part1, "Part I (GD / Minibatch / SGD)", "comparison_part1")

    # ---- Comparison (b) Part II ----
    part2 = [n for n in OPTIMIZER_CONFIGS if OPTIMIZER_CONFIGS[n]["part"] == "II"]
    plot_group_comparison(results_df, part2, "Part II (Momentum variants)", "comparison_part2")

    # ---- Comparison (c) Part III ----
    part3 = [n for n in OPTIMIZER_CONFIGS if OPTIMIZER_CONFIGS[n]["part"] == "III"]
    plot_group_comparison(results_df, part3, "Part III (Adaptive methods)", "comparison_part3")

    # ---- Comparison (d) Best of each group ----
    def best_of(group):
        sub = results_df[results_df["name"].isin(group)]
        return sub.loc[sub["test_f1_macro"].idxmax(), "name"]

    best_names = [best_of(part1), best_of(part2), best_of(part3)]
    print("\nBest of each group:", best_names)
    plot_group_comparison(results_df, best_names, "Best of Part I/II/III", "comparison_best_of_each")

    # ---- Final ranking of all 9 ----
    plot_final_ranking(results_df)
    ranked = results_df.sort_values("test_f1_macro", ascending=False)
    print("\nFinal Ranking (by F1 macro):")
    print(ranked[["name", "test_accuracy", "test_f1_macro", "train_time_sec"]].to_string(index=False))


if __name__ == "__main__":
    main()