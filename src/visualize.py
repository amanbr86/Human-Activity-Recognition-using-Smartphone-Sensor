"""
All plotting functions. Saves figures to results/figures/.
"""
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

FIG_DIR = "results/figures"
sns.set_style("whitegrid")


def plot_single_history(name, history_dict):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history_dict["loss"], label="train")
    axes[0].plot(history_dict["val_loss"], label="val")
    axes[0].set_title(f"{name} - Loss")
    axes[0].set_xlabel("Epoch"); axes[0].legend()

    axes[1].plot(history_dict["accuracy"], label="train")
    axes[1].plot(history_dict["val_accuracy"], label="val")
    axes[1].set_title(f"{name} - Accuracy")
    axes[1].set_xlabel("Epoch"); axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/{name}_curves.png", dpi=150)
    plt.close()


def plot_confusion_matrix(name, cm, class_names):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/{name}_confusion.png", dpi=150)
    plt.close()


def plot_group_comparison(results_df, group_names, group_label, filename):
    """Overlaid loss curves + accuracy/F1 bar chart for a group of algorithms."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Overlaid val_loss curves
    for name in group_names:
        with open(f"results/history/{name}.json") as f:
            hist = json.load(f)
        axes[0].plot(hist["val_loss"], label=name)
    axes[0].set_title(f"{group_label}: Validation Loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].legend()

    # Bar chart accuracy vs f1
    sub = results_df[results_df["name"].isin(group_names)]
    x = np.arange(len(sub))
    width = 0.35
    axes[1].bar(x - width/2, sub["test_accuracy"], width, label="Accuracy")
    axes[1].bar(x + width/2, sub["test_f1_macro"], width, label="F1 (macro)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(sub["name"], rotation=30, ha="right")
    axes[1].set_title(f"{group_label}: Accuracy vs F1")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/{filename}.png", dpi=150)
    plt.close()


def plot_final_ranking(results_df):
    ranked = results_df.sort_values("test_f1_macro", ascending=True)
    plt.figure(figsize=(9, 6))
    plt.barh(ranked["name"], ranked["test_f1_macro"], color="steelblue")
    plt.xlabel("Test F1 (macro)")
    plt.title("Final Ranking of All 9 Optimization Algorithms")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/final_ranking.png", dpi=150)
    plt.close()