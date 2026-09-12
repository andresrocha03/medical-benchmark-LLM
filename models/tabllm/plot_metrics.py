import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd

try:
    from .setup_config import RESULTS_DIR
except ImportError:  # Allow direct execution from models/tabllm.
    from setup_config import RESULTS_DIR


DEFAULT_METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
]


METRIC_TITLES = {
    "accuracy": "Top Accuracy Model Per Dataset",
    "precision": "Top Positive-Class Precision Model Per Dataset",
    "recall": "Top Positive-Class Recall Model Per Dataset",
    "f1_score": "Top Macro F1 Model Per Dataset",
}


def get_best_rows_per_dataset(metrics_df, metric):
    """
    Select the best model/serialization row per dataset for one metric.
    """
    best_indices = metrics_df.groupby("dataset")[metric].idxmax()
    return (
        metrics_df.loc[best_indices]
        .sort_values("dataset")
        .reset_index(drop=True)
    )


def plot_best_metric(metrics_df, metric, output_dir):
    """
    Save one bar plot showing the best model per dataset for one metric.
    """
    best_df = get_best_rows_per_dataset(metrics_df, metric)
    labels = best_df["dataset"].tolist()
    values = best_df[metric].tolist()
    annotations = [
        f"{row.model}\n{row.serialization}"
        for row in best_df.itertuples(index=False)
    ]

    fig_width = max(8, 2.6 * len(best_df))
    fig, ax = plt.subplots(figsize=(fig_width, 5.5))
    bars = ax.bar(labels, values, color="#4C78A8")

    ax.set_title(METRIC_TITLES.get(metric, f"Top {metric} Model Per Dataset"))
    ax.set_xlabel("Dataset")
    ax.set_ylabel(metric)
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    for bar, value, annotation in zip(bars, values, annotations):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.3f}\n{annotation}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()

    output_path = Path(output_dir) / f"best_{metric}_per_dataset.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path, best_df


def generate_best_model_plots(
    metrics_path=RESULTS_DIR / "final_tabular_llm_metrics.csv",
    output_dir=RESULTS_DIR / "plots",
    metrics=None,
):
    """
    Generate one plot per metric showing the best model per dataset.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.read_csv(metrics_path)

    if metrics is None:
        metrics = DEFAULT_METRICS

    missing_metrics = [
        metric
        for metric in metrics
        if metric not in metrics_df.columns
    ]

    if missing_metrics:
        raise ValueError(f"Missing metric columns: {missing_metrics}")

    plot_paths = []
    best_rows = []

    for metric in metrics:
        plot_path, best_df = plot_best_metric(metrics_df, metric, output_dir)
        plot_paths.append(plot_path)

        metric_best_df = best_df.copy()
        metric_best_df.insert(0, "metric", metric)
        best_rows.append(metric_best_df)

    best_summary = pd.concat(best_rows, ignore_index=True)
    best_summary_path = output_dir / "best_models_per_dataset.csv"
    best_summary.to_csv(best_summary_path, index=False)

    print(f"Saved plots to {output_dir}")
    print(f"Saved best-model summary to {best_summary_path}")

    return plot_paths, best_summary


if __name__ == "__main__":
    generate_best_model_plots()
