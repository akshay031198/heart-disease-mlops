"""Exploratory Data Analysis - generates summary stats and plots in reports/figures/.

Usage:
    python -m heart_disease.eda
"""
import logging

import matplotlib

matplotlib.use("Agg")  # headless backend: works in VS Code terminal, CI and Docker
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from heart_disease.config import (  # noqa: E402
    CATEGORICAL_FEATURES,
    FIGURES_DIR,
    NUMERIC_FEATURES,
    RAW_DATA_PATH,
    REPORTS_DIR,
    TARGET,
)
from heart_disease.data import load_clean_data, load_raw_data  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", palette="Set2")


def _save(fig, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("Saved %s", path)


def plot_class_balance(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df[TARGET].value_counts().sort_index()
    sns.barplot(x=["No disease (0)", "Disease (1)"], y=counts.values, ax=ax)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 2, f"{v} ({v / counts.sum():.1%})", ha="center")
    ax.set_title("Class balance")
    ax.set_ylabel("Patients")
    _save(fig, "class_balance.png")


def plot_numeric_histograms(df):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col in zip(axes.flat, NUMERIC_FEATURES):
        sns.histplot(data=df, x=col, hue=TARGET, kde=True, ax=ax, element="step")
        ax.set_title(f"Distribution of {col} by target")
    axes.flat[-1].axis("off")
    _save(fig, "numeric_histograms.png")


def plot_categorical_counts(df):
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, col in zip(axes.flat, CATEGORICAL_FEATURES):
        sns.countplot(data=df, x=col, hue=TARGET, ax=ax)
        ax.set_title(f"{col} vs target")
    _save(fig, "categorical_counts.png")


def plot_correlation_heatmap(df):
    fig, ax = plt.subplots(figsize=(11, 9))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, square=True)
    ax.set_title("Feature correlation heatmap")
    _save(fig, "correlation_heatmap.png")


def plot_boxplots(df):
    fig, axes = plt.subplots(1, len(NUMERIC_FEATURES), figsize=(20, 4))
    for ax, col in zip(axes, NUMERIC_FEATURES):
        sns.boxplot(data=df, x=TARGET, y=col, ax=ax)
        ax.set_title(f"{col} (outliers by class)")
    _save(fig, "numeric_boxplots.png")


def plot_target_correlation(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    corr = df.corr(numeric_only=True)[TARGET].drop(TARGET).sort_values()
    corr.plot(kind="barh", ax=ax, color=["#d9534f" if v < 0 else "#5cb85c" for v in corr])
    ax.set_title("Correlation of each feature with target")
    _save(fig, "target_correlation.png")


def main() -> None:
    df = load_clean_data()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    lines = [f"Shape (clean): {df.shape}", ""]
    if RAW_DATA_PATH.exists():
        raw = load_raw_data(RAW_DATA_PATH)
        lines += ["Missing values in RAW data:", raw.isna().sum().to_string(), ""]
    lines += ["Descriptive statistics:", df.describe().T.round(2).to_string(), ""]
    lines += ["Class distribution:", df[TARGET].value_counts().to_string()]
    summary = "\n".join(lines)
    (REPORTS_DIR / "eda_summary.txt").write_text(summary, encoding="utf-8")
    print(summary)

    plot_class_balance(df)
    plot_numeric_histograms(df)
    plot_categorical_counts(df)
    plot_correlation_heatmap(df)
    plot_boxplots(df)
    plot_target_correlation(df)
    logger.info("EDA complete. Figures in %s", FIGURES_DIR)


if __name__ == "__main__":
    main()
