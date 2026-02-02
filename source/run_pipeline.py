import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler


THEME = {
    "bg": "#0b0f14",
    "panel": "#10161d",
    "text": "#e6edf3",
    "grid": "#2b3440",
    "accent": "#36c5f0",
    "accent2": "#f5a524",
    "accent3": "#7ee081",
    "danger": "#ff6b6b",
}

plt.rcParams.update(
    {
        "figure.facecolor": THEME["bg"],
        "axes.facecolor": THEME["panel"],
        "axes.edgecolor": THEME["grid"],
        "axes.labelcolor": THEME["text"],
        "text.color": THEME["text"],
        "xtick.color": THEME["text"],
        "ytick.color": THEME["text"],
        "grid.color": THEME["grid"],
        "grid.alpha": 0.35,
        "font.size": 11,
        "axes.titleweight": "bold",
    }
)


def _save(fig, filename, plot_dir):
    fig.tight_layout()
    fig.savefig(plot_dir / filename, dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)


def _find_label_column(df):
    exact_candidates = {"label", "class", "target", "attack", "outcome", "y"}
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in exact_candidates:
            return col
        if col_lower.endswith(("label", "class", "target", "outcome")):
            return col

    last_col = df.columns[-1]
    last_series = df[last_col]
    if last_series.dtype == object:
        return last_col
    unique_vals = pd.Series(last_series.dropna().unique())
    if unique_vals.empty:
        return None
    if set(unique_vals.tolist()).issubset({0, 1, -1}):
        return last_col
    unique_count = unique_vals.nunique()
    if unique_count <= 10:
        if unique_vals.apply(lambda v: float(v).is_integer()).all():
            return last_col
    return None


def _to_binary_labels(series):
    if series.dtype == object:
        return (series.astype(str).str.lower() != "normal").astype(int)
    unique = set(series.dropna().unique())
    if unique.issubset({0, 1}):
        return series.astype(int)
    return (series != 0).astype(int)


def _plot_placeholder(title, message, filename, plot_dir):
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.set_title(title, pad=10)
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        fontsize=12,
        color=THEME["text"],
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    _save(fig, filename, plot_dir)


def main():
    parser = argparse.ArgumentParser(description="Run Isolation Forest pipeline")
    parser.add_argument("--input", required=True, help="Path to input CSV")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    plot_dir = project_root / "results" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    label_col = _find_label_column(df)

    if label_col:
        y = df[label_col]
        X = df.drop(columns=[label_col])
    else:
        y = None
        X = df

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=["int64", "float64", "int32", "float32", "bool"]).columns.tolist()

    try:
        cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    transformers = []
    if categorical_cols:
        transformers.append(("cat", cat_encoder, categorical_cols))
    if numerical_cols:
        transformers.append(("num", StandardScaler(), numerical_cols))

    if not transformers:
        raise ValueError("No usable columns found in the uploaded CSV.")

    preprocessor = ColumnTransformer(transformers=transformers)

    X_processed = preprocessor.fit_transform(X)

    model = IsolationForest(n_estimators=200, contamination=0.5, random_state=42)
    model.fit(X_processed)

    predictions = model.predict(X_processed)
    scores = model.decision_function(X_processed)

    pd.DataFrame(X_processed).to_csv(data_dir / "X_processed.csv", index=False)
    pd.Series(predictions, name="prediction").to_csv(data_dir / "predictions.csv", index=False)
    pd.Series(scores, name="score").to_csv(data_dir / "scores.csv", index=False)
    if y is not None:
        y.to_csv(data_dir / "y.csv", index=False)

    summary = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing": int(df.isna().sum().sum()),
        "numeric_features": int(len(numerical_cols)),
        "normal_count": int((predictions == 1).sum()),
        "anomaly_count": int((predictions == -1).sum()),
        "has_labels": bool(y is not None),
    }

    if y is not None:
        y_true = _to_binary_labels(y)
        y_pred = (predictions == -1).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        cm_cmap = LinearSegmentedColormap.from_list(
            "cm_blend", [THEME["panel"], THEME["accent"]]
        )
        fig, ax = plt.subplots(figsize=(6.2, 5.4))
        unique_labels = sorted(pd.unique(y_true))
        display_labels = ["Normal", "Attack"] if len(unique_labels) == 2 else [
            "Normal" if unique_labels[0] == 0 else "Attack"
        ]
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
        disp.plot(values_format="d", cmap=cm_cmap, colorbar=False, ax=ax)
        ax.set_title("Confusion Matrix (Isolation Forest)", pad=10)
        ax.grid(False)
        _save(fig, "confusion_matrix.png", plot_dir)

        df_scores = pd.DataFrame({"score": scores, "label": y_true})
        normal_scores = df_scores[df_scores["label"] == 0]["score"]
        attack_scores = df_scores[df_scores["label"] == 1]["score"]

        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        ax.hist(
            normal_scores,
            bins=70,
            alpha=0.65,
            label="Normal",
            color=THEME["accent3"],
            edgecolor=THEME["panel"],
        )
        ax.hist(
            attack_scores,
            bins=70,
            alpha=0.65,
            label="Attack",
            color=THEME["danger"],
            edgecolor=THEME["panel"],
        )
        ax.set_title("Anomaly Score Distribution (higher = more normal)", pad=10)
        ax.set_xlabel("decision_function score")
        ax.set_ylabel("count")
        ax.grid(True, axis="y")
        ax.legend(frameon=False)
        _save(fig, "score_distribution.png", plot_dir)

        if len(set(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, -scores)
            roc_auc = auc(fpr, tpr)
            fig, ax = plt.subplots(figsize=(6.4, 5.2))
            ax.plot(fpr, tpr, color=THEME["accent"], lw=2.2, label=f"AUC = {roc_auc:.3f}")
            ax.fill_between(fpr, tpr, color=THEME["accent"], alpha=0.12)
            ax.plot([0, 1], [0, 1], linestyle="--", color=THEME["grid"], lw=1.2)
            ax.set_title("ROC Curve (using anomaly score)", pad=10)
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.legend(frameon=False, loc="lower right")
            ax.grid(True)
            _save(fig, "roc_curve.png", plot_dir)
        else:
            _plot_placeholder(
                "ROC Curve",
                "Need both Normal and Attack labels",
                "roc_curve.png",
                plot_dir,
            )

        acc = accuracy_score(y_true, y_pred)
        if len(set(y_true)) > 1:
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="binary", zero_division=0
            )
        else:
            precision, recall, f1 = 0.0, 0.0, 0.0
        summary["metrics"] = {
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        names = ["Accuracy", "Precision", "Recall", "F1"]
        values = [acc, precision, recall, f1]
        bars = ax.bar(names, values, color=THEME["accent2"])
        ax.set_ylim(0, 1)
        ax.set_title("Attack Detection Metrics", pad=10)
        ax.grid(True, axis="y")
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(val + 0.03, 0.98),
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
                color=THEME["text"],
            )
        _save(fig, "metrics_snapshot.png", plot_dir)
    else:
        _plot_placeholder(
            "Confusion Matrix",
            "Label column not found",
            "confusion_matrix.png",
            plot_dir,
        )

        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        ax.hist(scores, bins=70, alpha=0.75, color=THEME["accent3"], edgecolor=THEME["panel"])
        ax.set_title("Anomaly Score Distribution", pad=10)
        ax.set_xlabel("decision_function score")
        ax.set_ylabel("count")
        ax.grid(True, axis="y")
        _save(fig, "score_distribution.png", plot_dir)

        _plot_placeholder(
            "ROC Curve",
            "Label column not found",
            "roc_curve.png",
            plot_dir,
        )

        anomaly_rate = summary["anomaly_count"] / max(summary["rows"], 1)
        normal_rate = summary["normal_count"] / max(summary["rows"], 1)
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        names = ["Normal Rate", "Anomaly Rate"]
        values = [normal_rate, anomaly_rate]
        bars = ax.bar(names, values, color=THEME["accent2"])
        ax.set_ylim(0, 1)
        ax.set_title("Prediction Distribution", pad=10)
        ax.grid(True, axis="y")
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(val + 0.03, 0.98),
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
                color=THEME["text"],
            )
        _save(fig, "metrics_snapshot.png", plot_dir)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Network Intrusion Detection - Visual Dashboard", fontsize=16, weight="bold")

    img_cm = plt.imread(plot_dir / "confusion_matrix.png")
    img_dist = plt.imread(plot_dir / "score_distribution.png")
    img_roc = plt.imread(plot_dir / "roc_curve.png")
    img_metrics = plt.imread(plot_dir / "metrics_snapshot.png")

    axes[0, 0].imshow(img_cm)
    axes[0, 0].set_title("Confusion Matrix")
    axes[0, 1].imshow(img_dist)
    axes[0, 1].set_title("Score Distribution")
    axes[1, 0].imshow(img_roc)
    axes[1, 0].set_title("ROC Curve")
    axes[1, 1].imshow(img_metrics)
    axes[1, 1].set_title("Metrics Snapshot")

    for ax in axes.ravel():
        ax.axis("off")

    _save(fig, "dashboard.png", plot_dir)

    summary_path = project_root / "results" / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
