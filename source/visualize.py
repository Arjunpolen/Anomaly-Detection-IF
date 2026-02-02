from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_fscore_support,
    accuracy_score,
)

# --------------------------
# Theme
# --------------------------
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

# Paths
project_root = Path(__file__).resolve().parents[1]
data_dir = project_root / "data"
plot_dir = project_root / "results" / "plots"
plot_dir.mkdir(parents=True, exist_ok=True)

y_path = data_dir / "y.csv"
pred_path = data_dir / "predictions.csv"
score_path = data_dir / "scores.csv"

# Load
y = pd.read_csv(y_path).iloc[:, 0]
pred = pd.read_csv(pred_path).iloc[:, 0]
scores = pd.read_csv(score_path).iloc[:, 0]

# True labels: normal=0, attack=1
y_true = (y != "normal").astype(int)

# Pred labels: anomaly(-1)=1, normal(1)=0
y_pred = (pred == -1).astype(int)

def _save(fig, filename):
    fig.tight_layout()
    fig.savefig(plot_dir / filename, dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)


# --------------------------
# Plot 1: Confusion Matrix
# --------------------------
cm = confusion_matrix(y_true, y_pred)
cm_cmap = LinearSegmentedColormap.from_list(
    "cm_blend", [THEME["panel"], THEME["accent"]]
)
fig, ax = plt.subplots(figsize=(6.2, 5.4))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Attack"])
disp.plot(values_format="d", cmap=cm_cmap, colorbar=False, ax=ax)
ax.set_title("Confusion Matrix (Isolation Forest)", pad=10)
ax.grid(False)
_save(fig, "confusion_matrix.png")

# ----------------------------------------
# Plot 2: Score distribution (Normal vs Attack)
# ----------------------------------------
df = pd.DataFrame({"score": scores, "label": y_true})
normal_scores = df[df["label"] == 0]["score"]
attack_scores = df[df["label"] == 1]["score"]

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
_save(fig, "score_distribution.png")

# -------------------------------------------------
# Plot 3: ROC Curve using scores (optional but nice)
# Note: lower scores = more anomalous => use -scores
# -------------------------------------------------
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
_save(fig, "roc_curve.png")

# ----------------------------------------
# Plot 4: Metrics snapshot
# ----------------------------------------
precision, recall, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average=None, labels=[1]
)
acc = accuracy_score(y_true, y_pred)
metrics = {
    "Accuracy": acc,
    "Precision": precision[0],
    "Recall": recall[0],
    "F1": f1[0],
}

fig, ax = plt.subplots(figsize=(7.0, 4.2))
names = list(metrics.keys())
values = list(metrics.values())
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
_save(fig, "metrics_snapshot.png")

print("Saved plots to:", plot_dir)
print(" - confusion_matrix.png")
print(" - score_distribution.png")
print(" - roc_curve.png")
print(" - metrics_snapshot.png")

# ----------------------------------------
# Plot 5: Dashboard (2x2 montage)
# ----------------------------------------
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

_save(fig, "dashboard.png")
print(" - dashboard.png")
