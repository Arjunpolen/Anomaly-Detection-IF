import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

project_root = Path(__file__).resolve().parents[1]
data_dir = project_root / "data"

y_path = data_dir / "y.csv"
pred_path = data_dir / "predictions.csv"

y = pd.read_csv(y_path).iloc[:, 0]
pred = pd.read_csv(pred_path).iloc[:, 0]

# Safety check
if len(y) != len(pred):
    raise ValueError(f"Length mismatch: y={len(y)} pred={len(pred)}. Check preprocessing row drops.")

y_true = (y != "normal").astype(int)      # normal=0, attack=1
y_pred = (pred == -1).astype(int)         # anomaly(-1)=1, normal(1)=0

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["Normal", "Attack"]))
