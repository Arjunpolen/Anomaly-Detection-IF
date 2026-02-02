from pathlib import Path
import pandas as pd
from sklearn.ensemble import IsolationForest

print("Training started")

# Project paths
project_root = Path(__file__).resolve().parents[1]
data_dir = project_root / "data"

X_path = data_dir / "X_processed.csv"
pred_path = data_dir / "predictions.csv"
score_path = data_dir / "scores.csv"

# Load processed data
X = pd.read_csv(X_path)

# Build model
model = IsolationForest(
    n_estimators=200,
    contamination=0.5,   # we'll tune later
    random_state=42
)

# Train model
model.fit(X)

print("Training completed")

# Predict anomalies
predictions = model.predict(X)          # 1 = normal, -1 = anomaly
scores = model.decision_function(X)     # higher = more normal

print("Normal samples:", (predictions == 1).sum())
print("Anomalous samples:", (predictions == -1).sum())

# Save outputs (in /data)
pd.Series(predictions, name="prediction").to_csv(pred_path, index=False)
pd.Series(scores, name="score").to_csv(score_path, index=False)

print(f"Saved: {pred_path.name}, {score_path.name}")
