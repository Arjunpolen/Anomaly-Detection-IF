import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

print("Preprocessing started")

# --------------------------------------------------
# Task 1: Load & inspect data
# --------------------------------------------------
data_path = "../data/NSL-KDD.csv"
df = pd.read_csv(data_path)

print("\nYes, data is loading correctly.")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
print("\nFirst 5 rows:")
print(df.head())

# --------------------------------------------------
# Task 2: Separate features and labels
# --------------------------------------------------
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

print("\nX shape:", X.shape)
print("y shape:", y.shape)

# --------------------------------------------------
# Task 3: Identify column types
# --------------------------------------------------
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print("\nCategorical columns:", categorical_cols)
print("Numerical columns:", numerical_cols)

# --------------------------------------------------
# Task 4 & 5: Encode + Scale
# --------------------------------------------------
categorical_pipeline = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
numerical_pipeline = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_pipeline, categorical_cols),
        ('num', numerical_pipeline, numerical_cols)
    ]
)

X_processed = preprocessor.fit_transform(X)

print("\nFinal processed feature shape:", X_processed.shape)

# --------------------------------------------------
# Task 6: Save processed data
# --------------------------------------------------
X_processed_df = pd.DataFrame(X_processed)
X_processed_df.to_csv("../data/X_processed.csv", index=False)
y.to_csv("../data/y.csv", index=False)

print("\nProcessed files saved:")
print("✔ X_processed.csv")
print("✔ y.csv")

print("\nPreprocessing DONE for today ✅")
