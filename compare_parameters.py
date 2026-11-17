
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

# Read reference labels
reference = pd.read_csv(
    "data/unifeedback_50_reference_labels.csv"
)

# Read Experiment B predictions
predictions = pd.read_csv(
    "data/experiment_C.csv"
)

# Remove unnecessary whitespace
reference["review"] = reference["review"].astype(str).str.strip()
predictions["review"] = predictions["review"].astype(str).str.strip()

# Validate that each review is unique
if reference["review"].duplicated().any():
    raise ValueError("Duplicate reviews in reference labels")

if predictions["review"].duplicated().any():
    raise ValueError("Duplicate reviews in experiment results")

# Match predictions to their reference labels
comparison = reference.merge(
    predictions[["review", "sentiment"]],
    on="review",
    how="left",
    validate="one_to_one"
)

# Include missing or invalid responses as failures
allowed_labels = ["Positive", "Negative", "Neutral", "Mixed"]

comparison["sentiment"] = (
    comparison["sentiment"].fillna("Analysis Failed")
)

comparison.loc[
    ~comparison["sentiment"].isin(allowed_labels),
    "sentiment"
] = "Analysis Failed"

# Calculate evaluation metrics
y_true = comparison["actual_sentiment"]
y_pred = comparison["sentiment"]

accuracy = accuracy_score(y_true, y_pred)

f1 = f1_score(
    y_true,
    y_pred,
    labels=allowed_labels,
    average="macro",
    zero_division=0
)

failed = (y_pred == "Analysis Failed").sum()

print("\n===== EXPERIMENT C RESULTS =====")
print("Temperature: 0.2")
print("Max Tokens: 300")
print(f"Total Reviews: {len(comparison)}")
print(f"Failed Analyses: {failed}")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Macro F1-score: {f1:.3f}")

# Show misclassified reviews
wrong = comparison[
    comparison["actual_sentiment"] != comparison["sentiment"]
]

print("\n===== WRONG PREDICTIONS =====")

if wrong.empty:
    print("No misclassified reviews.")
else:
    for _, row in wrong.iterrows():
        print("\nReview:", row["review"])
        print("Actual:", row["actual_sentiment"])
        print("Predicted:", row["sentiment"])
