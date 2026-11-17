
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Read files
actual = pd.read_csv(
    "data/unifeedback_50_reference_labels.csv"
)

predicted = pd.read_csv(
    "data/unifeedback_analysis.csv"
)

# Clean review text
actual["review"] = actual["review"].str.strip()
predicted["review"] = predicted["review"].str.strip()

# Check duplicates
if actual["review"].duplicated().any():
    raise ValueError("Duplicate reference reviews")

if predicted["review"].duplicated().any():
    raise ValueError("Duplicate predictions")

# Match reviews
data = actual.merge(
    predicted[["review", "sentiment"]],
    on="review",
    how="left",
    validate="one_to_one"
)

print("\n===== DEBUG =====")

print("Reference reviews:", len(actual))
print("Predicted reviews:", len(predicted))

print("\nFirst reference review:")
print(repr(actual["review"].iloc[0]))

print("\nFirst predicted review:")
print(repr(predicted["review"].iloc[0]))

print("\nMatched predictions:")
print(data["sentiment"].notna().sum())

labels = [
    "Positive",
    "Negative",
    "Neutral",
    "Mixed"
]

# Missing or invalid predictions count as failures
data["sentiment"] = data["sentiment"].fillna(
    "Analysis Failed"
)

data.loc[
    ~data["sentiment"].isin(labels),
    "sentiment"
] = "Analysis Failed"

# Calculate metrics
accuracy = accuracy_score(
    data["actual_sentiment"],
    data["sentiment"]
)

successful = (
    data["sentiment"] != "Analysis Failed"
).sum()

failed = len(data) - successful

print("\n===== FINAL MODEL EVALUATION =====")

print(f"Total Reviews: {len(data)}")
print(f"Successful Analyses: {successful}")
print(f"Failed Analyses: {failed}")
print(f"Overall Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")

print(classification_report(
    data["actual_sentiment"],
    data["sentiment"],
    labels=labels,
    zero_division=0
))

print("\nConfusion Matrix:")

matrix = confusion_matrix(
    data["actual_sentiment"],
    data["sentiment"],
    labels=labels + ["Analysis Failed"]
)

print(pd.DataFrame(
    matrix,
    index=labels + ["Analysis Failed"],
    columns=labels + ["Analysis Failed"]
))

data["correct"] = (
    data["actual_sentiment"] == data["sentiment"]
)

data.to_csv(
    "data/final_evaluation_results.csv",
    index=False
)

print("\nEvaluation completed!")

# ==========================================
# ERROR ANALYSIS
# ==========================================

print("\n===== MISCLASSIFIED REVIEWS =====")

wrong_predictions = data[
    data["correct"] == False
]

if wrong_predictions.empty:
    print("No misclassified reviews.")

else:
    for _, row in wrong_predictions.iterrows():

        print("\nReview:")
        print(row["review"])

        print("\nActual Sentiment:")
        print(row["actual_sentiment"])

        print("\nAI Prediction:")
        print(row["sentiment"])

        print("-" * 40)
