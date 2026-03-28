import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from preprocess import preprocess_data


def evaluate_models():
    # 1. Load Preprocessed Test Data
    _, X_test, _, y_test, _ = preprocess_data(
        "DATA/UNSW_NB15_training-set.csv",
        "DATA/UNSW_NB15_testing-set.csv",
    )

    # 2. Load the trained models
    print("Loading models...")
    rf_model = joblib.load("rf_model.joblib")
    svm_model = joblib.load("svm_model.joblib")

    models = {"Random Forest": rf_model, "SVM": svm_model}
    results = {}

    # 3. Calculate Metrics
    for name, model in models.items():
        print(f"Evaluating {name}...")
        predictions = model.predict(X_test)

        results[name] = {
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(y_test, predictions),
            "Recall": recall_score(y_test, predictions),
            "F1": f1_score(y_test, predictions),
        }

    # 4. Print Results Table
    results_df = pd.DataFrame(results).T
    print("\n--- Performance Metrics ---")
    print(results_df)

    # 5. Visualization: Compare Accuracy
    results_df["Accuracy"].plot(kind="bar", color=["skyblue", "salmon"])
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.savefig("accuracy_comparison.png")
    print("\nComparison chart saved as 'accuracy_comparison.png'")
    plt.close()


if __name__ == "__main__":
    evaluate_models()
