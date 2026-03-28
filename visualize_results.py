# visualize_results.py — uses same preprocessing and saved models as train_models.py
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from preprocess import preprocess_data

TRAIN_CSV = "DATA/UNSW_NB15_training-set.csv"
TEST_CSV = "DATA/UNSW_NB15_testing-set.csv"


def main():
    _, X_test, _, y_test, feature_names = preprocess_data(TRAIN_CSV, TEST_CSV)

    print("Loading trained models...")
    rf_model = joblib.load("rf_model.joblib")
    svm_model = joblib.load("svm_model.joblib")

    y_pred_rf = rf_model.predict(X_test)
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    ConfusionMatrixDisplay(cm_rf).plot()
    plt.title("Random Forest Confusion Matrix")
    plt.savefig("confusion_rf.png")
    plt.close()

    importances = rf_model.feature_importances_
    feat_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    top15 = feat_imp.sort_values(by="Importance", ascending=False).head(15)

    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=top15)
    plt.title("Top 15 Feature Importances (Random Forest)")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.close()

    y_pred_svm = svm_model.predict(X_test)
    cm_svm = confusion_matrix(y_test, y_pred_svm)
    ConfusionMatrixDisplay(cm_svm).plot()
    plt.title("SVM Confusion Matrix")
    plt.savefig("confusion_svm.png")
    plt.close()

    print(
        "Visualizations saved: confusion_rf.png, confusion_svm.png, feature_importance.png"
    )


if __name__ == "__main__":
    main()
