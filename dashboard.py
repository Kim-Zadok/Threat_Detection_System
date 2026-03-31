# dashboard.py — Streamlit UI using UNSW-NB15 preprocessing and saved models
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix

from preprocess import preprocess_data

TRAIN_CSV = "DATA/UNSW_NB15_training-set.csv"
TEST_CSV = "DATA/UNSW_NB15_testing-set.csv"


@st.cache_data(show_spinner="Loading and preprocessing data…")
def load_eval_bundle():
    _, X_test, _, y_test, feature_names = preprocess_data(TRAIN_CSV, TEST_CSV)
    return X_test, y_test, feature_names


@st.cache_resource
def load_models():
    return joblib.load("rf_model.joblib"), joblib.load("svm_model.joblib")


st.set_page_config(page_title="Threat Detection Dashboard", layout="wide")
st.title("Threat Detection System Dashboard")

try:
    X_test, y_test, feature_names = load_eval_bundle()
    rf_model, svm_model = load_models()
except FileNotFoundError as e:
    st.error(f"Missing file or model: {e}")
    st.stop()
except Exception as e:
    st.error(f"Failed to load data or models: {e}")
    st.stop()

y_test = np.asarray(y_test).ravel()

st.header("Model performance")
y_pred_rf = rf_model.predict(X_test)
y_pred_svm = svm_model.predict(X_test)
st.write("Random Forest accuracy:", float(accuracy_score(y_test, y_pred_rf)))
st.write("SVM accuracy:", float(accuracy_score(y_test, y_pred_svm)))

st.subheader("Random Forest confusion matrix")
cm_rf = confusion_matrix(y_test, y_pred_rf)
fig_rf, ax_rf = plt.subplots()
ConfusionMatrixDisplay(cm_rf).plot(ax=ax_rf)
st.pyplot(fig_rf)
plt.close(fig_rf)

st.subheader("SVM confusion matrix")
cm_svm = confusion_matrix(y_test, y_pred_svm)
fig_svm, ax_svm = plt.subplots()
ConfusionMatrixDisplay(cm_svm).plot(ax=ax_svm)
st.pyplot(fig_svm)
plt.close(fig_svm)

st.subheader("Feature importance (Random Forest, top 15)")
importances = rf_model.feature_importances_
feat_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances})
top15 = feat_imp.sort_values(by="Importance", ascending=False).head(15)
fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=top15, ax=ax_imp)
plt.tight_layout()
st.pyplot(fig_imp)
plt.close(fig_imp)

st.header("Demo: classify random test rows")
st.caption(
    "Models expect preprocessed numeric features (same shape as training). "
    "Below uses rows from the held-out test set."
)
n = st.slider("Number of rows", 1, 20, 5)
if st.button("Sample and predict"):
    rng = np.random.default_rng()
    idx = rng.choice(len(X_test), size=min(n, len(X_test)), replace=False)
    subset = X_test[idx]
    actual = y_test[idx]
    pred_rf = rf_model.predict(subset)
    pred_svm = svm_model.predict(subset)
    demo_df = pd.DataFrame(
        {
            "actual": np.where(actual == 1, "Attack", "Normal"),
            "random_forest": np.where(pred_rf == 1, "Attack", "Normal"),
            "svm": np.where(pred_svm == 1, "Attack", "Normal"),
        }
    )
    st.dataframe(demo_df, use_container_width=True)
