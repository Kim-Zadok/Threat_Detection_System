import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from preprocess import preprocess_data # Importing your previous work

def train_intelligence_tier():
    # 1. Load and Preprocess Data
    X_train, X_test, y_train, y_test, _ = preprocess_data(
        "DATA/UNSW_NB15_training-set.csv",
        "DATA/UNSW_NB15_testing-set.csv",
    )

    # 2. Train Random Forest
    print("\n--- Training Random Forest ---")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    joblib.dump(rf_model, 'rf_model.joblib')
    print("Random Forest model saved as 'rf_model.joblib'")

    # 3. Train SVM (using a subset for speed during presentation demo)
    print("\n--- Training SVM (using subset for speed) ---")
    # We take the first 10,000 samples to ensure the training finishes quickly
    svm_model = SVC(kernel='linear', random_state=42)
    svm_model.fit(X_train[:10000], y_train[:10000]) 
    joblib.dump(svm_model, 'svm_model.joblib')
    print("SVM model saved as 'svm_model.joblib'")

    print("\nIntelligence Tier training complete!")

if __name__ == "__main__":
    train_intelligence_tier()