import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(train_path, test_path):
    print("Loading datasets...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # 1. Data Cleaning: Drop redundant features (like 'id') and handle nulls
    if 'id' in train_df.columns:
        train_df = train_df.drop(columns=['id'])
        test_df = test_df.drop(columns=['id'])
        
    train_df = train_df.dropna()
    test_df = test_df.dropna()

    # The target column in this dataset is usually 'label' (0 for normal, 1 for attack)
    X_train = train_df.drop(columns=['label', 'attack_cat'], errors='ignore')
    y_train = train_df['label']
    
    X_test = test_df.drop(columns=['label', 'attack_cat'], errors='ignore')
    y_test = test_df['label']

    print("Transforming categorical data...")
    # 2. Transformation: One-Hot Encoding for categorical variables
    # Combine train and test to ensure identical columns after encoding
    combined_X = pd.concat([X_train, X_test])
    categorical_cols = combined_X.select_dtypes(include=['object']).columns
    combined_X = pd.get_dummies(combined_X, columns=categorical_cols)

    # Split back into train and test
    X_train = combined_X.iloc[:len(X_train)]
    X_test = combined_X.iloc[len(X_train):]

    print("Normalizing data...")
    # 3. Normalization: Apply StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Preprocessing complete!")
    return X_train_scaled, X_test_scaled, y_train, y_test

# Test the function if run directly
if __name__ == "__main__":
    # Adjust the paths to match where you saved your CSVs
    X_train, X_test, y_train, y_test = preprocess_data('data/UNSW_NB15_training-set.csv', 'data/UNSW_NB15_testing-set.csv')
    print(f"Training data shape: {X_train.shape}")