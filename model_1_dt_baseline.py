import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier

def main():
    print("=== MODEL 1: Decision Tree (No SHAP, No Tuning) ===")
    
    # 1. Load Data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, 'final_career_dataset.csv')
    df = pd.read_csv(dataset_path)

    # 2. Inject 10% Noise for Realistic Results
    np.random.seed(42)
    noise_indices = np.random.choice(df.index, size=int(len(df) * 0.10), replace=False)
    df.loc[noise_indices, 'career'] = np.random.choice(df['career'].unique(), size=len(noise_indices))

    # 3. Preprocess Categorical Variables
    if 'internet_access' in df.columns:
        df['internet_access'] = df['internet_access'].map({'yes': 1, 'no': 0})
    le = LabelEncoder()
    df['career'] = le.fit_transform(df['career'])

    X = df.drop('career', axis=1)
    y = df['career']

    # 4. Train-Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 5. Apply SMOTE to balance the training data
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # 6. Train Standard Decision Tree (using all features)
    print("Training standard Decision Tree on ALL features...")
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train_res, y_train_res)

    # 7. Evaluate Model
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    print("\n--- Final Performance ---")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
    print(f"F1 Score : {f1_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
    print(f"AUC Score: {roc_auc_score(y_test, y_pred_proba, multi_class='ovo', average='macro'):.4f}")

if __name__ == '__main__':
    main()
