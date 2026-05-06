import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier

def main():
    print("Loading dataset...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, 'final_career_dataset.csv')
    df = pd.read_csv(dataset_path)

    # --- INJECT REALISM (NOISE) ---
    print("Injecting random noise (same 10% as LightGBM)...")
    np.random.seed(42)
    noise_ratio = 0.10
    n_noise = int(len(df) * noise_ratio)
    noise_indices = np.random.choice(df.index, size=n_noise, replace=False)
    df.loc[noise_indices, 'career'] = np.random.choice(df['career'].unique(), size=n_noise)

    # Encoding
    print("Preprocessing data...")
    if 'internet_access' in df.columns:
        df['internet_access'] = df['internet_access'].map({'yes': 1, 'no': 0})
    
    le = LabelEncoder()
    df['career'] = le.fit_transform(df['career'])

    X = df.drop('career', axis=1)
    y = df['career']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Applying SMOTE for data balancing on training set...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Use the same top 15 features from SHAP for a fair comparison
    selected_features = ['art_interest', 'business_interest', 'analytical_skill', 'coding_skill', 'tech_interest', 'final_grade', 'neuroticism', 'extraversion', 'grade1', 'openness', 'conscientiousness', 'study_hours', 'participation', 'grade2', 'agreeableness']
    
    X_train_selected = X_train_resampled[selected_features]
    X_test_selected = X_test[selected_features]

    print("Training Decision Tree Classifier...")
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train_selected, y_train_resampled)

    # Predictions
    y_pred = model.predict(X_test_selected)
    y_pred_proba = model.predict_proba(X_test_selected)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovo', average='macro')

    print("\n--- Final Performance Metrics: Decision Tree (Test Set) ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f} (Macro)")
    print(f"Recall   : {rec:.4f} (Macro)")
    print(f"F1 Score : {f1:.4f} (Macro)")
    print(f"AUC Score: {auc:.4f} (Macro)")

if __name__ == '__main__':
    main()
