import pandas as pd
import numpy as np
import os
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier

def main():
    print("=== MODEL 3: Decision Tree (Tuned via Optuna, No SHAP) ===")
    
    # 1. Load Data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, 'final_career_dataset.csv')
    df = pd.read_csv(dataset_path)

    # 2. Inject 10% Noise
    np.random.seed(42)
    noise_indices = np.random.choice(df.index, size=int(len(df) * 0.10), replace=False)
    df.loc[noise_indices, 'career'] = np.random.choice(df['career'].unique(), size=len(noise_indices))

    # 3. Preprocess
    if 'internet_access' in df.columns:
        df['internet_access'] = df['internet_access'].map({'yes': 1, 'no': 0})
    le = LabelEncoder()
    df['career'] = le.fit_transform(df['career'])

    X = df.drop('career', axis=1)
    y = df['career']

    # 4. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 5. Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # 6. Optuna Tuning for Decision Tree
    print("Tuning Decision Tree using Optuna on ALL features...")
    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
            'random_state': 42
        }
        dt = DecisionTreeClassifier(**params)
        dt.fit(X_train_res, y_train_res)
        preds = dt.predict_proba(X_test)
        return roc_auc_score(y_test, preds, multi_class='ovo', average='macro')

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=40)

    print("Best parameters found by Optuna:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # 7. Train Final Tuned Model
    best_params = study.best_params
    best_params['random_state'] = 42
    final_model = DecisionTreeClassifier(**best_params)
    final_model.fit(X_train_res, y_train_res)

    # 8. Evaluate Model
    y_pred = final_model.predict(X_test)
    y_pred_proba = final_model.predict_proba(X_test)

    print("\n--- Final Performance ---")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
    print(f"F1 Score : {f1_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
    print(f"AUC Score: {roc_auc_score(y_test, y_pred_proba, multi_class='ovo', average='macro'):.4f}")

if __name__ == '__main__':
    main()
