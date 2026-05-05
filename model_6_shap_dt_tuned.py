import pandas as pd
import numpy as np
import os
import optuna
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
from sklearn.tree import DecisionTreeClassifier

def main():
    print("=== MODEL 6: SHAP + Feature Selection + Decision Tree + Tune ===")
    
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

    # 6. SHAP Feature Selection (Using LGBM as explainer)
    print("Running SHAP Feature Selection...")
    baseline_model = lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)
    baseline_model.fit(X_train_res, y_train_res)

    explainer = shap.TreeExplainer(baseline_model)
    shap_sample = X_train_res.sample(n=min(1000, len(X_train_res)), random_state=42)
    shap_values = explainer.shap_values(shap_sample)
    
    if isinstance(shap_values, list):
        mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    elif len(shap_values.shape) == 3:
        mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2))
    else:
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

    top_n = 15
    top_indices = np.argsort(mean_abs_shap)[::-1][:top_n]
    selected_features = X.columns[top_indices].tolist()
    print(f"Selected Top {top_n} Features: {selected_features}")

    X_train_selected = X_train_res[selected_features]
    X_test_selected = X_test[selected_features]

    # 7. Optuna Tuning for Decision Tree on selected features
    print("Tuning Decision Tree using Optuna on SHAP selected features...")
    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
            'random_state': 42
        }
        dt = DecisionTreeClassifier(**params)
        dt.fit(X_train_selected, y_train_res)
        preds = dt.predict_proba(X_test_selected)
        return roc_auc_score(y_test, preds, multi_class='ovo', average='macro')

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=40)

    print("Best parameters found by Optuna:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # 8. Train Final Tuned Model
    best_params = study.best_params
    best_params['random_state'] = 42
    final_model = DecisionTreeClassifier(**best_params)
    final_model.fit(X_train_selected, y_train_res)

    # 9. Evaluate Model
    y_pred = final_model.predict(X_test_selected)
    y_pred_proba = final_model.predict_proba(X_test_selected)

    print("\n--- Final Performance ---")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
    print(f"F1 Score : {f1_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
    print(f"AUC Score: {roc_auc_score(y_test, y_pred_proba, multi_class='ovo', average='macro'):.4f}")

if __name__ == '__main__':
    main()
