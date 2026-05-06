import pandas as pd
import numpy as np
import os
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy

def main():
    print("=== MODEL 8: Boruta Feature Selection + LightGBM + Tune ===")
    
    # 1. Load Data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, 'final_career_dataset.csv')
    df = pd.read_csv(dataset_path)

    # 2. Inject 10% Noise for generalizability
    np.random.seed(42)
    noise_indices = np.random.choice(df.index, size=int(len(df) * 0.10), replace=False)
    df.loc[noise_indices, 'career'] = np.random.choice(df['career'].unique(), size=len(noise_indices))

    # 3. Preprocess categorical features and encode labels
    if 'internet_access' in df.columns:
        df['internet_access'] = df['internet_access'].map({'yes': 1, 'no': 0})
    le = LabelEncoder()
    df['career'] = le.fit_transform(df['career'])

    X = df.drop('career', axis=1)
    y = df['career']

    # 4. Train-Test Split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 5. Apply SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # 6. Boruta Feature Selection
    print("Running Boruta Feature Selection...")
    # Define a random forest estimator required by Boruta
    rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5, random_state=42)
    
    # Define Boruta feature selection method
    boruta_selector = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=42, max_iter=50)
    
    # Fit Boruta (it needs numpy arrays)
    boruta_selector.fit(X_train_res.values, y_train_res.values)
    
    # Check selected features
    selected_features = X.columns[boruta_selector.support_].tolist()
    
    # If Boruta is too strict and selects very few/no features, include tentative ones or fallback to top 15
    if len(selected_features) < 5:
        print("Boruta selected very few features. Including tentative features as well.")
        selected_features = X.columns[boruta_selector.support_ | boruta_selector.support_weak_].tolist()
    
    print(f"Selected Features by Boruta ({len(selected_features)}): {selected_features}")

    # Reduce dataset to selected features
    X_train_selected = X_train_res[selected_features]
    X_test_selected = X_test[selected_features]

    # 7. Optuna Tuning for LightGBM on selected features
    print("Tuning LightGBM using Optuna on Boruta selected features...")
    def objective(trial):
        params = {
            'objective': 'multiclass',
            'metric': 'multi_logloss',
            'num_class': len(np.unique(y)),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'random_state': 42,
            'verbose': -1,
            'n_jobs': -1
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train_selected, y_train_res)
        preds = model.predict_proba(X_test_selected)
        return roc_auc_score(y_test, preds, multi_class='ovo', average='macro')

    # Suppress optuna logging for cleaner output
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=40)

    print("Best parameters found by Optuna:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # 8. Train Final Tuned Model
    best_params = study.best_params
    best_params.update({'objective': 'multiclass', 'num_class': len(np.unique(y)), 'random_state': 42, 'verbose': -1, 'n_jobs': -1})
    final_model = lgb.LGBMClassifier(**best_params)
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
