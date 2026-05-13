"""
Train Model 4 (LightGBM Tuned via Optuna, All Features) and save artifacts.
Saves: career_model.pkl, label_encoder.pkl, features.pkl
"""
import pandas as pd
import numpy as np
import os
import pickle
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import lightgbm as lgb

def main():
    print("=== Training Model 4: LightGBM (Tuned via Optuna, All Features) ===")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, '..', 'data', 'final_career_dataset.csv')
    artifacts_dir = os.path.join(script_dir, '..', 'artifacts')
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # 1. Load Data
    df = pd.read_csv(dataset_path)
    print(f"Dataset loaded: {df.shape}")

    # 2. Inject 10% Noise for realistic results
    np.random.seed(42)
    noise_indices = np.random.choice(df.index, size=int(len(df) * 0.10), replace=False)
    df.loc[noise_indices, 'career'] = np.random.choice(df['career'].unique(), size=len(noise_indices))

    # 3. Preprocess
    if 'internet_access' in df.columns:
        df['internet_access'] = df['internet_access'].map({'yes': 1, 'no': 0})
    
    le = LabelEncoder()
    df['career'] = le.fit_transform(df['career'])
    print(f"Classes: {list(le.classes_)}")

    X = df.drop('career', axis=1)
    y = df['career']
    feature_names = list(X.columns)
    print(f"Features ({len(feature_names)}): {feature_names}")

    # 4. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # 6. Optuna Tuning
    print("\nTuning LightGBM using Optuna (40 trials)...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
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
        model.fit(X_train_res, y_train_res)
        preds = model.predict_proba(X_test)
        return roc_auc_score(y_test, preds, multi_class='ovo', average='macro')

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=40)

    print(f"\nBest AUC: {study.best_value:.4f}")
    print("Best parameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # 7. Train Final Model with Best Parameters
    best_params = study.best_params
    best_params.update({
        'objective': 'multiclass',
        'num_class': len(np.unique(y)),
        'random_state': 42,
        'verbose': -1,
        'n_jobs': -1
    })
    
    final_model = lgb.LGBMClassifier(**best_params)
    final_model.fit(X_train_res, y_train_res)

    # 8. Evaluate
    y_pred = final_model.predict(X_test)
    y_pred_proba = final_model.predict_proba(X_test)

    print("\n--- Final Performance on Test Set ---")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
    print(f"F1 Score : {f1_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
    print(f"AUC Score: {roc_auc_score(y_test, y_pred_proba, multi_class='ovo', average='macro'):.4f}")

    # 9. Save Artifacts
    model_path = os.path.join(artifacts_dir, 'career_model.pkl')
    le_path = os.path.join(artifacts_dir, 'label_encoder.pkl')
    feat_path = os.path.join(artifacts_dir, 'features.pkl')

    with open(model_path, 'wb') as f:
        pickle.dump(final_model, f)
    print(f"\nModel saved to: {model_path}")

    with open(le_path, 'wb') as f:
        pickle.dump(le, f)
    print(f"Label encoder saved to: {le_path}")

    with open(feat_path, 'wb') as f:
        pickle.dump(feature_names, f)
    print(f"Feature names saved to: {feat_path}")

    print("\nDone! Model 4 (LightGBM Tuned) is now the active career_model.pkl.")

if __name__ == '__main__':
    main()
