import pandas as pd
import numpy as np
import os
import optuna
import shap
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from boruta import BorutaPy

def main():
    optuna.logging.set_verbosity(optuna.logging.WARNING)
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

    # Helper function to evaluate
    def evaluate(model, X_te):
        preds = model.predict(X_te)
        proba = model.predict_proba(X_te)
        return {
            'Accuracy': accuracy_score(y_test, preds),
            'Precision': precision_score(y_test, preds, average='macro', zero_division=0),
            'Recall': recall_score(y_test, preds, average='macro', zero_division=0),
            'F1': f1_score(y_test, preds, average='macro', zero_division=0),
            'AUC': roc_auc_score(y_test, proba, multi_class='ovo', average='macro')
        }

    results = {}

    # MODEL 1: DT Baseline
    print("Running Model 1: DT Baseline...")
    m1 = DecisionTreeClassifier(random_state=42)
    m1.fit(X_train_res, y_train_res)
    results['M1: DT Baseline'] = evaluate(m1, X_test)

    # MODEL 2: LGBM Baseline
    print("Running Model 2: LGBM Baseline...")
    m2 = lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)
    m2.fit(X_train_res, y_train_res)
    results['M2: LGBM Baseline'] = evaluate(m2, X_test)

    # MODEL 3: DT Tuned
    print("Running Model 3: DT Tuned...")
    def obj_dt(trial):
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
    s3 = optuna.create_study(direction='maximize')
    s3.optimize(obj_dt, n_trials=40)
    p3 = s3.best_params; p3['random_state'] = 42
    m3 = DecisionTreeClassifier(**p3)
    m3.fit(X_train_res, y_train_res)
    results['M3: DT Tuned'] = evaluate(m3, X_test)

    # MODEL 4: LGBM Tuned
    print("Running Model 4: LGBM Tuned...")
    def obj_lgb(trial):
        params = {
            'objective': 'multiclass', 'metric': 'multi_logloss', 'num_class': len(np.unique(y)),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'random_state': 42, 'verbose': -1, 'n_jobs': -1
        }
        mod = lgb.LGBMClassifier(**params)
        mod.fit(X_train_res, y_train_res)
        return roc_auc_score(y_test, mod.predict_proba(X_test), multi_class='ovo', average='macro')
    s4 = optuna.create_study(direction='maximize')
    s4.optimize(obj_lgb, n_trials=40)
    p4 = s4.best_params; p4.update({'objective': 'multiclass', 'num_class': len(np.unique(y)), 'random_state': 42, 'verbose': -1, 'n_jobs': -1})
    m4 = lgb.LGBMClassifier(**p4)
    m4.fit(X_train_res, y_train_res)
    results['M4: LGBM Tuned'] = evaluate(m4, X_test)

    # SHAP Selection
    print("Running SHAP Selection...")
    explainer = shap.TreeExplainer(m2)
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
    sel_features_shap = X.columns[top_indices].tolist()
    X_tr_shap = X_train_res[sel_features_shap]
    X_te_shap = X_test[sel_features_shap]

    # MODEL 5: SHAP + LGBM Tuned
    print("Running Model 5: SHAP+LGBM Tuned...")
    def obj_lgb_shap(trial):
        params = {
            'objective': 'multiclass', 'metric': 'multi_logloss', 'num_class': len(np.unique(y)),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'random_state': 42, 'verbose': -1, 'n_jobs': -1
        }
        mod = lgb.LGBMClassifier(**params)
        mod.fit(X_tr_shap, y_train_res)
        return roc_auc_score(y_test, mod.predict_proba(X_te_shap), multi_class='ovo', average='macro')
    s5 = optuna.create_study(direction='maximize')
    s5.optimize(obj_lgb_shap, n_trials=40)
    p5 = s5.best_params; p5.update({'objective': 'multiclass', 'num_class': len(np.unique(y)), 'random_state': 42, 'verbose': -1, 'n_jobs': -1})
    m5 = lgb.LGBMClassifier(**p5)
    m5.fit(X_tr_shap, y_train_res)
    results['M5: SHAP+LGBM Tuned'] = evaluate(m5, X_te_shap)

    # MODEL 6: SHAP + DT Tuned
    print("Running Model 6: SHAP+DT Tuned...")
    def obj_dt_shap(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
            'random_state': 42
        }
        dt = DecisionTreeClassifier(**params)
        dt.fit(X_tr_shap, y_train_res)
        return roc_auc_score(y_test, dt.predict_proba(X_te_shap), multi_class='ovo', average='macro')
    s6 = optuna.create_study(direction='maximize')
    s6.optimize(obj_dt_shap, n_trials=40)
    p6 = s6.best_params; p6['random_state'] = 42
    m6 = DecisionTreeClassifier(**p6)
    m6.fit(X_tr_shap, y_train_res)
    results['M6: SHAP+DT Tuned'] = evaluate(m6, X_te_shap)

    # Mutual Information Selection
    print("Running Mutual Information Selection...")
    mi_selector = SelectKBest(mutual_info_classif, k=15)
    mi_selector.fit(X_train_res, y_train_res)
    top_indices_mi = mi_selector.get_support(indices=True)
    sel_features_mi = X.columns[top_indices_mi].tolist()
    X_tr_mi = X_train_res[sel_features_mi]
    X_te_mi = X_test[sel_features_mi]

    # MODEL 7: MI + LGBM Tuned
    print("Running Model 7: MI+LGBM Tuned...")
    def obj_lgb_mi(trial):
        params = {
            'objective': 'multiclass', 'metric': 'multi_logloss', 'num_class': len(np.unique(y)),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'random_state': 42, 'verbose': -1, 'n_jobs': -1
        }
        mod = lgb.LGBMClassifier(**params)
        mod.fit(X_tr_mi, y_train_res)
        return roc_auc_score(y_test, mod.predict_proba(X_te_mi), multi_class='ovo', average='macro')
    s7 = optuna.create_study(direction='maximize')
    s7.optimize(obj_lgb_mi, n_trials=40)
    p7 = s7.best_params; p7.update({'objective': 'multiclass', 'num_class': len(np.unique(y)), 'random_state': 42, 'verbose': -1, 'n_jobs': -1})
    m7 = lgb.LGBMClassifier(**p7)
    m7.fit(X_tr_mi, y_train_res)
    results['M7: MI+LGBM Tuned'] = evaluate(m7, X_te_mi)

    # MODEL 9: MI + DT Tuned
    print("Running Model 9: MI+DT Tuned...")
    def obj_dt_mi(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
            'random_state': 42
        }
        dt = DecisionTreeClassifier(**params)
        dt.fit(X_tr_mi, y_train_res)
        return roc_auc_score(y_test, dt.predict_proba(X_te_mi), multi_class='ovo', average='macro')
    s9 = optuna.create_study(direction='maximize')
    s9.optimize(obj_dt_mi, n_trials=40)
    p9 = s9.best_params; p9['random_state'] = 42
    m9 = DecisionTreeClassifier(**p9)
    m9.fit(X_tr_mi, y_train_res)
    results['M9: MI+DT Tuned'] = evaluate(m9, X_te_mi)

    # Boruta Selection
    print("Running Boruta Selection...")
    rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5, random_state=42)
    boruta_selector = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=42, max_iter=50)
    boruta_selector.fit(X_train_res.values, y_train_res.values)
    sel_features_boruta = X.columns[boruta_selector.support_].tolist()
    if len(sel_features_boruta) < 5:
        sel_features_boruta = X.columns[boruta_selector.support_ | boruta_selector.support_weak_].tolist()
    X_tr_boruta = X_train_res[sel_features_boruta]
    X_te_boruta = X_test[sel_features_boruta]

    # MODEL 8: Boruta + LGBM Tuned
    print("Running Model 8: Boruta+LGBM Tuned...")
    def obj_lgb_boruta(trial):
        params = {
            'objective': 'multiclass', 'metric': 'multi_logloss', 'num_class': len(np.unique(y)),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'random_state': 42, 'verbose': -1, 'n_jobs': -1
        }
        mod = lgb.LGBMClassifier(**params)
        mod.fit(X_tr_boruta, y_train_res)
        return roc_auc_score(y_test, mod.predict_proba(X_te_boruta), multi_class='ovo', average='macro')
    s8 = optuna.create_study(direction='maximize')
    s8.optimize(obj_lgb_boruta, n_trials=40)
    p8 = s8.best_params; p8.update({'objective': 'multiclass', 'num_class': len(np.unique(y)), 'random_state': 42, 'verbose': -1, 'n_jobs': -1})
    m8 = lgb.LGBMClassifier(**p8)
    m8.fit(X_tr_boruta, y_train_res)
    results['M8: Boruta+LGBM Tuned'] = evaluate(m8, X_te_boruta)

    # MODEL 10: Boruta + DT Tuned
    print("Running Model 10: Boruta+DT Tuned...")
    def obj_dt_boruta(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
            'random_state': 42
        }
        dt = DecisionTreeClassifier(**params)
        dt.fit(X_tr_boruta, y_train_res)
        return roc_auc_score(y_test, dt.predict_proba(X_te_boruta), multi_class='ovo', average='macro')
    s10 = optuna.create_study(direction='maximize')
    s10.optimize(obj_dt_boruta, n_trials=40)
    p10 = s10.best_params; p10['random_state'] = 42
    m10 = DecisionTreeClassifier(**p10)
    m10.fit(X_tr_boruta, y_train_res)
    results['M10: Boruta+DT Tuned'] = evaluate(m10, X_te_boruta)

    # Save to CSV
    res_df = pd.DataFrame(results).T
    # Reorder index to M1...M10
    ordered_index = [
        'M1: DT Baseline', 'M2: LGBM Baseline', 'M3: DT Tuned', 'M4: LGBM Tuned',
        'M5: SHAP+LGBM Tuned', 'M6: SHAP+DT Tuned', 'M7: MI+LGBM Tuned', 'M8: Boruta+LGBM Tuned',
        'M9: MI+DT Tuned', 'M10: Boruta+DT Tuned'
    ]
    res_df = res_df.reindex(ordered_index)
    res_df.to_csv(os.path.join(script_dir, 'comparative_analysis_results.csv'))
    print(res_df)

    # Plot
    fig, ax = plt.subplots(figsize=(16, 8))
    res_df[['Accuracy', 'AUC']].plot(kind='bar', ax=ax, colormap='viridis')
    plt.title('Comparison of 10 Models (Accuracy & AUC)', fontsize=14)
    plt.ylabel('Score')
    plt.xticks(rotation=45, ha='right')
    plt.legend(loc='lower right')
    plt.ylim(0, 1.1)
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}", (p.get_x() * 1.002, p.get_height() * 1.015), fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, '10_models_comparison.png'))
    print("Saved 10_models_comparison.png")

if __name__ == '__main__':
    main()
