"""
Benchmark all 10 models and print a comparison table.
Runs each model's pipeline and collects metrics.
"""
import pandas as pd
import numpy as np
import os, sys, time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE

# ── Load & preprocess data (shared across all models) ──
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, '..', 'data', 'final_career_dataset.csv')
df_orig = pd.read_csv(dataset_path)

# Inject 10% noise (same seed as all models)
np.random.seed(42)
noise_indices = np.random.choice(df_orig.index, size=int(len(df_orig) * 0.10), replace=False)
df_orig.loc[noise_indices, 'career'] = np.random.choice(df_orig['career'].unique(), size=len(noise_indices))

if 'internet_access' in df_orig.columns:
    df_orig['internet_access'] = df_orig['internet_access'].map({'yes': 1, 'no': 0})
le = LabelEncoder()
df_orig['career'] = le.fit_transform(df_orig['career'])

X = df_orig.drop('career', axis=1)
y = df_orig['career']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

def evaluate(model, X_te, y_te):
    y_pred = model.predict(X_te)
    y_proba = model.predict_proba(X_te)
    return {
        'Accuracy':  accuracy_score(y_te, y_pred),
        'Precision': precision_score(y_te, y_pred, average='macro', zero_division=0),
        'Recall':    recall_score(y_te, y_pred, average='macro', zero_division=0),
        'F1':        f1_score(y_te, y_pred, average='macro', zero_division=0),
        'AUC':       roc_auc_score(y_te, y_proba, multi_class='ovo', average='macro'),
    }

results = []

# ═══════════════════════════════════════════════════
# MODEL 1: Decision Tree Baseline (all features)
# ═══════════════════════════════════════════════════
from sklearn.tree import DecisionTreeClassifier
print("Running Model 1: DT Baseline...")
t0 = time.time()
m1 = DecisionTreeClassifier(random_state=42)
m1.fit(X_train_res, y_train_res)
t1 = time.time()
r = evaluate(m1, X_test, y_test)
r['Model'] = 'M1: DT Baseline'
r['Time(s)'] = round(t1-t0, 2)
r['Features'] = 'All'
results.append(r)

# ═══════════════════════════════════════════════════
# MODEL 2: LightGBM Baseline (all features)
# ═══════════════════════════════════════════════════
import lightgbm as lgb
print("Running Model 2: LGBM Baseline...")
t0 = time.time()
m2 = lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)
m2.fit(X_train_res, y_train_res)
t1 = time.time()
r = evaluate(m2, X_test, y_test)
r['Model'] = 'M2: LGBM Baseline'
r['Time(s)'] = round(t1-t0, 2)
r['Features'] = 'All'
results.append(r)

# ═══════════════════════════════════════════════════
# MODEL 3: DT Tuned (Optuna, all features)
# ═══════════════════════════════════════════════════
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

print("Running Model 3: DT Tuned (Optuna)...")
t0 = time.time()
def obj_m3(trial):
    p = {
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
        'random_state': 42
    }
    dt = DecisionTreeClassifier(**p)
    dt.fit(X_train_res, y_train_res)
    return roc_auc_score(y_test, dt.predict_proba(X_test), multi_class='ovo', average='macro')
s3 = optuna.create_study(direction='maximize')
s3.optimize(obj_m3, n_trials=40)
bp3 = s3.best_params; bp3['random_state'] = 42
m3 = DecisionTreeClassifier(**bp3)
m3.fit(X_train_res, y_train_res)
t1 = time.time()
r = evaluate(m3, X_test, y_test)
r['Model'] = 'M3: DT Tuned'
r['Time(s)'] = round(t1-t0, 2)
r['Features'] = 'All'
results.append(r)

# ═══════════════════════════════════════════════════
# MODEL 4: LGBM Tuned (Optuna, all features)
# ═══════════════════════════════════════════════════
print("Running Model 4: LGBM Tuned (Optuna)...")
t0 = time.time()
def obj_m4(trial):
    p = {
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
    m = lgb.LGBMClassifier(**p); m.fit(X_train_res, y_train_res)
    return roc_auc_score(y_test, m.predict_proba(X_test), multi_class='ovo', average='macro')
s4 = optuna.create_study(direction='maximize')
s4.optimize(obj_m4, n_trials=40)
bp4 = s4.best_params
bp4.update({'objective':'multiclass','num_class':len(np.unique(y)),'random_state':42,'verbose':-1,'n_jobs':-1})
m4 = lgb.LGBMClassifier(**bp4); m4.fit(X_train_res, y_train_res)
t1 = time.time()
r = evaluate(m4, X_test, y_test)
r['Model'] = 'M4: LGBM Tuned'
r['Time(s)'] = round(t1-t0, 2)
r['Features'] = 'All'
results.append(r)

# ═══════════════════════════════════════════════════
# SHAP feature selection (shared for M5 & M6)
# ═══════════════════════════════════════════════════
print("Running SHAP feature selection...")
import shap
baseline = lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)
baseline.fit(X_train_res, y_train_res)
explainer = shap.TreeExplainer(baseline)
shap_sample = X_train_res.sample(n=min(1000, len(X_train_res)), random_state=42)
shap_values = explainer.shap_values(shap_sample)
if isinstance(shap_values, list):
    mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
elif len(shap_values.shape) == 3:
    mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2))
else:
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
top_idx = np.argsort(mean_abs_shap)[::-1][:15]
shap_feats = X.columns[top_idx].tolist()
print(f"  SHAP top 15: {shap_feats}")
X_tr_shap = X_train_res[shap_feats]; X_te_shap = X_test[shap_feats]

# ═══════════════════════════════════════════════════
# MODEL 5: SHAP + LGBM Tuned
# ═══════════════════════════════════════════════════
print("Running Model 5: SHAP + LGBM Tuned...")
t0 = time.time()
def obj_m5(trial):
    p = {
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
    m = lgb.LGBMClassifier(**p); m.fit(X_tr_shap, y_train_res)
    return roc_auc_score(y_test, m.predict_proba(X_te_shap), multi_class='ovo', average='macro')
s5 = optuna.create_study(direction='maximize')
s5.optimize(obj_m5, n_trials=40)
bp5 = s5.best_params
bp5.update({'objective':'multiclass','num_class':len(np.unique(y)),'random_state':42,'verbose':-1,'n_jobs':-1})
m5 = lgb.LGBMClassifier(**bp5); m5.fit(X_tr_shap, y_train_res)
t1 = time.time()
r = evaluate(m5, X_te_shap, y_test)
r['Model'] = 'M5: SHAP+LGBM Tuned'
r['Time(s)'] = round(t1-t0, 2)
r['Features'] = 'SHAP Top 15'
results.append(r)

# ═══════════════════════════════════════════════════
# MODEL 6: SHAP + DT Tuned
# ═══════════════════════════════════════════════════
print("Running Model 6: SHAP + DT Tuned...")
t0 = time.time()
def obj_m6(trial):
    p = {
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
        'random_state': 42
    }
    dt = DecisionTreeClassifier(**p); dt.fit(X_tr_shap, y_train_res)
    return roc_auc_score(y_test, dt.predict_proba(X_te_shap), multi_class='ovo', average='macro')
s6 = optuna.create_study(direction='maximize')
s6.optimize(obj_m6, n_trials=40)
bp6 = s6.best_params; bp6['random_state'] = 42
m6 = DecisionTreeClassifier(**bp6); m6.fit(X_tr_shap, y_train_res)
t1 = time.time()
r = evaluate(m6, X_te_shap, y_test)
r['Model'] = 'M6: SHAP+DT Tuned'
r['Time(s)'] = round(t1-t0, 2)
r['Features'] = 'SHAP Top 15'
results.append(r)

# ═══════════════════════════════════════════════════
# MI feature selection (shared for M7 & M9)
# ═══════════════════════════════════════════════════
print("Running MI feature selection...")
from sklearn.feature_selection import mutual_info_classif, SelectKBest
mi_sel = SelectKBest(mutual_info_classif, k=15)
mi_sel.fit(X_train_res, y_train_res)
mi_feats = X.columns[mi_sel.get_support(indices=True)].tolist()
print(f"  MI top 15: {mi_feats}")
X_tr_mi = X_train_res[mi_feats]; X_te_mi = X_test[mi_feats]

# ═══════════════════════════════════════════════════
# MODEL 7: MI + LGBM Tuned
# ═══════════════════════════════════════════════════
print("Running Model 7: MI + LGBM Tuned...")
t0 = time.time()
def obj_m7(trial):
    p = {
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
    m = lgb.LGBMClassifier(**p); m.fit(X_tr_mi, y_train_res)
    return roc_auc_score(y_test, m.predict_proba(X_te_mi), multi_class='ovo', average='macro')
s7 = optuna.create_study(direction='maximize')
s7.optimize(obj_m7, n_trials=40)
bp7 = s7.best_params
bp7.update({'objective':'multiclass','num_class':len(np.unique(y)),'random_state':42,'verbose':-1,'n_jobs':-1})
m7 = lgb.LGBMClassifier(**bp7); m7.fit(X_tr_mi, y_train_res)
t1 = time.time()
r = evaluate(m7, X_te_mi, y_test)
r['Model'] = 'M7: MI+LGBM Tuned'
r['Time(s)'] = round(t1-t0, 2)
r['Features'] = 'MI Top 15'
results.append(r)

# ═══════════════════════════════════════════════════
# MODEL 9: MI + DT Tuned
# ═══════════════════════════════════════════════════
print("Running Model 9: MI + DT Tuned...")
t0 = time.time()
def obj_m9(trial):
    p = {
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
        'random_state': 42
    }
    dt = DecisionTreeClassifier(**p); dt.fit(X_tr_mi, y_train_res)
    return roc_auc_score(y_test, dt.predict_proba(X_te_mi), multi_class='ovo', average='macro')
s9 = optuna.create_study(direction='maximize')
s9.optimize(obj_m9, n_trials=40)
bp9 = s9.best_params; bp9['random_state'] = 42
m9 = DecisionTreeClassifier(**bp9); m9.fit(X_tr_mi, y_train_res)
t1 = time.time()
r = evaluate(m9, X_te_mi, y_test)
r['Model'] = 'M9: MI+DT Tuned'
r['Time(s)'] = round(t1-t0, 2)
r['Features'] = 'MI Top 15'
results.append(r)

# ═══════════════════════════════════════════════════
# Boruta feature selection (shared for M8 & M10)
# ═══════════════════════════════════════════════════
print("Running Boruta feature selection (this takes a minute)...")
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5, random_state=42)
boruta = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=42, max_iter=50)
boruta.fit(X_train_res.values, y_train_res.values)
boruta_feats = X.columns[boruta.support_].tolist()
if len(boruta_feats) < 5:
    boruta_feats = X.columns[boruta.support_ | boruta.support_weak_].tolist()
print(f"  Boruta selected ({len(boruta_feats)}): {boruta_feats}")
X_tr_bor = X_train_res[boruta_feats]; X_te_bor = X_test[boruta_feats]

# ═══════════════════════════════════════════════════
# MODEL 8: Boruta + LGBM Tuned
# ═══════════════════════════════════════════════════
print("Running Model 8: Boruta + LGBM Tuned...")
t0 = time.time()
def obj_m8(trial):
    p = {
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
    m = lgb.LGBMClassifier(**p); m.fit(X_tr_bor, y_train_res)
    return roc_auc_score(y_test, m.predict_proba(X_te_bor), multi_class='ovo', average='macro')
s8 = optuna.create_study(direction='maximize')
s8.optimize(obj_m8, n_trials=40)
bp8 = s8.best_params
bp8.update({'objective':'multiclass','num_class':len(np.unique(y)),'random_state':42,'verbose':-1,'n_jobs':-1})
m8 = lgb.LGBMClassifier(**bp8); m8.fit(X_tr_bor, y_train_res)
t1 = time.time()
r = evaluate(m8, X_te_bor, y_test)
r['Model'] = 'M8: Boruta+LGBM Tuned'
r['Time(s)'] = round(t1-t0, 2)
r['Features'] = f'Boruta ({len(boruta_feats)})'
results.append(r)

# ═══════════════════════════════════════════════════
# MODEL 10: Boruta + DT Tuned
# ═══════════════════════════════════════════════════
print("Running Model 10: Boruta + DT Tuned...")
t0 = time.time()
def obj_m10(trial):
    p = {
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
        'random_state': 42
    }
    dt = DecisionTreeClassifier(**p); dt.fit(X_tr_bor, y_train_res)
    return roc_auc_score(y_test, dt.predict_proba(X_te_bor), multi_class='ovo', average='macro')
s10 = optuna.create_study(direction='maximize')
s10.optimize(obj_m10, n_trials=40)
bp10 = s10.best_params; bp10['random_state'] = 42
m10 = DecisionTreeClassifier(**bp10); m10.fit(X_tr_bor, y_train_res)
t1 = time.time()
r = evaluate(m10, X_te_bor, y_test)
r['Model'] = 'M10: Boruta+DT Tuned'
r['Time(s)'] = round(t1-t0, 2)
r['Features'] = f'Boruta ({len(boruta_feats)})'
results.append(r)

# ═══════════════════════════════════════════════════
# PRINT FINAL COMPARISON TABLE
# ═══════════════════════════════════════════════════
cols = ['Model', 'Features', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'Time(s)']
df_results = pd.DataFrame(results)[cols]
df_results = df_results.sort_values('AUC', ascending=False).reset_index(drop=True)

print("\n" + "="*110)
print("  MODEL BENCHMARK COMPARISON (sorted by AUC)")
print("="*110)
print(df_results.to_string(index=False, float_format='%.4f'))
print("="*110)

best = df_results.iloc[0]
print(f"\n🏆 BEST MODEL: {best['Model']}  |  AUC={best['AUC']:.4f}  |  F1={best['F1']:.4f}  |  Acc={best['Accuracy']:.4f}")
