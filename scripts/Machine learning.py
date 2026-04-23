#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Batch 5-fold cross-validation + Optuna with dual hard constraints
- R² of each fold ≥ 0.6
- |R²_fold − mean| ≤ 0.15
All results are aggregated into AllResults.xlsx (same directory as script)
"""
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import optuna

# -------------------- Configuration --------------------
KFOLD         = 5
TEST_SIZE     = 0.2
RANDOM_STATE  = 42
N_TRIALS      = 4000        # Optuna search trials
MAX_DEV       = 0.15       # Maximum deviation from mean R²
MIN_R2_FOLD   = 0.6        # Minimum R² per fold
OUTPUT_FILE   = 'AllResults.xlsx'   # Aggregated results file

# -------------------- General numeric cleaning --------------------
def clean_numeric(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = (
                df[col].astype(str)
                .str.strip()
                .str.replace(r'^\[|\]$', '', regex=True)
            )
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# -------------------- Optuna objective (dual hard constraints) --------------------
def objective(trial, X, y):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 400, step=50),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
    }

    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=RANDOM_STATE,
        **params
    )

    kf = KFold(n_splits=KFOLD, shuffle=True, random_state=RANDOM_STATE)
    r2_list = []

    for tr_idx, val_idx in kf.split(X):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        model.fit(X_tr, y_tr, verbose=False)
        fold_r2 = r2_score(y_val, model.predict(X_val))
        r2_list.append(fold_r2)

    mean_r2 = np.mean(r2_list)

    # Dual hard constraints
    if any(r < MIN_R2_FOLD for r in r2_list):
        return -1.0
    if any(abs(r - mean_r2) > MAX_DEV for r in r2_list):
        return -1.0

    return mean_r2


def optuna_search(X, y):
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )
    study.optimize(lambda trial: objective(trial, X, y),
                   n_trials=N_TRIALS,
                   show_progress_bar=True)
    return study.best_params


# -------------------- Main pipeline for one target --------------------
def run_one_target(target):
    out_dir = f"{target}_CV_Results"
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_excel("Database.xlsx")
    df = clean_numeric(df)

    X = df[['Fe', 'Co', 'Concavity', 'Porosity']]
    y = df[target]

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # 1. Linear regression baseline
    lr = LinearRegression().fit(X_train_full, y_train_full)
    lr_r2   = r2_score(y_test, lr.predict(X_test))
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr.predict(X_test)))

    # 2. XGBoost with constrained hyperparameter optimization
    best_params = optuna_search(X_train_full, y_train_full)
    xgb_final = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=RANDOM_STATE,
        **best_params
    )
    xgb_final.fit(X_train_full, y_train_full)

    # 3. 5-fold CV evaluation (store per-fold parameters and metrics)
    kf = KFold(n_splits=KFOLD, shuffle=True, random_state=RANDOM_STATE)
    cv_r2, cv_rmse = [], []
    fold_detail = []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_full), 1):
        X_tr, X_val = X_train_full.iloc[tr_idx], X_train_full.iloc[val_idx]
        y_tr, y_val = y_train_full.iloc[tr_idx], y_train_full.iloc[val_idx]

        xgb_final.fit(X_tr, y_tr, verbose=False)
        pred_val = xgb_final.predict(X_val)

        fold_r2   = r2_score(y_val, pred_val)
        fold_rmse = np.sqrt(mean_squared_error(y_val, pred_val))

        cv_r2.append(fold_r2)
        cv_rmse.append(fold_rmse)

        fold_detail.append({
            'Target': target,
            'Fold': fold,
            **best_params,
            'R2': fold_r2,
            'RMSE': fold_rmse
        })

    # Export per-fold details
    pd.DataFrame(fold_detail).to_excel(
        os.path.join(out_dir, 'XGBoost_Foldwise_Params_Metrics.xlsx'),
        index=False
    )

    # 4. Summary statistics
    summary = pd.DataFrame({
        'Target': [target],
        'Model': ['XGBoost'],
        'CV_R2_mean': np.mean(cv_r2),
        'CV_R2_std':  np.std(cv_r2, ddof=1),
        'CV_R2_min':  np.min(cv_r2),
        'CV_R2_max':  np.max(cv_r2),
        'CV_R2_range': np.ptp(cv_r2),
        'CV_RMSE_mean': np.mean(cv_rmse),
        'CV_RMSE_std':  np.std(cv_rmse, ddof=1),
        'CV_RMSE_min':  np.min(cv_rmse),
        'CV_RMSE_max':  np.max(cv_rmse),
        'CV_RMSE_range': np.ptp(cv_rmse),
        'Test_R2': r2_score(y_test, xgb_final.predict(X_test)),
        'Test_RMSE': np.sqrt(mean_squared_error(y_test, xgb_final.predict(X_test)))
    })

    return summary


# -------------------- Batch execution and aggregation --------------------
def main():
    target_list = [
        'S1_peak', 'S1_min', 'S1_inte',
        'S2_peak', 'S2_min', 'S2_inte',
        'C1_peak', 'C1_min', 'C1_inte',
        'C2_peak', 'C2_min', 'C2_inte',
        'C3_peak', 'C3_min', 'C3_inte',
        'C4_peak', 'C4_min', 'C4_inte'
    ]

    all_res = []
    for tgt in target_list:
        try:
            print('\n>>> Processing', tgt)
            res = run_one_target(tgt)
            all_res.append(res)
        except Exception as e:
            print(f'>>> {tgt} failed: {e}')
            continue

    final_df = pd.concat(all_res, ignore_index=True)
    final_df.to_excel(OUTPUT_FILE, index=False)
    print('\n>>> All results have been saved to:', OUTPUT_FILE)


if __name__ == '__main__':
    main()

