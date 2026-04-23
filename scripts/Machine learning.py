
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Full version: multi-model k-fold cross-validation + linear regression
+ error-bar data + reviewer-ready table + plotting
Special data generator for reviewer comment #3

Before running, make sure that "Database.xlsx" is in the same directory.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# sklearn
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression

# tree-based model
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# other models
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

import shap


# ========== Universal cleaning: convert "[x.xE-xx]" strings to float ==========
def clean_numeric(df):
    """Clean in place: remove square brackets and convert to numeric; invalid values become NaN."""
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = (
                df[col].astype(str)
                .str.strip()
                .str.replace(r'^\[|\]$', '', regex=True)
            )
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


# -------------------- 1. Basic configuration --------------------
KFOLD = 5
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ITER = 3   # reduced from 10 to improve stability


def main(TARGET):
    OUT_DIR = f"{TARGET}_CV_Results"
    os.makedirs(OUT_DIR, exist_ok=True)

    # -------------------- 2. Load data --------------------
    df = pd.read_excel("Database.xlsx")
    df = clean_numeric(df)

    # Keep the original data logic
    X = df[['Fe', 'Co', 'Concavity', 'Porosity']]
    y = df[TARGET]

    # Hold out the independent test set first
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # -------------------- 3. Define models and hyperparameter spaces --------------------
    MODELS = {
        'LinearRegression': {
            'model': LinearRegression(),
            'params': {}
        },
        'XGBoost': {
            'model': xgb.XGBRegressor(
                objective='reg:squarederror',
                random_state=RANDOM_STATE,
                n_jobs=1
            ),
            'params': {
                'n_estimators': [50, 100],
                'learning_rate': [0.05, 0.1],
                'max_depth': [3, 4],
                'subsample': [0.7, 0.8],
                'colsample_bytree': [0.7, 0.8],
                'reg_alpha': [0.0, 0.1],
                'reg_lambda': [1.0, 2.0]
            }
        },
        'RandomForest': {
            'model': RandomForestRegressor(
                random_state=RANDOM_STATE,
                n_jobs=1
            ),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        },
        'MLP': {
            'model': MLPRegressor(
                max_iter=1000,
                random_state=RANDOM_STATE
            ),
            'params': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.01]
            }
        },
        'KNN': {
            'model': KNeighborsRegressor(),
            'params': {
                'n_neighbors': [3, 5, 7],
                'weights': ['uniform', 'distance']
            }
        },
        'SVR': {
            'model': SVR(),
            'params': {
                'kernel': ['rbf', 'linear'],
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto']
            }
        },
        'GPR': {
            'model': GaussianProcessRegressor(
                kernel=ConstantKernel(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2)),
                n_restarts_optimizer=10,
                random_state=RANDOM_STATE
            ),
            'params': {}
        }
    }

    # -------------------- 4. Training / validation / testing --------------------
    kfold = KFold(n_splits=KFOLD, shuffle=True, random_state=RANDOM_STATE)

    summary_rows = []
    detail_rows = []

    for name, cfg in MODELS.items():
        print(f"\n>>> {name}")
        val_r2, val_rmse = [], []
        tr_r2, tr_rmse = [], []

        for fold, (tr_idx, val_idx) in enumerate(kfold.split(X_train_full), 1):
            print(f"    Fold {fold}/{KFOLD}")

            X_tr = X_train_full.iloc[tr_idx]
            y_tr = y_train_full.iloc[tr_idx]
            X_val = X_train_full.iloc[val_idx]
            y_val = y_train_full.iloc[val_idx]

            # Hyperparameter search
            if cfg['params']:
                print(f"    Starting hyperparameter search for {name} (fold {fold})...")
                search = RandomizedSearchCV(
                    cfg['model'],
                    cfg['params'],
                    n_iter=N_ITER,
                    cv=3,
                    scoring='r2',
                    n_jobs=1,
                    random_state=RANDOM_STATE
                )
                search.fit(X_tr, y_tr)
                best = search.best_estimator_
                print(f"    Finished hyperparameter search for {name} (fold {fold})")
            else:
                best = cfg['model'].fit(X_tr, y_tr)

            # Validation performance
            pred_val = best.predict(X_val)
            val_r2.append(r2_score(y_val, pred_val))
            val_rmse.append(np.sqrt(mean_squared_error(y_val, pred_val)))

            # Training performance
            pred_tr = best.predict(X_tr)
            tr_r2.append(r2_score(y_tr, pred_tr))
            tr_rmse.append(np.sqrt(mean_squared_error(y_tr, pred_tr)))

            detail_rows.append({
                'Model': name,
                'Fold': fold,
                'Train_R2': tr_r2[-1],
                'Train_RMSE': tr_rmse[-1],
                'Validation_R2': val_r2[-1],
                'Validation_RMSE': val_rmse[-1]
            })

        # Final model trained on full training set -> independent test set
        if cfg['params']:
            print(f"    Starting final hyperparameter search for {name}...")
            final_search = RandomizedSearchCV(
                cfg['model'],
                cfg['params'],
                n_iter=N_ITER,
                cv=3,
                scoring='r2',
                n_jobs=1,
                random_state=RANDOM_STATE
            )
            final_search.fit(X_train_full, y_train_full)
            final_model = final_search.best_estimator_
            print(f"    Finished final hyperparameter search for {name}")
        else:
            final_model = cfg['model'].fit(X_train_full, y_train_full)

        test_pred = final_model.predict(X_test)
        test_r2 = r2_score(y_test, test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))

        # Save test set predictions
        pd.DataFrame({
            'Observed': y_test,
            'Predicted': test_pred
        }).to_excel(
            os.path.join(OUT_DIR, f'{name}_TestSet_Prediction.xlsx'),
            index=False
        )

        summary_rows.append({
            'Model': name,
            'Validation_R2_mean': np.mean(val_r2),
            'Validation_R2_std': np.std(val_r2, ddof=1),
            'Validation_RMSE_mean': np.mean(val_rmse),
            'Validation_RMSE_std': np.std(val_rmse, ddof=1),
            'Test_R2': test_r2,
            'Test_RMSE': test_rmse
        })

    # -------------------- 5. Export Excel files --------------------
    summary_df = pd.DataFrame(summary_rows)
    detail_df = pd.DataFrame(detail_rows)

    summary_df.to_excel(os.path.join(OUT_DIR, 'Model_Performance_Summary.xlsx'), index=False)
    detail_df.to_excel(os.path.join(OUT_DIR, 'Foldwise_Detailed_Results.xlsx'), index=False)

    # Reviewer-ready table
    reviewer_df = summary_df[['Model', 'Validation_R2_mean', 'Validation_R2_std', 'Test_R2']].copy()
    reviewer_df['Validation_R2_Mean±SD'] = reviewer_df.apply(
        lambda x: f"{x['Validation_R2_mean']:.3f}±{x['Validation_R2_std']:.3f}", axis=1
    )
    reviewer_df = reviewer_df[['Model', 'Validation_R2_Mean±SD', 'Test_R2']]
    reviewer_df.to_excel(
        os.path.join(OUT_DIR, 'Reviewer_Required_Performance_Table.xlsx'),
        index=False
    )

    # -------------------- 6. Export error-bar data --------------------
    error_bar_df = pd.DataFrame({
        'Model': summary_df['Model'],
        'Metric': ['R²'] * len(summary_df),
        'Mean': summary_df['Validation_R2_mean'],
        'SD': summary_df['Validation_R2_std'],
        'Mean±SD': summary_df.apply(
            lambda x: f"{x['Validation_R2_mean']:.3f}±{x['Validation_R2_std']:.3f}", axis=1
        )
    })

    rmse_bar_df = pd.DataFrame({
        'Model': summary_df['Model'],
        'Metric': ['RMSE'] * len(summary_df),
        'Mean': summary_df['Validation_RMSE_mean'],
        'SD': summary_df['Validation_RMSE_std'],
        'Mean±SD': summary_df.apply(
            lambda x: f"{x['Validation_RMSE_mean']:.3f}±{x['Validation_RMSE_std']:.3f}", axis=1
        )
    })

    error_bar_all = pd.concat([error_bar_df, rmse_bar_df], ignore_index=True)

    error_bar_file = os.path.join(OUT_DIR, 'CrossValidation_ErrorBar_Data.xlsx')
    error_bar_all.to_excel(error_bar_file, index=False)
    error_bar_all.to_csv(os.path.join(OUT_DIR, 'CrossValidation_ErrorBar_Data.csv'), index=False)

    # -------------------- 7. Plotting instructions --------------------
    how_to_plot = (
        "Error-bar plotting instructions:\n"
        "1) Software: Origin / GraphPad Prism / Python Matplotlib\n"
        "2) Data: use the 'Mean±SD' column in 'CrossValidation_ErrorBar_Data.xlsx'\n"
        "3) Plot type: bar chart with error bars = mean ± standard deviation (±SD)\n"
        "4) Example (Python):\n"
        "   import pandas as pd, matplotlib.pyplot as plt\n"
        "   df = pd.read_excel('CrossValidation_ErrorBar_Data.xlsx')\n"
        "   r2_df = df[df['Metric']=='R²']\n"
        "   plt.bar(r2_df['Model'], r2_df['Mean'], yerr=r2_df['SD'], capsize=5)\n"
        "   plt.ylabel('R²'); plt.show()"
    )

    with open(os.path.join(OUT_DIR, 'How_to_Plot_Error_Bars.txt'), 'w', encoding='utf-8') as f:
        f.write(how_to_plot)

    print('\n>>> Error-bar data exported to:', error_bar_file)
    print('>>> Plotting instructions saved to:', os.path.join(OUT_DIR, 'How_to_Plot_Error_Bars.txt'))

    # ===================== 8. SHAP analysis (tree models only) =====================
    print('\n>>> Computing SHAP values (tree models only: XGBoost & RandomForest)...')

    X_train_clean = clean_numeric(X_train_full.copy())
    X_test_clean = clean_numeric(X_test.copy())

    X_train_np = X_train_clean.values.astype(np.float32)
    X_test_np = X_test_clean.values.astype(np.float32)
    y_train_np = y_train_full.values.astype(np.float32)

    feature_names = list(X_train_full.columns)

    TREE_MODELS = {
        'XGBoost': MODELS['XGBoost'],
        'RandomForest': MODELS['RandomForest']
    }

    for name, cfg in TREE_MODELS.items():
        print(f'\n--- {name} SHAP analysis ---')

        try:
            if name == 'XGBoost':
                if cfg['params']:
                    print(f"    Starting SHAP hyperparameter search for {name}...")
                    search = RandomizedSearchCV(
                        cfg['model'],
                        cfg['params'],
                        n_iter=N_ITER,
                        cv=3,
                        scoring='r2',
                        n_jobs=1,
                        random_state=RANDOM_STATE
                    )
                    search.fit(X_train_np, y_train_np)
                    best_params = search.best_params_
                    print(f"    Finished SHAP hyperparameter search for {name}")
                else:
                    best_params = {}

                base_score_val = float(np.nanmean(y_train_np))

                final_model = xgb.XGBRegressor(
                    objective='reg:squarederror',
                    random_state=RANDOM_STATE,
                    base_score=base_score_val,
                    n_jobs=1,
                    **best_params
                )
                final_model.fit(X_train_np, y_train_np)

            elif name == 'RandomForest':
                if cfg['params']:
                    print(f"    Starting SHAP hyperparameter search for {name}...")
                    search = RandomizedSearchCV(
                        cfg['model'],
                        cfg['params'],
                        n_iter=N_ITER,
                        cv=3,
                        scoring='r2',
                        n_jobs=1,
                        random_state=RANDOM_STATE
                    )
                    search.fit(X_train_np, y_train_np)
                    final_model = search.best_estimator_
                    print(f"    Finished SHAP hyperparameter search for {name}")
                else:
                    final_model = cfg['model'].fit(X_train_np, y_train_np)

            explainer = shap.TreeExplainer(final_model)
            shap_values = explainer.shap_values(X_test_np)

            shap_df = pd.DataFrame(shap_values, columns=feature_names)
            shap_summary = pd.DataFrame({
                'Feature': feature_names,
                'SHAP_mean': shap_df.mean(),
                'SHAP_abs_mean': shap_df.abs().mean(),
                'SHAP_std': shap_df.std()
            }).sort_values('SHAP_abs_mean', ascending=False).reset_index(drop=True)

            shap_summary.to_excel(
                os.path.join(OUT_DIR, f'{name}_SHAP_summary.xlsx'),
                index=False
            )
            shap_df.to_excel(
                os.path.join(OUT_DIR, f'{name}_SHAP_values_per_sample.xlsx'),
                index=False
            )

            print(f'>>> {name} SHAP data exported successfully')

        except Exception as e:
            print(f'>>> {name} SHAP analysis failed: {e}')
            continue


col_names = [
    'S1_peak', 'S1_min', 
    'S2_peak', 'S2_min', 
    'C1_peak', 'C1_min', 
    'C2_peak', 'C2_min', 
    'C3_peak', 'C3_min', 
    'C4_peak', 'C4_min', 
]

for name in col_names:
    try:
        main(name)
    except Exception as e:
        print(f"{name} failed: {e}")
