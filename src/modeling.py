import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from src.data_processing import build_preprocessor
from src.feature_selection import build_feature_selector
from src.evaluation import evaluate_model, conformal_radius, prediction_interval, interval_statistics

def build_model(num_cols, cat_cols, k='all'):
    return Pipeline([
        ('preprocessor', build_preprocessor(num_cols, cat_cols)),
        ('selector', build_feature_selector(k)),
        ('regressor', LinearRegression()),
    ])

def run_experiment(X_train, X_test, y_train, y_test, num_cols, cat_cols, alpha=0.05):
    # 校準集只用來決定區間，不參與訓練、選 K 或模型比較。
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X_train, y_train, test_size=0.25, random_state=123
    )
    probe = build_preprocessor(num_cols, cat_cols)
    probe.fit(X_fit)
    n_features = len(probe.get_feature_names_out())
    # 不同 CV fold 可能出現略不同的類別，因此固定選較保守的 K 候選。
    candidate_ks = list(range(1, min(20, len(num_cols) + 2 * len(cat_cols), n_features) + 1))
    candidate_ks.append('all')
    candidate_ks = list(dict.fromkeys(candidate_ks))
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    search = GridSearchCV(
        estimator=build_model(num_cols, cat_cols),
        param_grid={'selector__k': candidate_ks},
        scoring='neg_root_mean_squared_error',
        cv=cv,
        n_jobs=1,
        refit=True,
        return_train_score=True,
        error_score='raise',
    )
    search.fit(X_fit, y_fit)
    selected = search.best_estimator_
    baseline = build_model(num_cols, cat_cols, k='all')
    baseline.fit(X_fit, y_fit)

    test_pred = selected.predict(X_test)
    base_pred = baseline.predict(X_test)
    cal_pred = selected.predict(X_cal)
    radius = conformal_radius(y_cal, cal_pred, alpha=alpha)
    lower, upper = prediction_interval(test_pred, radius)

    cv_df = pd.DataFrame({
        'K': [v if v != 'all' else 'all' for v in search.cv_results_['param_selector__k']],
        'CV RMSE': -search.cv_results_['mean_test_score'],
        'CV Std': search.cv_results_['std_test_score'],
        'Train RMSE': -search.cv_results_['mean_train_score'],
        'Rank': search.cv_results_['rank_test_score'],
    }).sort_values(['Rank', 'CV RMSE']).reset_index(drop=True)

    metrics = pd.DataFrame([
        {'Model': 'Baseline (All Features)', 'Dataset': 'Train', **evaluate_model(y_fit, baseline.predict(X_fit))},
        {'Model': 'Baseline (All Features)', 'Dataset': 'Test', **evaluate_model(y_test, base_pred)},
        {'Model': 'Selected (CV Best K)', 'Dataset': 'Train', **evaluate_model(y_fit, selected.predict(X_fit))},
        {'Model': 'Selected (CV Best K)', 'Dataset': 'Test', **evaluate_model(y_test, test_pred)},
    ])
    test_detail = pd.DataFrame({
        'Actual': np.asarray(y_test),
        'Predicted': test_pred,
        'Lower 95%': lower,
        'Upper 95%': upper,
        'Residual': np.asarray(y_test) - test_pred,
        'Covered': (np.asarray(y_test) >= lower) & (np.asarray(y_test) <= upper),
    }, index=y_test.index)
    return {
        'baseline': baseline, 'selected': selected, 'search': search,
        'cv_table': cv_df, 'metrics': metrics, 'test_detail': test_detail,
        'radius': radius, 'interval_stats': interval_statistics(y_test, lower, upper),
        'best_k': search.best_params_['selector__k'],
        'fit_size': len(X_fit), 'cal_size': len(X_cal), 'test_size': len(X_test),
        'n_encoded_features': n_features,
    }
