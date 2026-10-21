import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression

def build_feature_selector(k=10):
    return SelectKBest(score_func=f_regression, k=k)

def get_feature_ranking(fitted_pipeline):
    pre = fitted_pipeline.named_steps['preprocessor']
    selector = fitted_pipeline.named_steps['selector']
    names = pre.get_feature_names_out()
    ranking = pd.DataFrame({
        'Feature': names,
        'F_Score': selector.scores_,
        'P_Value': selector.pvalues_,
        'Selected': selector.get_support()
    })
    ranking['F_Score'] = ranking['F_Score'].replace([np.inf, -np.inf], np.nan)
    ranking = ranking.sort_values('F_Score', ascending=False, na_position='last').reset_index(drop=True)
    ranking.insert(0, 'Rank', np.arange(1, len(ranking) + 1))
    return ranking

def get_selected_features(fitted_pipeline):
    ranking = get_feature_ranking(fitted_pipeline)
    return ranking.loc[ranking['Selected'], 'Feature'].tolist()
