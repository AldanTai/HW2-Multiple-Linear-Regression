import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {'MAE': float(mae), 'MSE': float(mse), 'RMSE': float(rmse), 'R2': float(r2)}

def conformal_radius(y_cal, pred_cal, alpha=0.05):
    errors = np.sort(np.abs(np.asarray(y_cal) - np.asarray(pred_cal)))
    n = len(errors)
    rank = int(np.ceil((n + 1) * (1 - alpha)))
    if rank > n:
        raise ValueError('校準集樣本太少，無法產生有限的指定涵蓋率區間')
    return float(errors[rank - 1])

def prediction_interval(pred, radius):
    pred = np.asarray(pred, dtype=float)
    return pred - radius, pred + radius

def interval_statistics(y, lower, upper):
    y = np.asarray(y)
    return {
        'Coverage': float(np.mean((y >= lower) & (y <= upper))),
        'Mean Width': float(np.mean(upper - lower)),
    }
