# utils/metrics.py
# this module provides metrics for evaluating regression models

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def calculate_metrics(actuals, predictions, original_test_target=None):
    if actuals is None or predictions is None or len(actuals) == 0:
        return {'mse': np.nan, 'rmse': np.nan, 'mae': np.nan, 'nrmse': np.nan}

    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, predictions)
    target_data_for_range = original_test_target if original_test_target is not None else actuals
    
    try:
        target_range = np.ptp(target_data_for_range, axis=0)
        mean_target_range = np.mean(target_range)
        if mean_target_range == 0 or not np.isfinite(mean_target_range):
            mean_target_range = np.ptp(actuals, axis=0).mean()
            
    except Exception:
        mean_target_range = np.ptp(actuals, axis=0).mean()

    nrmse = rmse / mean_target_range if mean_target_range > 1e-9 else np.inf

    metrics = { 
        'mse': mse, 
        'rmse': rmse, 
        'mae': mae, 
        'nrmse': nrmse 
    }

    return metrics