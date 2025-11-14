# utils/exporting.py
# this module provides functions to export metrics and predictions to CSV files

import pandas as pd
import numpy as np

def export_results(metrics, predictions, actuals, 
                    metrics_filename='esn_results_metrics.csv', 
                    pred_filename='esn_predictions.csv'):
    if metrics is None:
        return None

    df_metrics = pd.DataFrame([metrics])
    df_metrics.to_csv(metrics_filename, index=False)

    if predictions is not None and actuals is not None and len(predictions) > 0:
        
        pred_flat = predictions.flatten() if predictions.ndim > 1 and predictions.shape[1]==1 else predictions
        act_flat = actuals.flatten() if actuals.ndim > 1 and actuals.shape[1]==1 else actuals

        if predictions.ndim > 1 and predictions.shape[1] > 1:
            pred_data = {'Step': np.arange(len(predictions))}
            for i in range(predictions.shape[1]):
                pred_data[f'Prediction_{i}'] = predictions[:, i]
                pred_data[f'Actual_{i}'] = actuals[:, i]
        else:
            pred_data = {
                'Step': np.arange(len(pred_flat)),
                'Prediction': pred_flat,
                'Actual': act_flat
            }

        df_preds = pd.DataFrame(pred_data)
        df_preds.to_csv(pred_filename, index=False)

    return df_metrics