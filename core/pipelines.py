# core/pipelines.py
# this module defines pipelines for training and evaluating ESN models

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings
from utils.metrics import calculate_metrics

# - standard pipeline for ESN models -
class StandardPipeline:
    def __init__(self, esn_model, scaler=None):
        self.esn = esn_model
        self.scaler = scaler if scaler is not None else StandardScaler()
        self.trainData_scaled = None
        self.testData_scaled = None
        self._is_trained = False
        self.original_testTarget_scaled = None 

    def prepareData(self, timeseries, trainRatio=0.7, predictionHorizon=1, input_cols=None, target_cols=None):
        timeseries = np.atleast_2d(timeseries)
        if timeseries.ndim == 1:
            timeseries = timeseries.reshape(-1, 1)

        if input_cols is None:
            input_cols = list(range(timeseries.shape[1]))
        
        if target_cols is None:
            target_cols = list(range(timeseries.shape[1]))
        
        input_cols = np.array(input_cols)
        target_cols = np.array(target_cols)

        splitIdx = int(len(timeseries) * trainRatio)
        train_raw = timeseries[:splitIdx]
        test_raw = timeseries[splitIdx:]
        if len(train_raw) == 0 or len(test_raw) == 0:
            raise ValueError("Train or test split resulted in zero samples.")

        self.scaler.fit(train_raw) 
        self.trainData_scaled = self.scaler.transform(train_raw)
        self.testData_scaled = self.scaler.transform(test_raw)

        trainInput = self.trainData_scaled[:-predictionHorizon, input_cols]
        trainTarget = self.trainData_scaled[predictionHorizon:, target_cols]
        testInput = self.testData_scaled[:-predictionHorizon, input_cols]
        testTarget = self.testData_scaled[predictionHorizon:, target_cols]
        
        self.original_testTarget_scaled = testTarget # Guardamos esto

        return trainInput, trainTarget, testInput, testTarget

    def trainModel(self, trainInput, trainTarget, washout=100):
        self.esn.train(trainInput, trainTarget, washout=washout)
        self._is_trained = True
        return self

    def evaluateModel(self, testInput, testTarget, washout_pred=0):
        if not self._is_trained:
            raise RuntimeError("Model must be trained before evaluation.")

        predictions_scaled, internal_states_scaled = self.esn.predict(testInput, washout=washout_pred, continuePrevious=False)
        testTarget_scaled_eval = testTarget[washout_pred:]

        min_len = min(predictions_scaled.shape[0], testTarget_scaled_eval.shape[0])
        metrics = {'mse': np.nan, 'rmse': np.nan, 'mae': np.nan, 'nrmse': np.nan}
        predictions_orig, testTarget_orig = np.array([]), np.array([])

        if min_len > 0:
            predictions_scaled = predictions_scaled[:min_len]
            testTarget_scaled_eval = testTarget_scaled_eval[:min_len]
            internal_states_scaled = internal_states_scaled[:, :min_len]

            predictions_orig = self.scaler.inverse_transform(predictions_scaled)
            testTarget_orig = self.scaler.inverse_transform(testTarget_scaled_eval)

            full_testTarget_orig = self.scaler.inverse_transform(self.original_testTarget_scaled)
            metrics = calculate_metrics(testTarget_orig, predictions_orig, full_testTarget_orig)

        return predictions_orig, testTarget_orig, metrics, internal_states_scaled

    def run(self, timeseries, trainRatio=0.7, predictionHorizon=1, washout_train=100, washout_pred=0):
        trainInput, trainTarget, testInput, testTarget = self.prepareData(timeseries, trainRatio=trainRatio, predictionHorizon=predictionHorizon)
        self.trainModel(trainInput, trainTarget, washout=washout_train)
        predictions_orig, testTarget_orig, metrics, internal_states_scaled = self.evaluateModel(testInput, testTarget, washout_pred=washout_pred)

        return predictions_orig, testTarget_orig, metrics, internal_states_scaled

# - delay Pipeline that handles inverse scaling properly and data preparation for delay-based ESNs -
class DelayPipeline:
    def __init__(self, esn_model, scaler=None):
        self.esn = esn_model
        self.scaler = scaler if scaler is not None else StandardScaler()
        self.trainData_scaled = None
        self.testData_scaled = None
        self._is_trained = False
        self._target_cols_indices = None
        self.original_testTarget_scaled = None

    def prepareData(self, timeseries, trainRatio=0.7, predictionHorizon=1, input_cols=None, target_cols=None):
        timeseries = np.atleast_2d(timeseries)
        if timeseries.ndim == 1:
            timeseries = timeseries.reshape(-1, 1)

        n_features = timeseries.shape[1]

        if input_cols is None: input_cols = list(range(n_features))
        if target_cols is None: target_cols = list(range(n_features))

        self._target_cols_indices = np.array(target_cols)

        input_cols = np.array(input_cols)
        target_cols = np.array(target_cols)

        if len(input_cols) != self.esn.inputSize:
            raise ValueError(f"Number of input_cols ({len(input_cols)}) does not match ESN inputSize ({self.esn.inputSize}).")

        if len(target_cols) != self.esn.outputSize:
            raise ValueError(f"Number of target_cols ({len(target_cols)}) does not match ESN outputSize ({self.esn.outputSize}).")

        splitIdx = int(len(timeseries) * trainRatio)
        train_raw = timeseries[:splitIdx]
        test_raw = timeseries[splitIdx:]
        if len(train_raw) == 0 or len(test_raw) == 0:
            raise ValueError("Train or test split resulted in zero samples.")

        self.scaler.fit(train_raw)
        self.trainData_scaled_full = self.scaler.transform(train_raw)
        self.testData_scaled_full = self.scaler.transform(test_raw)

        if len(self.trainData_scaled_full) <= predictionHorizon or len(self.testData_scaled_full) <= predictionHorizon:
            raise ValueError(f"Not enough data in split for predictionHorizon={predictionHorizon}.")

        trainInput = self.trainData_scaled_full[:-predictionHorizon][:, input_cols]
        trainTarget = self.trainData_scaled_full[predictionHorizon:][:, target_cols]
        testInput = self.testData_scaled_full[:-predictionHorizon][:, input_cols]
        testTarget = self.testData_scaled_full[predictionHorizon:][:, target_cols]

        self.original_testTarget_scaled = testTarget

        if trainTarget.ndim == 1: trainTarget = trainTarget.reshape(-1, 1)
        if testTarget.ndim == 1: testTarget = testTarget.reshape(-1, 1)
        
        if trainInput.ndim == 1: trainInput = trainInput.reshape(-1, 1)
        if testInput.ndim == 1: testInput = testInput.reshape(-1, 1)

        return trainInput, trainTarget, testInput, testTarget

    def _inverse_transform_output(self, scaled_output):
        if self.scaler is None or self._target_cols_indices is None:
            return scaled_output

        n_samples = scaled_output.shape[0]
        n_features_original = self.scaler.n_features_in_
        temp_full_scaled = np.zeros((n_samples, n_features_original))

        if len(self._target_cols_indices) != scaled_output.shape[1]:
            if len(self._target_cols_indices) < scaled_output.shape[1]:
                scaled_output = scaled_output[:, :len(self._target_cols_indices)]
            else: return scaled_output

        temp_full_scaled[:, self._target_cols_indices] = scaled_output
        original_scale_full = self.scaler.inverse_transform(temp_full_scaled)
        original_output = original_scale_full[:, self._target_cols_indices]

        return original_output

    def trainModel(self, trainInput, trainTarget, washout_k=100):
        self.esn.train(trainInput, trainTarget, washout_k=washout_k)
        self._is_trained = True
        return self

    def evaluateModel(self, testInput, testTarget_orig_scaled, washout_pred_k=0):
        if not self._is_trained:
            raise RuntimeError("Model must be trained before evaluation.")

        predictions_scaled, internal_states_scaled = self.esn.predict(testInput, washout_k=washout_pred_k, continuePrevious=False)
        testTarget_scaled_eval = testTarget_orig_scaled[washout_pred_k:]

        min_len = min(predictions_scaled.shape[0], testTarget_scaled_eval.shape[0])
        metrics = {'mse': np.nan, 'rmse': np.nan, 'mae': np.nan, 'nrmse': np.nan}
        predictions_orig, testTarget_orig_eval = np.array([]), np.array([])

        if min_len > 0:
            predictions_scaled = predictions_scaled[:min_len]
            testTarget_scaled_eval = testTarget_scaled_eval[:min_len]
            if internal_states_scaled.shape[1] > 0 :
                internal_states_scaled = internal_states_scaled[:, :min_len]
            else:
                internal_states_scaled = np.zeros((self.esn.numVirtualNodes, 0))

            predictions_orig = self._inverse_transform_output(predictions_scaled)
            testTarget_orig_eval = self._inverse_transform_output(testTarget_scaled_eval)

            full_testTarget_orig = self._inverse_transform_output(self.original_testTarget_scaled)
            metrics = calculate_metrics(testTarget_orig_eval, predictions_orig, full_testTarget_orig)

        return predictions_orig, testTarget_orig_eval, metrics, internal_states_scaled

    def run(self, timeseries, trainRatio=0.7, predictionHorizon=1, washout_train=100, washout_pred=0):
        trainInput, trainTarget, testInput, testTarget_orig_scaled = self.prepareData(timeseries, trainRatio=trainRatio, predictionHorizon=predictionHorizon)
        self.trainModel(trainInput, trainTarget, washout_k=washout_train)
        predictions_orig, testTarget_orig_eval, metrics, internal_states_scaled = self.evaluateModel(testInput, testTarget_orig_scaled, washout_pred_k=washout_pred)

        return predictions_orig, testTarget_orig_eval, metrics, internal_states_scaled
    
# - -