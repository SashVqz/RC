import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd

class EchoStateNetwork:
    def __init__(self, inputSize, reservoirSize, outputSize,
                spectralRadius=0.9, inputScaling=1.0, leakingRate=1.0,
                sparsity=0.1, ridgeParam=1e-8, activation=None,
                feedback=False, feedbackScaling=1.0, stateNoise=0.0,
                randomSeed=None):

        if randomSeed is not None:
            np.random.seed(randomSeed)

        self.inputSize = inputSize
        self.reservoirSize = reservoirSize
        self.outputSize = outputSize
        self.leakingRate = leakingRate
        self.ridgeParam = ridgeParam
        self.activation = activation if activation is not None else np.tanh
        self.feedback = feedback
        self.stateNoise = stateNoise

        self.Win = np.random.uniform(-inputScaling, inputScaling, (reservoirSize, inputSize + 1))

        density = max(0.0, min(1.0, 1.0 - sparsity))
        if density == 0:
            self.W = sparse.csr_matrix((reservoirSize, reservoirSize))
            currentRadius = 0
        else:
            try:
                W_sparse = sparse.random(reservoirSize, reservoirSize, density=density, data_rvs=np.random.randn, format='csr')
                eigenvalues, _ = eigs(W_sparse, k=1, which='LM', maxiter=10000, tol=1e-8)
                currentRadius = np.abs(eigenvalues[0])
            except sparse.linalg.ArpackNoConvergence:
                W_dense = W_sparse.toarray()
                eigenvalues = np.linalg.eigvals(W_dense)
                currentRadius = np.max(np.abs(eigenvalues))
                W_sparse = sparse.csr_matrix(W_dense)

            if currentRadius > 1e-12:
                self.W = W_sparse * (spectralRadius / currentRadius)
            else:
                self.W = W_sparse

        if self.feedback:
            self.Wfeed = np.random.uniform(-feedbackScaling, feedbackScaling, (reservoirSize, outputSize))
            self.y_prev = np.zeros((outputSize, 1))
        else:
            self.Wfeed = None
            self.y_prev = None

        self.Wout = None
        self.x = np.zeros((reservoirSize, 1))

    def _validate_input_shape(self, u):
        u = np.atleast_2d(u)
        
        if u.shape[1] != self.inputSize and u.shape[0] == self.inputSize:
            u = u.T
        
        return u

    def _updateState(self, u_single_step):
        u_single_step = np.atleast_1d(u_single_step).flatten()
        if len(u_single_step) != self.inputSize:
            raise ValueError(f"Input dimension mismatch single step")

        inputVector = np.vstack([1, u_single_step.reshape(-1, 1)])
        feedbackTerm = np.dot(self.Wfeed, self.y_prev) if self.feedback and self.y_prev is not None else 0

        if not isinstance(self.x, np.ndarray) or self.x.ndim != 2 or self.x.shape[1] != 1:
            self.x = self.x.reshape(-1, 1)

        state_potential = np.dot(self.Win, inputVector) + self.W.dot(self.x) + feedbackTerm
        xHat = self.activation(state_potential)

        if self.stateNoise > 0:
            xHat += np.random.normal(0, self.stateNoise, xHat.shape)

        self.x = (1 - self.leakingRate) * self.x + self.leakingRate * xHat
        return self.x

    def _harvestStates(self, inputSignal, washout=0):
        inputSignal = self._validate_input_shape(inputSignal)
        numSteps = inputSignal.shape[0]
        if washout >= numSteps:
            raise ValueError(f"Washout ({washout}) >= numSteps ({numSteps}).")

        states = np.zeros((self.reservoirSize, numSteps - washout))
        self.x = np.zeros((self.reservoirSize, 1))
        if self.feedback:
            self.y_prev = np.zeros((self.outputSize, 1))

        for t in range(numSteps):
            current_x = self._updateState(inputSignal[t])
            if t >= washout:
                states[:, t - washout] = current_x.flatten()
        return states

    def train(self, inputSignal, targetSignal, washout=100):
        inputSignal = self._validate_input_shape(inputSignal)
        targetSignal = np.atleast_2d(targetSignal)
        if targetSignal.shape[1] != self.outputSize and targetSignal.shape[0] == self.outputSize:
            targetSignal = targetSignal.T

        if inputSignal.shape[0] != targetSignal.shape[0]:
            raise ValueError("Input and target signals must have the same length.")
        
        numSteps = inputSignal.shape[0]
        effectiveSteps = numSteps - washout
        if effectiveSteps <= 0:
            raise ValueError(f"Washout ({washout}) leaves no data for training.")

        X = self._harvestStates(inputSignal, washout)

        Ytarget = targetSignal[washout:].T
        X_T = X.T
        iden = np.eye(self.reservoirSize)

        try:
            if self.ridgeParam == 0: # eq. Pseudo-inverse like Tesis Lennert 
                pinv_X = np.linalg.pinv(X)
                self.Wout = np.dot(Ytarget, pinv_X)
            else:
                # eq. Tikhonov (Ridge Regression)
                term1 = np.dot(X, X_T)
                term2 = term1 + self.ridgeParam * iden
                inv_term2 = np.linalg.inv(term2)
                term3 = np.dot(Ytarget, X_T)
                self.Wout = np.dot(term3, inv_term2)
        except np.linalg.LinAlgError:
            raise np.linalg.LinAlgError("Matrix inversion failed during training.")

        return self

    def predict(self, inputSignal, washout=0, continuePrevious=False):
        inputSignal = self._validate_input_shape(inputSignal)
        numSteps = inputSignal.shape[0]
        effective_pred_steps = numSteps - washout
        predictions = np.zeros((effective_pred_steps, self.outputSize))
        internal_states = np.zeros((self.reservoirSize, effective_pred_steps))

        if self.Wout is None:
            raise RuntimeError("Model must be trained before prediction.")

        if not continuePrevious:
            self.x = np.zeros((self.reservoirSize, 1))
            if self.feedback:
                self.y_prev = np.zeros((self.outputSize, 1))

        pred_idx = 0
        state_idx = 0
        for t in range(numSteps):
            u = inputSignal[t]
            current_x = self._updateState(u)
            y = np.dot(self.Wout, current_x).flatten()

            if t >= washout:
                predictions[pred_idx] = y
                internal_states[:, state_idx] = current_x.flatten()
                pred_idx += 1
                state_idx += 1

            if self.feedback:
                self.y_prev = y.reshape(-1, 1)

        return predictions, internal_states

    def generativePredict(self, initialInputs, generationSteps, output_to_input_mapper=None):
        if self.Wout is None:
            raise RuntimeError("Model must be trained before generative prediction.")

        initialInputs = self._validate_input_shape(initialInputs)
        numInitial = initialInputs.shape[0]
        predictions = np.zeros((numInitial + generationSteps, self.outputSize))

        for t in range(numInitial):
            current_x = self._updateState(initialInputs[t])
            y = np.dot(self.Wout, current_x).flatten()
            predictions[t] = y
            self.y_prev = y.reshape(-1, 1)

        last_y = predictions[numInitial-1]
        for t in range(generationSteps):
            
            if self.inputSize == self.outputSize:
                u = last_y
            else:
                u = output_to_input_mapper(last_y)
                if not isinstance(u, np.ndarray):
                    u = np.array(u)
                    
                if u.ndim > 1:
                    u = u.flatten()
                    
            current_x = self._updateState(u)
            y = np.dot(self.Wout, current_x).flatten()
            predictions[numInitial + t] = y
            last_y = y
            self.y_prev = y.reshape(-1, 1)

        return predictions

class ESNPipeline:
    def __init__(self, esn_params, scaler=None):
        self.esn_params = esn_params
        self.esn = EchoStateNetwork(**esn_params)
        self.scaler = scaler if scaler is not None else StandardScaler()
        self.trainData_scaled = None
        self.testData_scaled = None
        self._is_trained = False

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

            mse = mean_squared_error(testTarget_orig, predictions_orig)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(testTarget_orig, predictions_orig)
            target_range = np.ptp(testTarget_orig)
            nrmse = rmse / target_range if target_range > 1e-9 else np.inf
            metrics = { 'mse': mse, 'rmse': rmse, 'mae': mae, 'nrmse': nrmse }

        return predictions_orig, testTarget_orig, metrics, internal_states_scaled

    def exportResults(self, metrics, filename='esn_results_metrics.csv', predictions=None, actuals=None, pred_filename='esn_predictions.csv'):
        if metrics is None:
            return None

        df_metrics = pd.DataFrame([metrics])
        df_metrics.to_csv(filename, index=False)

        if predictions is not None and actuals is not None and len(predictions) > 0:
            pred_data = {
                'Step': np.arange(len(predictions)),
                'Prediction': predictions.flatten(),
                'Actual': actuals.flatten()
            }
            df_preds = pd.DataFrame(pred_data)
            df_preds.to_csv(pred_filename, index=False)

        return df_metrics

    def run(self, timeseries, trainRatio=0.7, predictionHorizon=1, washout_train=100, washout_pred=0):
        trainInput, trainTarget, testInput, testTarget = self.prepareData(timeseries, trainRatio=trainRatio, predictionHorizon=predictionHorizon)
        self.trainModel(trainInput, trainTarget, washout=washout_train)
        predictions_orig, testTarget_orig, metrics, internal_states_scaled = self.evaluateModel(testInput, testTarget, washout_pred=washout_pred)

        return predictions_orig, testTarget_orig, metrics, internal_states_scaled