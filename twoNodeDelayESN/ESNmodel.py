# Archivo: ESNmodel_delay.py
import numpy as np
from scipy import sparse
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import warnings
from collections import deque
import math

class TwoNodeDelayESN:
    def __init__(self, inputSize, numVirtualNodes, outputSize,
                tau, theta,
                eta=0.5,
                gamma=0.1,
                maskScaling=0.1,
                integrationStep=0.1,
                ridgeParam=1e-8,
                activation=None,
                feedback=False,
                feedbackScaling=1.0,
                stateNoise=0.0,
                randomSeed=None):

        if randomSeed is not None:
            np.random.seed(randomSeed)

        if numVirtualNodes % 2 != 0:
            raise ValueError("numVirtualNodes (N) must be an even number for the two-node architecture.")

        if not np.isclose(tau, numVirtualNodes * theta):
            theta = tau / numVirtualNodes

        self.inputSize = inputSize
        self.numVirtualNodes = numVirtualNodes
        self.outputSize = outputSize
        self.tau = tau
        self.theta = theta
        self.eta = eta
        self.gamma = gamma
        self.integrationStep = integrationStep
        self.ridgeParam = ridgeParam
        self.activation = activation if activation is not None else np.tanh
        self.stateNoise = stateNoise

        self.numNodesPerUnit = numVirtualNodes // 2
        self.tau_steps = max(1, int(round(self.tau / self.integrationStep)))
        self.theta_steps = max(1, int(round(self.theta / self.integrationStep)))
        self.half_tau_steps = max(1, int(round(self.tau_steps / 2)))

        self.numVirtualNodes = self.numNodesPerUnit * 2
        self.tau_steps = self.numNodesPerUnit * 2 * self.theta_steps
        self.half_tau_steps = self.numNodesPerUnit * self.theta_steps
        self.tau = self.tau_steps * self.integrationStep
        self.theta = self.theta_steps * self.integrationStep

        self.T_step = self.half_tau_steps * self.integrationStep
        self.T_step_steps = self.half_tau_steps

        self.mask1 = np.random.choice([-maskScaling, maskScaling], self.numNodesPerUnit)
        self.mask2 = np.random.choice([-maskScaling, maskScaling], self.numNodesPerUnit)
        while np.array_equal(self.mask1, self.mask2):
            self.mask2 = np.random.choice([-maskScaling, maskScaling], self.numNodesPerUnit)

        buffer_size = self.tau_steps + 5
        self.history1 = deque(np.zeros(buffer_size), maxlen=buffer_size)
        self.history2 = deque(np.zeros(buffer_size), maxlen=buffer_size)
        self.x1_current = 0.0
        self.x2_current = 0.0
        self.simulation_time = 0.0

        self.feedback = feedback

        self.Wout = None

    def _validate_input_shape(self, u):
        u = np.atleast_2d(u)
        if u.shape[1] != self.inputSize and u.shape[0] == self.inputSize:
            u = u.T
        if u.shape[1] != self.inputSize:
            raise ValueError(f"Input signal shape mismatch: expected (n_steps, {self.inputSize}), got {u.shape}")
        return u

    def _get_masked_inputs(self, u_k, t):
        time_in_half_cycle = t % self.T_step
        virtual_node_index_unit = int(math.floor(time_in_half_cycle / self.theta))
        virtual_node_index_unit = max(0, min(virtual_node_index_unit, self.numNodesPerUnit - 1))

        mask_value1 = self.mask1[virtual_node_index_unit]
        mask_value2 = self.mask2[virtual_node_index_unit]

        if self.inputSize != 1:
            warnings.warn(f"Input size {self.inputSize} > 1. Assuming scalar input processing.")
        input_value = u_k[0]

        J1_t = self.gamma * mask_value1 * input_value
        J2_t = self.gamma * mask_value2 * input_value
        return J1_t, J2_t

    def _get_delayed_state(self, history_deque, steps_ago):
        if len(history_deque) < steps_ago + 1:
            return 0.0
        try:
            idx_for_delay = len(history_deque) - steps_ago - 1
            if idx_for_delay < 0: return 0.0
            return history_deque[idx_for_delay]
        except IndexError:
            return 0.0

    def _updateStates_CoupledDDE(self, u_k_step, current_sim_time):
        x1_delayed_tau = self._get_delayed_state(self.history1, self.tau_steps)
        x2_delayed_tau = self._get_delayed_state(self.history2, self.tau_steps)
        x1_delayed_half_tau = self._get_delayed_state(self.history1, self.half_tau_steps)
        x2_delayed_half_tau = self._get_delayed_state(self.history2, self.half_tau_steps)

        J1_t, J2_t = self._get_masked_inputs(u_k_step, current_sim_time)

        nonlinear_input1 = x1_delayed_tau + x2_delayed_half_tau + J1_t
        nonlinear_input2 = x2_delayed_tau + x1_delayed_half_tau + J2_t

        nonlinear_output1 = self.eta * self.activation(nonlinear_input1)
        nonlinear_output2 = self.eta * self.activation(nonlinear_input2)

        x1_new = self.x1_current + self.integrationStep * (-self.x1_current + nonlinear_output1)
        x2_new = self.x2_current + self.integrationStep * (-self.x2_current + nonlinear_output2)

        if self.stateNoise > 0:
            x1_new += np.random.normal(0, self.stateNoise * np.sqrt(self.integrationStep))
            x2_new += np.random.normal(0, self.stateNoise * np.sqrt(self.integrationStep))

        self.history1.append(self.x1_current)
        self.history2.append(self.x2_current)
        self.x1_current = x1_new
        self.x2_current = x2_new
        self.simulation_time += self.integrationStep

        return self.x1_current, self.x2_current

    def _harvestStates(self, inputSignal_k, washout_k=0):
        inputSignal_k = self._validate_input_shape(inputSignal_k)
        numSteps_k = inputSignal_k.shape[0]

        if washout_k >= numSteps_k:
            raise ValueError(f"Washout ({washout_k}) >= numSteps ({numSteps_k}).")

        num_simulation_steps_total = numSteps_k * self.T_step_steps
        num_effective_k_steps = numSteps_k - washout_k

        self.x1_current = 0.0
        self.x2_current = 0.0
        buffer_size = self.tau_steps + 5
        self.history1 = deque(np.zeros(buffer_size), maxlen=buffer_size)
        self.history2 = deque(np.zeros(buffer_size), maxlen=buffer_size)
        self.simulation_time = 0.0

        collected_states = np.zeros((self.numVirtualNodes, num_effective_k_steps))
        state_collect_idx = 0

        current_k_step_index = 0
        u_k = inputSignal_k[current_k_step_index]

        for sim_step in range(num_simulation_steps_total):
            if sim_step > 0 and sim_step % self.T_step_steps == 0:
                current_k_step_index += 1
                if current_k_step_index < numSteps_k:
                    u_k = inputSignal_k[current_k_step_index]

                if current_k_step_index > washout_k:
                    current_global_time = current_k_step_index * self.T_step
                    state_vector_k = np.zeros(self.numVirtualNodes)

                    for i_node_unit in range(self.numNodesPerUnit):
                        t_sample1 = current_global_time - (self.numNodesPerUnit - (i_node_unit + 1)) * self.theta
                        steps_ago1 = int(round((current_global_time - t_sample1) / self.integrationStep))
                        state_vector_k[i_node_unit] = self._get_delayed_state(self.history1, steps_ago1)

                    for i_node_unit in range(self.numNodesPerUnit):
                        global_node_index = self.numNodesPerUnit + i_node_unit
                        t_sample2 = current_global_time - (self.numVirtualNodes - (global_node_index + 1)) * self.theta
                        steps_ago2 = int(round((current_global_time - t_sample2) / self.integrationStep))
                        state_vector_k[global_node_index] = self._get_delayed_state(self.history2, steps_ago2)

                    if state_collect_idx < num_effective_k_steps:
                        collected_states[:, state_collect_idx] = state_vector_k
                        state_collect_idx += 1
                    else:
                        warnings.warn(f"State collection index {state_collect_idx} out of bounds ({num_effective_k_steps}).")

            u_k_vector = u_k if isinstance(u_k, np.ndarray) else np.array([u_k])
            self._updateStates_CoupledDDE(u_k_vector, self.simulation_time)

        if collected_states.shape[1] != num_effective_k_steps:
            warnings.warn(f"Collected states shape mismatch after loop: expected {num_effective_k_steps}, got {collected_states.shape[1]}. Truncating/Padding.")
            if collected_states.shape[1] > num_effective_k_steps:
                collected_states = collected_states[:, :num_effective_k_steps]
            else:
                padding = np.zeros((self.numVirtualNodes, num_effective_k_steps - collected_states.shape[1]))
                collected_states = np.hstack((collected_states, padding))

        return collected_states

    def train(self, inputSignal_k, targetSignal_k, washout_k=100):
        inputSignal_k = self._validate_input_shape(inputSignal_k)
        targetSignal_k = np.atleast_2d(targetSignal_k)
        if targetSignal_k.shape[1] != self.outputSize and targetSignal_k.shape[0] == self.outputSize:
            targetSignal_k = targetSignal_k.T

        if inputSignal_k.shape[0] != targetSignal_k.shape[0]:
            raise ValueError("Input and target signals must have the same length (numSteps_k).")

        numSteps_k = inputSignal_k.shape[0]
        effectiveSteps_k = numSteps_k - washout_k
        if effectiveSteps_k <= 0:
            raise ValueError(f"Washout ({washout_k}) leaves no data for training.")

        X = self._harvestStates(inputSignal_k, washout_k=washout_k)

        Ytarget = targetSignal_k[washout_k:].T
        if Ytarget.shape[0] != self.outputSize or Ytarget.shape[1] != effectiveSteps_k:
            raise ValueError(f"Target signal shape mismatch after washout: expected ({self.outputSize}, {effectiveSteps_k}), got {Ytarget.shape}")

        X_T = X.T
        iden = np.eye(self.numVirtualNodes)

        try:
            term1 = X @ X_T
            term2 = term1 + self.ridgeParam * iden
            inv_term2 = np.linalg.inv(term2)
            term3 = Ytarget @ X_T
            self.Wout = term3 @ inv_term2

        except np.linalg.LinAlgError:
            try:
                self.Wout = Ytarget @ np.linalg.pinv(X)
            except np.linalg.LinAlgError:
                raise np.linalg.LinAlgError("Matrix inversion and pseudo-inverse failed during training.")

        return self

    def predict(self, inputSignal_k, washout_k=0, continuePrevious=False):
        if self.Wout is None:
            raise RuntimeError("Model must be trained before prediction.")

        inputSignal_k = self._validate_input_shape(inputSignal_k)
        numSteps_k = inputSignal_k.shape[0]
        effective_pred_steps_k = numSteps_k - washout_k

        if effective_pred_steps_k <= 0:
            warnings.warn(f"Washout ({washout_k}) is >= number of steps ({numSteps_k}). No predictions will be generated.")
            return np.zeros((0, self.outputSize)), np.zeros((self.numVirtualNodes, 0))

        predictions = np.zeros((effective_pred_steps_k, self.outputSize))
        internal_states = np.zeros((self.numVirtualNodes, effective_pred_steps_k))

        if not continuePrevious:
            self.x1_current = 0.0
            self.x2_current = 0.0
            buffer_size = self.tau_steps + 5
            self.history1 = deque(np.zeros(buffer_size), maxlen=buffer_size)
            self.history2 = deque(np.zeros(buffer_size), maxlen=buffer_size)
            self.simulation_time = 0.0

        num_simulation_steps_total = numSteps_k * self.T_step_steps
        pred_collect_idx = 0
        state_collect_idx = 0
        current_k_step_index = 0
        u_k = inputSignal_k[current_k_step_index]

        for sim_step in range(num_simulation_steps_total):
            if sim_step > 0 and sim_step % self.T_step_steps == 0:
                current_k_step_index += 1

                current_global_time = current_k_step_index * self.T_step
                state_vector_k_minus_1 = np.zeros(self.numVirtualNodes)
                for i_node_unit in range(self.numNodesPerUnit):
                    t_sample1 = current_global_time - (self.numNodesPerUnit - (i_node_unit + 1)) * self.theta
                    steps_ago1 = int(round((current_global_time - t_sample1) / self.integrationStep))
                    state_vector_k_minus_1[i_node_unit] = self._get_delayed_state(self.history1, steps_ago1)
                for i_node_unit in range(self.numNodesPerUnit):
                    global_node_index = self.numNodesPerUnit + i_node_unit
                    t_sample2 = current_global_time - (self.numVirtualNodes - (global_node_index + 1)) * self.theta
                    steps_ago2 = int(round((current_global_time - t_sample2) / self.integrationStep))
                    state_vector_k_minus_1[global_node_index] = self._get_delayed_state(self.history2, steps_ago2)

                y_k_minus_1 = (self.Wout @ state_vector_k_minus_1).flatten()

                if current_k_step_index > washout_k:
                    if pred_collect_idx < effective_pred_steps_k:
                        predictions[pred_collect_idx] = y_k_minus_1
                        internal_states[:, state_collect_idx] = state_vector_k_minus_1
                        pred_collect_idx += 1
                        state_collect_idx += 1
                    else:
                        warnings.warn("Prediction index out of bounds, check logic.")

                if current_k_step_index < numSteps_k:
                    u_k = inputSignal_k[current_k_step_index]

            u_k_vector = u_k if isinstance(u_k, np.ndarray) else np.array([u_k])
            self._updateStates_CoupledDDE(u_k_vector, self.simulation_time)

        if current_k_step_index == numSteps_k -1:
            current_global_time = numSteps_k * self.T_step
            state_vector_last_k = np.zeros(self.numVirtualNodes)
            for i_node_unit in range(self.numNodesPerUnit):
                t_sample1 = current_global_time - (self.numNodesPerUnit - (i_node_unit + 1)) * self.theta
                steps_ago1 = int(round((current_global_time - t_sample1) / self.integrationStep))
                state_vector_last_k[i_node_unit] = self._get_delayed_state(self.history1, steps_ago1)
            
            for i_node_unit in range(self.numNodesPerUnit):
                global_node_index = self.numNodesPerUnit + i_node_unit
                t_sample2 = current_global_time - (self.numVirtualNodes - (global_node_index + 1)) * self.theta
                steps_ago2 = int(round((current_global_time - t_sample2) / self.integrationStep))
                state_vector_last_k[global_node_index] = self._get_delayed_state(self.history2, steps_ago2)

            y_last_k = (self.Wout @ state_vector_last_k).flatten()

            if numSteps_k > washout_k:
                if pred_collect_idx < effective_pred_steps_k:
                    predictions[pred_collect_idx] = y_last_k
                    internal_states[:, state_collect_idx] = state_vector_last_k
                    pred_collect_idx += 1
                    state_collect_idx += 1

        if predictions.shape[0] != effective_pred_steps_k:
            warnings.warn(f"Final prediction shape mismatch: expected {effective_pred_steps_k}, got {predictions.shape[0]}. Adjusting.")
            predictions = predictions[:effective_pred_steps_k]
            internal_states = internal_states[:,:effective_pred_steps_k]

        return predictions, internal_states

    def generativePredict(self, initialInputs_k, generationSteps_k, output_to_input_mapper=None):
        if self.Wout is None:
            raise RuntimeError("Model must be trained before generative prediction.")

        initialInputs_k = self._validate_input_shape(initialInputs_k)
        numInitial_k = initialInputs_k.shape[0]

        total_k_steps = numInitial_k + generationSteps_k
        predictions = np.zeros((total_k_steps, self.outputSize))

        self.x1_current = 0.0
        self.x2_current = 0.0
        buffer_size = self.tau_steps + 5
        self.history1 = deque(np.zeros(buffer_size), maxlen=buffer_size)
        self.history2 = deque(np.zeros(buffer_size), maxlen=buffer_size)
        self.simulation_time = 0.0
        last_y = np.zeros(self.outputSize)

        current_k_step_index = 0
        u_k = initialInputs_k[current_k_step_index]
        num_sim_steps_warmup = numInitial_k * self.T_step_steps

        for sim_step in range(num_sim_steps_warmup):
            if sim_step > 0 and sim_step % self.T_step_steps == 0:
                current_k_step_index += 1
                current_global_time = current_k_step_index * self.T_step
                state_vector_k_minus_1 = np.zeros(self.numVirtualNodes)
                for i_node_unit in range(self.numNodesPerUnit):
                    t_sample1 = current_global_time - (self.numNodesPerUnit - (i_node_unit + 1)) * self.theta
                    steps_ago1 = int(round((current_global_time - t_sample1) / self.integrationStep))
                    state_vector_k_minus_1[i_node_unit] = self._get_delayed_state(self.history1, steps_ago1)
                
                for i_node_unit in range(self.numNodesPerUnit):
                    global_node_index = self.numNodesPerUnit + i_node_unit
                    t_sample2 = current_global_time - (self.numVirtualNodes - (global_node_index + 1)) * self.theta
                    steps_ago2 = int(round((current_global_time - t_sample2) / self.integrationStep))
                    state_vector_k_minus_1[global_node_index] = self._get_delayed_state(self.history2, steps_ago2)

                y_k_minus_1 = (self.Wout @ state_vector_k_minus_1).flatten()
                predictions[current_k_step_index - 1] = y_k_minus_1
                last_y = y_k_minus_1

                if current_k_step_index < numInitial_k:
                    u_k = initialInputs_k[current_k_step_index]

            u_k_vector = u_k if isinstance(u_k, np.ndarray) else np.array([u_k])
            self._updateStates_CoupledDDE(u_k_vector, self.simulation_time)

        for k_gen in range(generationSteps_k):
            current_k_step_index = numInitial_k + k_gen

            if self.inputSize == self.outputSize:
                u_k_gen = last_y
            else:
                if output_to_input_mapper is None:
                    raise ValueError(f"inputSize != outputSize. Provide output_to_input_mapper.")
                u_k_gen = output_to_input_mapper(last_y)
                if not isinstance(u_k_gen, np.ndarray): u_k_gen = np.array(u_k_gen)
                if u_k_gen.ndim > 1: u_k_gen = u_k_gen.flatten()
                if len(u_k_gen) != self.inputSize: raise ValueError(f"output_to_input_mapper failed size check.")

            u_k_vector = u_k_gen.reshape(1, -1)

            num_sim_steps_gen = self.T_step_steps
            for _ in range(num_sim_steps_gen):
                self._updateStates_CoupledDDE(u_k_vector[0], self.simulation_time)

            current_global_time = (current_k_step_index + 1) * self.T_step
            state_vector_k = np.zeros(self.numVirtualNodes)
            for i_node_unit in range(self.numNodesPerUnit):
                t_sample1 = current_global_time - (self.numNodesPerUnit - (i_node_unit + 1)) * self.theta
                steps_ago1 = int(round((current_global_time - t_sample1) / self.integrationStep))
                state_vector_k[i_node_unit] = self._get_delayed_state(self.history1, steps_ago1)

            for i_node_unit in range(self.numNodesPerUnit):
                global_node_index = self.numNodesPerUnit + i_node_unit
                t_sample2 = current_global_time - (self.numVirtualNodes - (global_node_index + 1)) * self.theta
                steps_ago2 = int(round((current_global_time - t_sample2) / self.integrationStep))
                state_vector_k[global_node_index] = self._get_delayed_state(self.history2, steps_ago2)

            y_k = (self.Wout @ state_vector_k).flatten()
            predictions[current_k_step_index] = y_k
            last_y = y_k

        return predictions

class ESNPipeline:
    def __init__(self, esn_params, scaler=None):
        self.esn_params = esn_params
        self.esn = TwoNodeDelayESN(**esn_params)
        self.scaler = scaler if scaler is not None else StandardScaler()
        self.trainData_scaled = None
        self.testData_scaled = None
        self._is_trained = False
        self._target_cols_indices = None

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

        if self.esn.inputSize == 1 and len(input_cols) > 1:
            input_cols = input_cols[[0]]
        elif len(input_cols) != self.esn.inputSize:
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

        return trainInput, trainTarget, testInput, testTarget

    def _inverse_transform_output(self, scaled_output):
        if self.scaler is None or self._target_cols_indices is None:
            return scaled_output

        n_samples = scaled_output.shape[0]
        n_features_original = self.scaler.n_features_in_
        temp_full_scaled = np.zeros((n_samples, n_features_original))

        if len(self._target_cols_indices) != scaled_output.shape[1]:
            warnings.warn("Mismatch between target columns indices and scaled output shape during inverse transform.")
            if len(self._target_cols_indices) < scaled_output.shape[1]:
                scaled_output = scaled_output[:, :len(self._target_cols_indices)]
            else:
                return scaled_output

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

            mse = mean_squared_error(testTarget_orig_eval, predictions_orig)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(testTarget_orig_eval, predictions_orig)

            try:
                full_testTarget_orig = self._inverse_transform_output(self.original_testTarget_scaled)
                target_range = np.ptp(full_testTarget_orig, axis=0)
                mean_target_range = np.mean(target_range) if target_range.size > 0 else 0
            except Exception:
                warnings.warn("Could not calculate target range on original test set. Using evaluation target range for NRMSE.")
                target_range = np.ptp(testTarget_orig_eval, axis=0)
                mean_target_range = np.mean(target_range) if target_range.size > 0 else 0

            nrmse = rmse / mean_target_range if mean_target_range > 1e-9 else np.inf
            metrics = { 'mse': mse, 'rmse': rmse, 'mae': mae, 'nrmse': nrmse }

        return predictions_orig, testTarget_orig_eval, metrics, internal_states_scaled

    def exportResults(self, metrics, filename='esn_results_metrics.csv', predictions=None, actuals=None, pred_filename='esn_predictions.csv'):
        if metrics is None:
            return None

        df_metrics = pd.DataFrame([metrics])
        df_metrics.to_csv(filename, index=False)

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

    def run(self, timeseries, trainRatio=0.7, predictionHorizon=1, washout_train=100, washout_pred=0):
        trainInput, trainTarget, testInput, testTarget_orig_scaled = self.prepareData(timeseries, trainRatio=trainRatio, predictionHorizon=predictionHorizon)
        self.trainModel(trainInput, trainTarget, washout_k=washout_train)
        predictions_orig, testTarget_orig_eval, metrics, internal_states_scaled = self.evaluateModel(testInput, testTarget_orig_scaled, washout_pred_k=washout_pred)

        return predictions_orig, testTarget_orig_eval, metrics, internal_states_scaled