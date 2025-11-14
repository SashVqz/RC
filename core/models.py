# core/models.py
# this module implements various reservoir architectures for Echo State Networks (ESNs)

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs
from collections import deque
import math
import warnings

# - Standard -
class EchoStateNetwork:
    def __init__(self, inputSize, reservoirSize, outputSize,
                spectralRadius=0.9,
                inputScaling=1.0,
                leakingRate=1.0,
                sparsity=0.1,
                ridgeParam=1e-8,
                activation=None,
                feedback=False,
                feedbackScaling=1.0,
                stateNoise=0.0,
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

# - Single-node delay, Reservoir computing based on delay-dynamical systems by Lennert Appeltant (1.4) -
class SingleNodeDelayESN:
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
        self.feedback = feedback
        self.stateNoise = stateNoise

        self.tau_steps = max(1, int(round(self.tau / self.integrationStep)))
        self.theta_steps = max(1, int(round(self.theta / self.integrationStep)))

        self.numVirtualNodes = int(round(self.tau_steps / self.theta_steps))
        self.tau = self.numVirtualNodes * self.theta_steps * self.integrationStep
        self.theta = self.theta_steps * self.integrationStep

        self.input_masks = np.random.choice([-maskScaling, maskScaling], size=(self.inputSize, self.numVirtualNodes))

        buffer_size = self.tau_steps + 5
        self.history = deque(np.zeros(buffer_size), maxlen=buffer_size)
        self.x_current = 0.0
        self.simulation_time = 0.0

        if self.feedback:
            self.feedbackScaling = feedbackScaling
            self.y_prev = np.zeros(self.outputSize)
        else:
            self.feedbackScaling = 0.0
            self.y_prev = None

        self.Wout = None

    def _validate_input_shape(self, u):
        u = np.atleast_2d(u)
        if u.shape[1] != self.inputSize and u.shape[0] == self.inputSize:
            u = u.T
        if u.shape[1] != self.inputSize:
            raise ValueError(f"Input signal shape mismatch: expected (n_steps, {self.inputSize}), got {u.shape}")
        return u

    def _get_masked_input(self, u_k, t):
        time_in_cycle = t % self.tau
        virtual_node_index = int(math.floor(time_in_cycle / self.theta))
        virtual_node_index = max(0, min(virtual_node_index, self.numVirtualNodes - 1))
        
        current_mask_slice = self.input_masks[:, virtual_node_index]
        J_t = self.gamma * np.dot(current_mask_slice, u_k)
        
        return J_t

    def _updateState_DDE(self, u_k_step, current_sim_time):
        if len(self.history) < self.tau_steps:
            x_delayed = 0.0
        else:
            try:
                idx_for_delay = len(self.history) - self.tau_steps -1
                if idx_for_delay < 0: x_delayed = 0.0
                else: x_delayed = self.history[idx_for_delay]
            except IndexError:
                x_delayed = 0.0

        J_t = self._get_masked_input(u_k_step, current_sim_time) 

        feedback_term = 0.0
        if self.feedback and self.y_prev is not None:
            feedback_term = self.feedbackScaling * np.sum(self.y_prev)

        nonlinear_input = x_delayed + J_t + feedback_term
        nonlinear_output = self.eta * self.activation(nonlinear_input)

        x_new = self.x_current + self.integrationStep * (-self.x_current + nonlinear_output)

        if self.stateNoise > 0:
            x_new += np.random.normal(0, self.stateNoise * np.sqrt(self.integrationStep))

        self.history.append(self.x_current)
        self.x_current = x_new
        self.simulation_time += self.integrationStep

        return self.x_current

    def _harvestStates(self, inputSignal_k, washout_k=0):
        inputSignal_k = self._validate_input_shape(inputSignal_k)
        numSteps_k = inputSignal_k.shape[0]

        if washout_k >= numSteps_k:
            raise ValueError(f"Washout ({washout_k}) >= numSteps ({numSteps_k}).")

        num_simulation_steps_total = numSteps_k * self.tau_steps
        num_effective_k_steps = numSteps_k - washout_k

        self.x_current = 0.0
        buffer_size = self.tau_steps + 5
        self.history = deque(np.zeros(buffer_size), maxlen=buffer_size)
        self.simulation_time = 0.0
        if self.feedback:
            self.y_prev = np.zeros(self.outputSize)

        collected_states = np.zeros((self.numVirtualNodes, num_effective_k_steps))
        state_collect_idx = 0

        current_k_step_index = 0
        u_k = inputSignal_k[current_k_step_index]

        for sim_step in range(num_simulation_steps_total):
            if sim_step > 0 and sim_step % self.tau_steps == 0:
                current_k_step_index += 1
                if current_k_step_index < numSteps_k:
                    u_k = inputSignal_k[current_k_step_index]

                if current_k_step_index > washout_k:
                    current_global_time = current_k_step_index * self.tau
                    temp_states_k = np.zeros(self.numVirtualNodes)
                    for i_node in range(self.numVirtualNodes):
                        t_sample = current_global_time - (self.numVirtualNodes - (i_node + 1)) * self.theta
                        steps_ago = int(round((current_global_time - t_sample) / self.integrationStep))
                        hist_index = len(self.history) - 1 - steps_ago
                        if 0 <= hist_index < len(self.history):
                            try:
                                temp_states_k[i_node] = self.history[hist_index]
                            except IndexError:
                                temp_states_k[i_node] = 0.0
                        else:
                            temp_states_k[i_node] = 0.0

                    if state_collect_idx < num_effective_k_steps:
                        collected_states[:, state_collect_idx] = temp_states_k
                        state_collect_idx += 1
                    else:
                        warnings.warn(f"State collection index {state_collect_idx} out of bounds ({num_effective_k_steps}).")

            self._updateState_DDE(u_k, self.simulation_time)

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
            self.x_current = 0.0
            buffer_size = self.tau_steps + 5
            self.history = deque(np.zeros(buffer_size), maxlen=buffer_size)
            self.simulation_time = 0.0
            if self.feedback:
                self.y_prev = np.zeros(self.outputSize)

        num_simulation_steps_total = numSteps_k * self.tau_steps
        pred_collect_idx = 0
        state_collect_idx = 0
        current_k_step_index = 0
        u_k = inputSignal_k[current_k_step_index]
        last_collected_state_vector = None

        for sim_step in range(num_simulation_steps_total):
            if sim_step > 0 and sim_step % self.tau_steps == 0:
                current_k_step_index += 1

                current_global_time = current_k_step_index * self.tau
                state_vector_k_minus_1 = np.zeros(self.numVirtualNodes)
                for i_node in range(self.numVirtualNodes):
                    t_sample = current_global_time - (self.numVirtualNodes - (i_node + 1)) * self.theta
                    steps_ago = int(round((current_global_time - t_sample) / self.integrationStep))
                    hist_index = len(self.history) - 1 - steps_ago
                    if 0 <= hist_index < len(self.history):
                        try:
                            state_vector_k_minus_1[i_node] = self.history[hist_index]
                        except IndexError: state_vector_k_minus_1[i_node] = 0.0
                    else:
                        state_vector_k_minus_1[i_node] = 0.0

                y_k_minus_1 = (self.Wout @ state_vector_k_minus_1).flatten()

                if current_k_step_index > washout_k:
                    if pred_collect_idx < effective_pred_steps_k:
                        predictions[pred_collect_idx] = y_k_minus_1
                        internal_states[:, state_collect_idx] = state_vector_k_minus_1
                        pred_collect_idx += 1
                        state_collect_idx += 1
                    else:
                        warnings.warn("Prediction index out of bounds, check logic.")

                if self.feedback:
                    self.y_prev = y_k_minus_1

                if current_k_step_index < numSteps_k:
                    u_k = inputSignal_k[current_k_step_index]

            self._updateState_DDE(u_k, self.simulation_time)

        if current_k_step_index == numSteps_k -1:
            current_global_time = numSteps_k * self.tau
            state_vector_last_k = np.zeros(self.numVirtualNodes)
            for i_node in range(self.numVirtualNodes):
                t_sample = current_global_time - (self.numVirtualNodes - (i_node + 1)) * self.theta
                steps_ago = int(round((current_global_time - t_sample) / self.integrationStep))
                hist_index = len(self.history) - 1 - steps_ago
                if 0 <= hist_index < len(self.history):
                    try: state_vector_last_k[i_node] = self.history[hist_index]
                    except IndexError: state_vector_last_k[i_node] = 0.0
                else: state_vector_last_k[i_node] = 0.0

            y_last_k = (self.Wout @ state_vector_last_k).flatten()

            if numSteps_k > washout_k:
                if pred_collect_idx < effective_pred_steps_k:
                    predictions[pred_collect_idx] = y_last_k
                    internal_states[:, state_collect_idx] = state_vector_last_k
                    pred_collect_idx += 1
                    state_collect_idx += 1

            if self.feedback:
                self.y_prev = y_last_k

        if predictions.shape[0] != effective_pred_steps_k:
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

        self.x_current = 0.0
        buffer_size = self.tau_steps + 5
        self.history = deque(np.zeros(buffer_size), maxlen=buffer_size)
        self.simulation_time = 0.0
        self.y_prev = np.zeros(self.outputSize)

        current_k_step_index = 0
        u_k = initialInputs_k[current_k_step_index]
        num_sim_steps_warmup = numInitial_k * self.tau_steps

        for sim_step in range(num_sim_steps_warmup):
            if sim_step > 0 and sim_step % self.tau_steps == 0:
                current_k_step_index += 1
                current_global_time = current_k_step_index * self.tau
                state_vector_k_minus_1 = np.zeros(self.numVirtualNodes)
                for i_node in range(self.numVirtualNodes):
                    t_sample = current_global_time - (self.numVirtualNodes - (i_node + 1)) * self.theta
                    steps_ago = int(round((current_global_time - t_sample) / self.integrationStep))
                    hist_index = len(self.history) - 1 - steps_ago
                    if 0 <= hist_index < len(self.history):
                        try: state_vector_k_minus_1[i_node] = self.history[hist_index]
                        except IndexError: state_vector_k_minus_1[i_node] = 0.0
                    else: state_vector_k_minus_1[i_node] = 0.0

                y_k_minus_1 = (self.Wout @ state_vector_k_minus_1).flatten()
                predictions[current_k_step_index - 1] = y_k_minus_1
                self.y_prev = y_k_minus_1

                if current_k_step_index < numInitial_k:
                    u_k = initialInputs_k[current_k_step_index]

            self._updateState_DDE(u_k, self.simulation_time)

        last_y = self.y_prev
        for k_gen in range(generationSteps_k):
            current_k_step_index = numInitial_k + k_gen

            if self.inputSize == self.outputSize:
                u_k_gen = last_y
            else:
                if output_to_input_mapper is None:
                    raise ValueError(f"inputSize ({self.inputSize}) != outputSize ({self.outputSize}). Provide output_to_input_mapper.")
                u_k_gen = output_to_input_mapper(last_y)
                if not isinstance(u_k_gen, np.ndarray): u_k_gen = np.array(u_k_gen)
                if u_k_gen.ndim > 1: u_k_gen = u_k_gen.flatten()
                if len(u_k_gen) != self.inputSize: raise ValueError(f"output_to_input_mapper must return vector of size {self.inputSize}, got {len(u_k_gen)}")

            num_sim_steps_gen = self.tau_steps
            for _ in range(num_sim_steps_gen):
                self._updateState_DDE(u_k_gen, self.simulation_time)

            current_global_time = (current_k_step_index + 1) * self.tau
            state_vector_k = np.zeros(self.numVirtualNodes)
            for i_node in range(self.numVirtualNodes):
                t_sample = current_global_time - (self.numVirtualNodes - (i_node + 1)) * self.theta
                steps_ago = int(round((current_global_time - t_sample) / self.integrationStep))
                hist_index = len(self.history) - 1 - steps_ago
                if 0 <= hist_index < len(self.history):
                    try: state_vector_k[i_node] = self.history[hist_index]
                    except IndexError: state_vector_k[i_node] = 0.0
                else: state_vector_k[i_node] = 0.0

            y_k = (self.Wout @ state_vector_k).flatten()
            predictions[current_k_step_index] = y_k

            last_y = y_k
            self.y_prev = y_k

        return predictions
    
# - Two-node delay, Reservoir computing based on delay-dynamical systems by Lennert Appeltant (6.2) -
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

        self.input_masks1 = np.random.choice([-maskScaling, maskScaling], size=(self.inputSize, self.numNodesPerUnit))
        self.input_masks2 = np.random.choice([-maskScaling, maskScaling], size=(self.inputSize, self.numNodesPerUnit))

        buffer_size = self.tau_steps + 5
        self.history1 = deque(np.zeros(buffer_size), maxlen=buffer_size)
        self.history2 = deque(np.zeros(buffer_size), maxlen=buffer_size)
        self.x1_current = 0.0
        self.x2_current = 0.0
        self.simulation_time = 0.0

        self.feedback = feedback

        if self.feedback:
            self.feedbackScaling = feedbackScaling
            self.y_prev = np.zeros(self.outputSize)
        else:
            self.feedbackScaling = 0.0
            self.y_prev = None

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
        
        current_mask_slice1 = self.input_masks1[:, virtual_node_index_unit]
        current_mask_slice2 = self.input_masks2[:, virtual_node_index_unit]

        J1_t = self.gamma * np.dot(current_mask_slice1, u_k)
        J2_t = self.gamma * np.dot(current_mask_slice2, u_k)
        
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

        feedback_term = 0.0
        if self.feedback and self.y_prev is not None:
            feedback_term = self.feedbackScaling * np.sum(self.y_prev)

        nonlinear_input1 = x1_delayed_tau + x2_delayed_half_tau + J1_t + feedback_term
        nonlinear_input2 = x2_delayed_tau + x1_delayed_half_tau + J2_t + feedback_term

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

        num_simulation_steps_total = numSteps_k * self.half_tau_steps
        num_effective_k_steps = numSteps_k - washout_k

        self.x1_current = 0.0
        self.x2_current = 0.0
        buffer_size = self.tau_steps + 5
        self.history1 = deque(np.zeros(buffer_size), maxlen=buffer_size)
        self.history2 = deque(np.zeros(buffer_size), maxlen=buffer_size)
        self.simulation_time = 0.0
        
        if self.feedback:
            self.y_prev = np.zeros(self.outputSize)

        collected_states = np.zeros((self.numVirtualNodes, num_effective_k_steps))
        state_collect_idx = 0

        current_k_step_index = 0
        u_k = inputSignal_k[current_k_step_index]

        for sim_step in range(num_simulation_steps_total):
            if sim_step > 0 and sim_step % self.half_tau_steps == 0:
                current_k_step_index += 1
                if current_k_step_index < numSteps_k:
                    u_k = inputSignal_k[current_k_step_index]

                if current_k_step_index > washout_k:
                    current_global_time = current_k_step_index * self.T_step
                    state_vector_k = np.zeros(self.numVirtualNodes)

                    for i_node_unit in range(self.numNodesPerUnit):
                        t_sample1 = current_global_time - i_node_unit * self.theta
                        steps_ago1 = int(round((current_global_time - t_sample1) / self.integrationStep))
                        state_vector_k[i_node_unit] = self._get_delayed_state(self.history1, steps_ago1)

                    for i_node_unit in range(self.numNodesPerUnit):
                        global_node_index = self.numNodesPerUnit + i_node_unit
                        t_sample2 = current_global_time - i_node_unit * self.theta
                        steps_ago2 = int(round((current_global_time - t_sample2) / self.integrationStep))
                        state_vector_k[global_node_index] = self._get_delayed_state(self.history2, steps_ago2)


                    if state_collect_idx < num_effective_k_steps:
                        collected_states[:, state_collect_idx] = state_vector_k
                        state_collect_idx += 1
                    else:
                        warnings.warn(f"State collection index {state_collect_idx} out of bounds ({num_effective_k_steps}).")

            self._updateStates_CoupledDDE(u_k, self.simulation_time)

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
            if self.feedback: self.y_prev = np.zeros(self.outputSize)

        num_simulation_steps_total = numSteps_k * self.half_tau_steps
        pred_collect_idx = 0
        state_collect_idx = 0
        current_k_step_index = 0
        u_k = inputSignal_k[current_k_step_index]

        for sim_step in range(num_simulation_steps_total):
            if sim_step > 0 and sim_step % self.half_tau_steps == 0:
                current_k_step_index += 1

                current_global_time = current_k_step_index * self.T_step
                state_vector_k_minus_1 = np.zeros(self.numVirtualNodes)

                for i_node_unit in range(self.numNodesPerUnit):
                    t_sample1 = current_global_time - i_node_unit * self.theta
                    steps_ago1 = int(round((current_global_time - t_sample1) / self.integrationStep))
                    state_vector_k_minus_1[i_node_unit] = self._get_delayed_state(self.history1, steps_ago1)

                for i_node_unit in range(self.numNodesPerUnit):
                    global_node_index = self.numNodesPerUnit + i_node_unit
                    t_sample2 = current_global_time - i_node_unit * self.theta
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
                
                if self.feedback:
                    self.y_prev = y_k_minus_1

                if current_k_step_index < numSteps_k:
                    u_k = inputSignal_k[current_k_step_index]

            self._updateStates_CoupledDDE(u_k, self.simulation_time)

        if current_k_step_index == numSteps_k -1:
            current_global_time = numSteps_k * self.T_step
            state_vector_last_k = np.zeros(self.numVirtualNodes)

            for i_node_unit in range(self.numNodesPerUnit):
                t_sample1 = current_global_time - i_node_unit * self.theta
                steps_ago1 = int(round((current_global_time - t_sample1) / self.integrationStep))
                state_vector_last_k[i_node_unit] = self._get_delayed_state(self.history1, steps_ago1)
            
            for i_node_unit in range(self.numNodesPerUnit):
                global_node_index = self.numNodesPerUnit + i_node_unit
                t_sample2 = current_global_time - i_node_unit * self.theta
                steps_ago2 = int(round((current_global_time - t_sample2) / self.integrationStep))
                state_vector_last_k[global_node_index] = self._get_delayed_state(self.history2, steps_ago2)

            y_last_k = (self.Wout @ state_vector_last_k).flatten()

            if numSteps_k > washout_k:
                if pred_collect_idx < effective_pred_steps_k:
                    predictions[pred_collect_idx] = y_last_k
                    internal_states[:, state_collect_idx] = state_vector_last_k
                    pred_collect_idx += 1
                    state_collect_idx += 1
            
            if self.feedback:
                self.y_prev = y_last_k

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
        
        self.y_prev = np.zeros(self.outputSize)
        last_y = np.zeros(self.outputSize)

        current_k_step_index = 0
        u_k = initialInputs_k[current_k_step_index]
        num_sim_steps_warmup = numInitial_k * self.half_tau_steps

        for sim_step in range(num_sim_steps_warmup):
            if sim_step > 0 and sim_step % self.half_tau_steps == 0:
                current_k_step_index += 1
                current_global_time = current_k_step_index * self.T_step
                state_vector_k_minus_1 = np.zeros(self.numVirtualNodes)
                for i_node_unit in range(self.numNodesPerUnit):
                    t_sample1 = current_global_time - i_node_unit * self.theta
                    steps_ago1 = int(round((current_global_time - t_sample1) / self.integrationStep))
                    state_vector_k_minus_1[i_node_unit] = self._get_delayed_state(self.history1, steps_ago1)
                
                for i_node_unit in range(self.numNodesPerUnit):
                    global_node_index = self.numNodesPerUnit + i_node_unit
                    t_sample2 = current_global_time - i_node_unit * self.theta
                    steps_ago2 = int(round((current_global_time - t_sample2) / self.integrationStep))
                    state_vector_k_minus_1[global_node_index] = self._get_delayed_state(self.history2, steps_ago2)

                y_k_minus_1 = (self.Wout @ state_vector_k_minus_1).flatten()
                predictions[current_k_step_index - 1] = y_k_minus_1
                last_y = y_k_minus_1
                
                if self.feedback:
                    self.y_prev = y_k_minus_1

                if current_k_step_index < numInitial_k:
                    u_k = initialInputs_k[current_k_step_index]

            self._updateStates_CoupledDDE(u_k, self.simulation_time)

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

            num_sim_steps_gen = self.half_tau_steps
            for _ in range(num_sim_steps_gen):
                self._updateStates_CoupledDDE(u_k_gen, self.simulation_time)

            current_global_time = (current_k_step_index + 1) * self.T_step
            state_vector_k = np.zeros(self.numVirtualNodes)
            for i_node_unit in range(self.numNodesPerUnit):
                t_sample1 = current_global_time - i_node_unit * self.theta
                steps_ago1 = int(round((current_global_time - t_sample1) / self.integrationStep))
                state_vector_k[i_node_unit] = self._get_delayed_state(self.history1, steps_ago1)

            for i_node_unit in range(self.numNodesPerUnit):
                global_node_index = self.numNodesPerUnit + i_node_unit
                t_sample2 = current_global_time - i_node_unit * self.theta
                steps_ago2 = int(round((current_global_time - t_sample2) / self.integrationStep))
                state_vector_k[global_node_index] = self._get_delayed_state(self.history2, steps_ago2)

            y_k = (self.Wout @ state_vector_k).flatten()
            predictions[current_k_step_index] = y_k
            
            last_y = y_k
            if self.feedback:
                self.y_prev = y_k

        return predictions