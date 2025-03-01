"""
This module implements a Variational Quantum Classifier (QVC) using PennyLane.
The classifier is compatible with scikit-learn's estimator API and supports several
ansatz implementations for variational quantum circuits.
"""

import operator
from functools import reduce
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pennylane as qml
import scipy.linalg as la
from pennylane import numpy as np
from pennylane.operation import Operation
from pennylane.optimize import NesterovMomentumOptimizer
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import (
    check_array,
    check_is_fitted,
    check_random_state,
    validate_data,
)

# Define type aliases for clarity
DatasetType = Dict[str, Union[List[str], np.ndarray, Any]]
DataDict = Dict[str, DatasetType]
SizeType = Union[int, Tuple["SizeType", ...]]


class BaseQVC(ClassifierMixin, BaseEstimator):
    """
    Variational Quantum Classifier (QVC).

    This classifier implements a variational quantum circuit using PennyLane and adheres
    to scikit-learn's estimator API. It can be employed as a base estimator, for instance,
    with ensemble methods such as AdaBoostClassifier.

    Attributes:
        n_layers (int): Number of layers in the variational circuit.
        max_iter (int): Maximum number of training iterations (epochs).
        batch_size (int): Batch size used during training.
        random_state (Optional[Any]): Seed or random state for reproducibility.
    """

    multi_class: bool = False
    _estimator_type: str = "classifier"

    def __init__(
        self,
        n_layers: int = 1,
        max_iter: int = 10,
        batch_size: int = 32,
        random_state: Optional[Any] = None,
        n_qubits: Optional[int] = None,
    ) -> None:
        """
        Initialize the Variational Quantum Classifier.

        Args:
            n_layers (int): Number of layers in the variational circuit.
            max_iter (int): Maximum number of training iterations.
            batch_size (int): Batch size for training.
            random_state (Optional[Any]): Seed or random state for reproducibility.
        """
        self.n_layers = n_layers
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.random_state = random_state
        self.n_qubits_ = n_qubits

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[Union[np.ndarray, List[float]]] = None,
    ) -> "BaseQVC":
        """
        Fit the variational quantum classifier to the training data.

        Args:
            X (np.ndarray): Training feature matrix.
            y (np.ndarray): Training labels.
            sample_weight (Optional[Union[np.ndarray, List[float]]]): Sample weights for training.

        Returns:
            BaseQVC: The fitted classifier.
        """
        X = check_array(X)
        y = self.transform_y(y)
        X, y = validate_data(self, X, y)

        if sample_weight is not None:
            sample_weight = np.array(sample_weight, dtype=float)
            if sample_weight.shape[0] != X.shape[0]:
                raise ValueError(
                    "sample_weight must have the same number of samples as X and y."
                )

        if self.n_qubits_ is None:
            self.n_qubits_ = self.get_n_qubits(X)
        self.device_ = qml.device("default.qubit", wires=self.n_qubits_)

        # Store the classes observed during fitting.
        self.classes_ = unique_labels(y)
        self.random_state_ = check_random_state(self.random_state)

        # Initialize optimizer.
        opt = NesterovMomentumOptimizer(0.01)

        self.weights_ = self.get_weights()
        self.bias_ = np.array(0.0, requires_grad=True)

        self.loss_curve_ = [self.compute_train_cost(X, y, sample_weight)]
        len_train: int = X.shape[0]

        for n_iter in range(1, self.max_iter + 1):
            self.n_iter_ = n_iter
            indices: np.ndarray = np.arange(len_train)
            self.random_state_.shuffle(indices)

            for start_idx in range(0, len_train, self.batch_size):
                end_idx = start_idx + self.batch_size
                batch_indices: np.ndarray = indices[start_idx:end_idx]

                x_batch: np.ndarray = np.array(
                    X[batch_indices], requires_grad=False, dtype=float
                )
                y_batch: np.ndarray = np.array(y[batch_indices], requires_grad=False)

                if sample_weight is not None:
                    sw_batch: np.ndarray = np.array(
                        sample_weight[batch_indices], requires_grad=False, dtype=float
                    )
                    (
                        self.weights_,
                        self.bias_,
                        _,
                        _,
                        _,
                    ) = opt.step(
                        self.cost, self.weights_, self.bias_, x_batch, y_batch, sw_batch
                    )
                else:
                    (
                        self.weights_,
                        self.bias_,
                        _,
                        _,
                    ) = opt.step(self.cost, self.weights_, self.bias_, x_batch, y_batch)

            train_cost = self.compute_train_cost(X, y, sample_weight)
            self.loss_curve_.append(train_cost)
            print(f"Epoch: {n_iter:5d} | Training Cost: {train_cost:0.7f}")

        return self

    def compute_train_cost(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[Union[np.ndarray, List[float]]] = None,
    ) -> float:
        """
        Compute the training cost using the current model parameters on the provided dataset.

        This method calculates the cost (loss) of the model on the training data. If sample weights
        are provided, they are incorporated into the cost calculation.

        Args:
            X (np.ndarray): The training feature matrix.
            y (np.ndarray): The training target vector.
            sample_weight (Optional[Union[np.ndarray, List[float]]]): Optional sample weights for each training instance.

        Returns:
            float: The computed training cost.
        """
        if sample_weight is not None:
            train_cost: float = self.cost(
                self.weights_, self.bias_, X, y, sample_weight
            )
        else:
            train_cost = self.cost(self.weights_, self.bias_, X, y)
        return train_cost

    def transform_y(self, y_true: Union[np.ndarray, List[int]]) -> np.ndarray:
        """
        Transform the original labels into a binary format of -1 and 1.

        Args:
            y_true (Union[np.ndarray, List[int]]): Original labels.

        Returns:
            np.ndarray: Transformed labels with values -1 and 1.
        """
        unique_labels_arr: np.ndarray = np.sort(np.unique(y_true))
        mapping: Dict[int, int] = {
            int(unique_labels_arr[0]): -1,
            int(unique_labels_arr[1]): 1,
        }
        self.inverse_mapping_ = {-1: unique_labels_arr[0], 1: unique_labels_arr[1]}
        y_mapped: np.ndarray = np.array([mapping[int(y)] for y in y_true])
        return y_mapped

    def inverse_transform_y(self, y_mapped: Union[np.ndarray, List[int]]) -> np.ndarray:
        """
        Revert the transformed labels back to their original representation.

        Args:
            y_mapped (Union[np.ndarray, List[int]]): Transformed labels (-1 and 1).

        Returns:
            np.ndarray: Original labels.
        """
        y_original: np.ndarray = np.array(
            [self.inverse_mapping_[int(y)] for y in y_mapped]
        )
        return y_original

    def get_n_qubits(self, X: np.ndarray) -> int:
        """
        Calculate the number of qubits required based on the dimensionality of the input features.

        Args:
            X (np.ndarray): Input feature matrix.

        Returns:
            int: Number of qubits, computed as the ceiling of log2(number of features).
        """
        return int(np.ceil(np.log2(X.shape[1])))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for the input data.

        Args:
            X (np.ndarray): Input feature matrix.

        Returns:
            np.ndarray: Predicted class labels in their original representation.
        """
        check_is_fitted(self, ["weights_", "bias_"])
        X = validate_data(self, X, reset=False)
        X = np.array(X, requires_grad=False, dtype=float)
        output: np.ndarray = np.sign(
            self.variational_classifier(self.weights_, self.bias_, X)
        )
        output = self.inverse_transform_y(output)
        return output

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for the input data.

        Args:
            X (np.ndarray): Input feature matrix.

        Returns:
            np.ndarray: Array of predicted probabilities for each class.
        """
        check_is_fitted(self, ["weights_", "bias_"])
        X = validate_data(self, X, reset=False)
        X = np.array(X, requires_grad=False, dtype=float)
        output: np.ndarray = self.variational_classifier(self.weights_, self.bias_, X)
        output = (1 + output) / 2  # Normalize output to [0, 1]
        return np.column_stack([1 - output, output])

    def variational_classifier(
        self, weights: np.ndarray, bias: float, x: np.ndarray
    ) -> np.ndarray:
        """
        Compute the output of the variational quantum classifier in a vectorized manner.

        Args:
            weights (np.ndarray): Weights for the quantum gates.
            bias (float): Bias term to be added to the circuit's output.
            x (np.ndarray): Input features embedded into the quantum circuit.

        Returns:
            np.ndarray: The output of the variational quantum classifier.
        """

        @qml.qnode(self.device_, interface="autograd")
        def quantum_circuit(weights: np.ndarray, x: np.ndarray) -> float:
            """
            Execute the quantum circuit for variational classification.

            Args:
                weights (np.ndarray): Weights for the quantum gates.
                x (np.ndarray): Input features for the quantum circuit.

            Returns:
                float: Expectation value of the observable (product of PauliZ operators).
            """
            wires = range(self.n_qubits_)
            qml.AmplitudeEmbedding(
                features=x, wires=wires, normalize=True, pad_with=0.0
            )
            self.ansatz(weights, self.n_layers)
            # Compute the product of PauliZ measurements over all wires.
            observable = reduce(operator.matmul, [qml.PauliZ(i) for i in wires])
            return qml.expval(observable)

        return quantum_circuit(weights, x) + bias

    @staticmethod
    def square_loss(
        labels: Union[np.ndarray, List[float]],
        predictions: Union[np.ndarray, List[float]],
        sample_weight: Optional[Union[np.ndarray, List[float]]] = None,
    ) -> float:
        """
        Calculate the mean squared error loss between the true labels and predictions.

        Args:
            labels (Union[np.ndarray, List[float]]): True labels.
            predictions (Union[np.ndarray, List[float]]): Predicted values.
            sample_weight (Optional[Union[np.ndarray, List[float]]]): Sample weights for loss calculation.

        Returns:
            float: Mean squared error loss.
        """
        labels = np.array(labels)
        predictions = np.array(predictions)
        if sample_weight is not None:
            sample_weight = np.array(sample_weight, dtype=float)
            return np.sum(sample_weight * (labels - predictions) ** 2) / np.sum(
                sample_weight
            )
        return np.mean((labels - predictions) ** 2)

    @staticmethod
    def log_loss(
        labels: Union[np.ndarray, List[int]],
        predictions: Union[np.ndarray, List[float]],
    ) -> float:
        """
        Calculate the logarithmic loss (cross-entropy loss) between the true labels and the predicted probabilities.

        Args:
            labels (Union[np.ndarray, List[int]]): True labels (e.g., 0 or 1 for binary classification).
            predictions (Union[np.ndarray, List[float]]): Predicted probabilities for the positive class.

        Returns:
            float: Logarithmic loss.
        """
        labels = np.array(labels)
        predictions = np.array(predictions)
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        return -np.mean(
            labels * np.log(predictions) + (1 - labels) * np.log(1 - predictions)
        )

    def cost(
        self,
        weights: np.ndarray,
        bias: float,
        x_batch: np.ndarray,
        y_batch: np.ndarray,
        sample_weight: Optional[Union[np.ndarray, List[float]]] = None,
    ) -> float:
        """
        Calculate the cost (mean squared error loss) for a batch of data using the variational quantum classifier.

        Args:
            weights (np.ndarray): Weights for the quantum gates.
            bias (float): Bias term to be added to the circuit's output.
            x_batch (np.ndarray): Batch of input features.
            y_batch (np.ndarray): True labels for the batch.
            sample_weight (Optional[Union[np.ndarray, List[float]]]): Sample weights for the batch.

        Returns:
            float: Mean squared error loss for the batch.
        """
        predictions: np.ndarray = self.variational_classifier(weights, bias, x_batch)
        return self.square_loss(y_batch, predictions, sample_weight)

    @staticmethod
    def accuracy(
        labels: Union[np.ndarray, List[float]],
        predictions: Union[np.ndarray, List[float]],
    ) -> float:
        """
        Calculate the accuracy of the predictions relative to the true labels.

        Args:
            labels (Union[np.ndarray, List[float]]): True labels.
            predictions (Union[np.ndarray, List[float]]): Predicted values.

        Returns:
            float: Accuracy as the fraction of correct predictions.
        """
        labels = np.array(labels)
        predictions = np.array(predictions)
        return float(np.mean(np.abs(labels - predictions) < 1e-5))

    def get_weights(self) -> np.ndarray:
        """
        Initialize the weights for the quantum circuit.

        This method should be implemented by subclasses.

        Returns:
            np.ndarray: Initial weights for the variational quantum circuit.
        """
        raise NotImplementedError("Subclasses must implement the get_weights method.")

    def get_weights_size(self) -> SizeType:
        """
        Retrieve the shape (size) of the weight tensor required by the circuit.

        This method should be implemented by subclasses.

        Returns:
            SizeType: The dimensions of the weight tensor.
        """
        raise NotImplementedError(
            "Subclasses must implement the get_weights_size method."
        )

    def ansatz(self, weights: np.ndarray, n_layers: int) -> None:
        """
        Construct the variational ansatz (quantum circuit) for the classifier.

        This method should be implemented by subclasses to define the specific quantum circuit.

        Args:
            weights (np.ndarray): Weights for the quantum gates.
            n_layers (int): The number of layers in the circuit.

        Returns:
            None
        """
        raise NotImplementedError("Subclasses must implement the ansatz method.")

    def random_unitary(self, n: int, samples: int = 1) -> np.ndarray:
        """
        Returns Haar-distributed random unitary matrix or matrices from U(n).

        Args:
            n (int): The dimension of the unitary matrix.
            samples (int, optional): The number of samples to generate. Defaults to 1.

        Returns:
            np.ndarray: If samples == 1, a Haar-distributed random unitary matrix of shape (n, n).
                        Otherwise, an array of shape (samples, n, n) containing Haar-distributed unitary matrices.
        """
        if samples == 1:
            # Generate a single random complex matrix.
            z = self.random_state_.randn(n, n) + 1.0j * self.random_state_.randn(n, n)
            q, r = np.linalg.qr(z)
            # Normalize the diagonal of r.
            d = np.diag(np.diagonal(r) / np.abs(np.diagonal(r)))
            return np.dot(q, d)
        else:
            # Generate a batch of random complex matrices of shape (samples, n, n).
            z = self.random_state_.randn(
                samples, n, n
            ) + 1.0j * self.random_state_.randn(samples, n, n)
            # Compute QR decomposition in batch.
            q, r = np.linalg.qr(z)
            # Normalize the diagonal of each R individually.
            d = np.array(
                [np.diag(np.diagonal(r_i) / np.abs(np.diagonal(r_i))) for r_i in r]
            )
            # Multiply each Q with its corresponding D.
            return np.matmul(q, d)

    def haar_integral(self, samples: int = 2048) -> np.ndarray:
        """
        Compute the Haar integral by averaging the density matrices computed from Haar-distributed unitaries.

        This implementation assumes that the method `self.random_unitary(n, samples)` returns an
        array of shape (samples, n, n) with Haar-distributed unitary matrices.

        Args:
            samples (int): The number of samples to average over.

        Returns:
            np.ndarray: The averaged density matrix computed from the Haar-distributed unitaries.
        """
        n: int = 2**self.n_qubits_
        # Create the zero state (basis state).
        zero_state: np.ndarray = np.zeros(n, dtype=complex)
        zero_state[0] = 1.0

        # Generate an array of unitary matrices (assumes vectorized implementation).
        unitaries: np.ndarray = self.random_unitary(
            n, samples
        )  # shape: (samples, n, n)

        # Since zero_state is [1, 0, ..., 0], the transformed state for each unitary is the first row.
        transformed_states: np.ndarray = unitaries[:, 0, :]  # shape: (samples, n)

        # Compute density matrices for each sample using the outer product in a vectorized fashion.
        density_matrices: np.ndarray = (
            transformed_states[:, :, np.newaxis]
            * transformed_states.conj()[:, np.newaxis, :]
        )

        # Return the average density matrix over all samples.
        return np.mean(density_matrices, axis=0)

    def pqc_integral(self, samples: int = 2048) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the integral of a Parameterized Quantum Circuit (PQC) over uniformly sampled parameters.

        This function samples parameters uniformly from the interval [-π, π], applies the ansatz to prepare
        a quantum state, and computes the density matrix |ψ⟩⟨ψ| for that state. The result is the average density
        matrix over the specified number of samples, along with the individual density matrices.

        Args:
            samples (int): The number of samples over which to average.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - The averaged density matrix of shape (2**num_qubits, 2**num_qubits).
                - The batch of density matrices for each sample.
        """
        dev = qml.device("default.qubit", wires=self.n_qubits_)

        @qml.qnode(dev, interface="autograd")
        def circuit(params: np.ndarray) -> np.ndarray:
            self.ansatz(params, self.n_layers)
            return qml.state()

        # Generate all parameter samples at once: shape (samples, *weights_size).
        parameters = self.random_state_.uniform(
            -np.pi, np.pi, (samples, *self.get_weights_size())
        )
        states = circuit(parameters)
        if states.shape != (samples, 2**self.n_qubits_):
            states = np.repeat(states[np.newaxis, :], samples, axis=0)

        # Compute the batch of density matrices via vectorized outer product.
        density_matrices: np.ndarray = np.einsum("bi,bj->bij", states, states.conj())

        # Compute the mean of all density matrices along axis 0.
        return np.mean(density_matrices, axis=0), density_matrices

    def haar(self, samples: int) -> Tuple[float, np.ndarray]:
        """
        Compute the difference between the Haar integral and the PQC integral.

        Args:
            samples (int): The number of samples to use in the integrals.

        Returns:
            Tuple[float, np.ndarray]:
                - The norm of the difference between the Haar integral and PQC integral.
                - The batch of density matrices from the PQC integral.
        """
        haar_value = self.haar_integral(samples)
        pqc_value, density_matrices = self.pqc_integral(samples)
        return np.linalg.norm(haar_value - pqc_value), density_matrices

    def meyer_wallach(self, density_matrices: np.ndarray, samples: int = 2048) -> float:
        """
        Compute the Meyer–Wallach entanglement measure for a given parameterized quantum circuit.

        Procedure:
          1. For each qubit, compute the reduced density matrices (via partial trace) for all samples.
          2. Calculate the purity of each reduced density matrix and average over all qubits.
          3. The Meyer–Wallach measure is 2 times the average over samples of (1 - average purity).

        Args:
            density_matrices (np.ndarray): Batch of density matrices with shape (samples, dim, dim).
            samples (int): The number of samples over which the measure is computed.

        Returns:
            float: The Meyer–Wallach entanglement measure.
        """
        qb: List[int] = list(range(self.n_qubits_))
        purity_sum: np.ndarray = np.zeros(samples, dtype=complex)

        # Loop over qubit indices (typically a small number) to compute reduced density matrices.
        for j in qb:
            indices_to_trace = qb.copy()
            indices_to_trace.remove(j)
            reduced_density: np.ndarray = qml.math.partial_trace(
                density_matrices, indices=indices_to_trace
            )
            # Compute the purity for each sample: Tr((ρ_reduced)^2).
            prod: np.ndarray = np.matmul(reduced_density, reduced_density)
            purity_j: np.ndarray = np.trace(prod, axis1=1, axis2=2)
            purity_sum += purity_j

        avg_purity_per_sample: np.ndarray = purity_sum / self.n_qubits_
        entanglement_values: np.ndarray = 1 - avg_purity_per_sample.real
        measure: float = 2 * np.mean(entanglement_values)
        return measure

    def measures(self, n_qubits: int, samples: int = 2048) -> Tuple[float, float]:
        """
        Compute both the Haar difference and the Meyer–Wallach entanglement measure.

        Args:
            n_qubits (int): The number of qubits.
            samples (int, optional): The number of samples to use for the integrals. Defaults to 2048.

        Returns:
            Tuple[float, float]: A tuple containing:
                - The norm difference between Haar and PQC integrals.
                - The Meyer–Wallach entanglement measure.
        """
        self.n_qubits_ = n_qubits
        self.random_state_ = check_random_state(self.random_state)
        haar_val, density_matrices = self.haar(samples)
        mw: float = self.meyer_wallach(density_matrices, samples)
        return haar_val, mw


class AnsatzSingleRot(BaseQVC):
    """
    Variational Quantum Classifier with a single rotation gate ansatz.

    This base class employs a single rotation gate (to be defined by subclasses) on each qubit.
    """

    _gate: Optional[Callable[..., Any]] = None

    def get_weights(self) -> np.ndarray:
        """
        Initialize the weights for the single rotation ansatz.

        Returns:
            np.ndarray: Initial weights with shape (n_layers, n_qubits, 1).
        """
        weights = 0.01 * self.random_state_.normal(size=self.get_weights_size())
        return np.array(weights, requires_grad=True)

    def get_weights_size(self) -> SizeType:
        """
        Retrieve the shape of the weight tensor for the single rotation ansatz.

        Returns:
            SizeType: A tuple representing (n_layers, n_qubits, 1).
        """
        return self.n_layers, self.n_qubits_, 1

    def ansatz(self, weights: np.ndarray, n_layers: int) -> None:
        """
        Apply the single rotation gate ansatz to the quantum circuit.

        Args:
            weights (np.ndarray): Weights for the rotation gates.
            n_layers (int): The number of layers in the circuit.

        Returns:
            None
        """
        if self._gate is None:
            raise NotImplementedError(
                "Subclasses must define the rotation gate (_gate)."
            )
        for n_layer in range(n_layers):
            for wire in range(self.n_qubits_):
                if len(weights.shape) == 4:
                    self._gate(weights[:, n_layer, wire, 0], wires=wire)
                else:
                    self._gate(weights[n_layer, wire, 0], wires=wire)


class AnsatzSingleRotX(AnsatzSingleRot):
    """
    Variational Quantum Classifier with a single RX rotation gate.
    """

    _gate = qml.RX


class AnsatzSingleRotY(AnsatzSingleRot):
    """
    Variational Quantum Classifier with a single RY rotation gate.
    """

    _gate = qml.RY


class AnsatzSingleRotZ(AnsatzSingleRot):
    """
    Variational Quantum Classifier with a single RZ rotation gate.
    """

    _gate = qml.RZ


class AnsatzRot(BaseQVC):
    """
    Variational Quantum Classifier using a general rotation (Rot) gate ansatz.
    """

    def get_weights(self) -> np.ndarray:
        """
        Initialize the weights for the Rot ansatz.

        Returns:
            np.ndarray: Initial weights with shape (n_layers, n_qubits, 3).
        """
        weights = 0.01 * self.random_state_.normal(size=self.get_weights_size())
        return np.array(weights, requires_grad=True)

    def get_weights_size(self) -> SizeType:
        """
        Retrieve the shape of the weight tensor for the Rot ansatz.

        Returns:
            SizeType: A tuple representing (n_layers, n_qubits, 3).
        """
        return self.n_layers, self.n_qubits_, 3

    def ansatz(self, weights: np.ndarray, n_layers: int) -> None:
        """
        Apply the Rot ansatz to the quantum circuit.

        Args:
            weights (np.ndarray): Weights for the rotation gates.
            n_layers (int): The number of layers in the circuit.

        Returns:
            None
        """
        for n_layer in range(n_layers):
            for wire in range(self.n_qubits_):
                if len(weights.shape) == 4:
                    qml.Rot(
                        weights[:, n_layer, wire, 0],
                        weights[:, n_layer, wire, 1],
                        weights[:, n_layer, wire, 2],
                        wires=wire,
                    )
                else:
                    qml.Rot(
                        weights[n_layer, wire, 0],
                        weights[n_layer, wire, 1],
                        weights[n_layer, wire, 2],
                        wires=wire,
                    )


class AnsatzRotCNOT(BaseQVC):
    """
    Variational Quantum Classifier using strongly entangling layers with CNOT gates.
    """

    def get_weights(self) -> np.ndarray:
        """
        Initialize the weights for the strongly entangling layers.

        Returns:
            np.ndarray: Initial weights with shape (n_layers, n_qubits, 3).
        """
        weights = 0.01 * self.random_state_.normal(size=self.get_weights_size())
        return np.array(weights, requires_grad=True)

    def get_weights_size(self) -> SizeType:
        """
        Retrieve the shape of the weight tensor for the strongly entangling layers.

        Returns:
            SizeType: A tuple representing (n_layers, n_qubits, 3).
        """
        return self.n_layers, self.n_qubits_, 3

    def ansatz(self, weights: np.ndarray, n_layers: int) -> None:
        """
        Apply strongly entangling layers to the quantum circuit.

        Args:
            weights (np.ndarray): Weights for the entangling layers.
            n_layers (int): The number of layers in the circuit.

        Returns:
            None
        """
        qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits_))


# List of available QVC models with different ansatz implementations.
models: List[Any] = [
    AnsatzSingleRotX,
    AnsatzSingleRotY,
    AnsatzSingleRotZ,
    AnsatzRot,
    AnsatzRotCNOT,
]
