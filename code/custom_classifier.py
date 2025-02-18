import operator
from functools import reduce
from typing import Any, Dict, List, Optional, Tuple, Union

import pennylane as qml
from pennylane import numpy as np
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


class BaseQVC(ClassifierMixin, BaseEstimator):
    """
    Variational Quantum Classifier (QVC)

    This classifier implements a variational quantum circuit using PennyLane and is
    compatible with scikit-learn's estimator API. It can be used as a base estimator,
    for example, with the AdaBoostClassifier.

    Attributes:
        n_layers (int): Number of layers in the variational circuit.
        max_iter (int): Maximum number of training iterations (epochs).
        batch_size (int): Batch size used during training.
        random_state (Optional[Any]): Seed or random state for reproducibility.
    """

    multi_class = False
    _estimator_type = "classifier"

    def __init__(
        self,
        n_layers: int = 1,
        max_iter: int = 10,
        batch_size: int = 32,
        random_state: Optional[Any] = None,
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
                    "sample_weight must have the same number of samples as X and y"
                )

        self.n_qubits_ = self.get_n_qubits(X)
        self.device_ = qml.device("default.qubit", wires=self.n_qubits_)

        # Store the classes observed during fitting
        self.classes_ = unique_labels(y)
        self.random_state_ = check_random_state(self.random_state)

        opt = NesterovMomentumOptimizer(0.01)

        self.weights_ = self.get_weights()
        self.bias_ = np.array(0.0, requires_grad=True)

        self.loss_curve_ = []

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
                    self.weights_, self.bias_, _, _, _ = opt.step(
                        self.cost, self.weights_, self.bias_, x_batch, y_batch, sw_batch
                    )
                else:
                    self.weights_, self.bias_, _, _ = opt.step(
                        self.cost, self.weights_, self.bias_, x_batch, y_batch
                    )

            if sample_weight is not None:
                train_cost: float = self.cost(
                    self.weights_, self.bias_, X, y, sample_weight
                )
            else:
                train_cost: float = self.cost(self.weights_, self.bias_, X, y)
            self.loss_curve_.append(train_cost)
            print("Epoch: {:5d} | Training Cost: {:0.7f}".format(n_iter, train_cost))

        return self

    def transform_y(self, y_true: Union[np.ndarray, List[int]]) -> np.ndarray:
        """
        Transform the original labels into a binary format of -1 and 1.

        Args:
            y_true (Union[np.ndarray, List[int]]): Original labels.

        Returns:
            np.ndarray: Transformed labels.
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
            int: Number of qubits.
        """
        return int(np.ceil(np.log2(X.shape[1])))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for the input data.

        Args:
            X (np.ndarray): Input feature matrix.

        Returns:
            np.ndarray: Predicted class labels.
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
        output = (1 + output) / 2
        return np.column_stack([1 - output, output])

    def variational_classifier(
        self, weights: np.ndarray, bias: float, x: np.ndarray
    ) -> np.ndarray:
        """
        Compute the output of the variational quantum classifier in a vectorized manner.

        Args:
            weights (np.ndarray): Weights for the quantum gates.
            bias (float): Bias term to be added to the circuit's output.
            x (np.ndarray): Input features to be embedded into the quantum circuit.

        Returns:
            np.ndarray: Output of the variational quantum classifier.
        """

        @qml.qnode(self.device_, interface="autograd")
        def quantum_circuit(weights: np.ndarray, x: np.ndarray) -> float:
            """
            Execute the quantum circuit for variational classification.

            Args:
                weights (np.ndarray): Weights for the quantum gates.
                x (np.ndarray): Input features for the quantum circuit.

            Returns:
                float: Expectation value of the PauliZ operator from the circuit.
            """
            wires = range(self.n_qubits_)
            qml.AmplitudeEmbedding(
                features=x, wires=wires, normalize=True, pad_with=0.0
            )
            self.ansatz(weights, wires, self.n_layers)
            result = reduce(operator.matmul, [qml.PauliZ(i) for i in wires])
            return qml.expval(result)

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

    def get_weights(self) -> Tuple[np.ndarray, float]:
        """
        Initialize the weights for the quantum circuit.

        Returns:
            Tuple[np.ndarray, float]: Initial weights and bias.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def ansatz(
        self, weights: np.ndarray, wires: Union[List[int], range], n_layers: int
    ) -> None:
        raise NotImplementedError("Subclasses must implement this method.")


class AnsatzSingleRot(BaseQVC):
    _gate = None

    def get_weights(self) -> Tuple[np.ndarray, float]:
        weights = 0.01 * self.random_state_.normal(
            size=(self.n_layers, self.n_qubits_, 1)
        )
        weights = np.array(weights, requires_grad=True)
        return weights

    def ansatz(
        self, weights: np.ndarray, wires: Union[List[int], range], n_layers: int
    ) -> None:
        if self._gate is None:
            raise NotImplementedError("Subclasses must implement this method.")
        for n_layer in range(n_layers):
            for wire in wires:
                self._gate(weights[n_layer, wire, 0], wires=wire)


class AnsatzSingleRotX(AnsatzSingleRot):
    _gate = qml.RX


class AnsatzSingleRotY(AnsatzSingleRot):
    _gate = qml.RY


class AnsatzSingleRotZ(AnsatzSingleRot):
    _gate = qml.RZ


class AnsatzRot(BaseQVC):
    def get_weights(self) -> Tuple[np.ndarray, float]:
        weights = 0.01 * self.random_state_.normal(
            size=(self.n_layers, self.n_qubits_, 3)
        )
        weights = np.array(weights, requires_grad=True)
        return weights

    def ansatz(
        self, weights: np.ndarray, wires: Union[List[int], range], n_layers: int
    ) -> None:
        for n_layer in range(n_layers):
            for wire in wires:
                qml.Rot(
                    weights[n_layer, wire, 0],
                    weights[n_layer, wire, 1],
                    weights[n_layer, wire, 2],
                    wires=wire,
                )


class AnsatzRotCNOT(BaseQVC):
    def get_weights(self) -> Tuple[np.ndarray, float]:
        weights = 0.01 * self.random_state_.normal(
            size=(self.n_layers, self.n_qubits_, 3)
        )
        weights = np.array(weights, requires_grad=True)
        return weights

    def ansatz(
        self, weights: np.ndarray, wires: Union[List[int], range], n_layers: int
    ) -> None:
        qml.StronglyEntanglingLayers(weights, wires=wires)


models = [
    AnsatzSingleRotX,
    AnsatzSingleRotY,
    AnsatzSingleRotZ,
    AnsatzRot,
    AnsatzRotCNOT,
]
