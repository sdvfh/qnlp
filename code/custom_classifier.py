import operator
from functools import reduce
from typing import Any, Dict, List, Union

import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import ClassifierTags
from sklearn.utils.estimator_checks import check_estimator
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


class QVC(ClassifierMixin, BaseEstimator):
    multi_class = False
    _estimator_type = "classifier"

    def __init__(self, n_layers=1, max_iter=10, batch_size=32, random_state=None):
        self.n_layers = n_layers
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.random_state = random_state

    def fit(self, X, y):
        # Check that X and y have correct shape, set n_features_in_, etc.
        X = check_array(X)
        y = self.transform_y(y)
        X, y = validate_data(self, X, y)

        self.n_qubits_ = self.get_n_qubits(X)
        self.device_ = qml.device("default.qubit", wires=self.n_qubits_)

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        self.random_state_ = check_random_state(self.random_state)

        opt = NesterovMomentumOptimizer(0.01)

        weights_init = 0.01 * self.random_state_.normal(
            size=(self.n_layers, self.n_qubits_, 3)
        )
        weights_init = np.array(weights_init, requires_grad=True)
        bias_init = np.array(0.0, requires_grad=True)

        self.weights_ = weights_init
        self.bias_ = bias_init

        len_train = X.shape[0]

        for n_iter in range(1, self.max_iter + 1):
            self.n_iter_ = n_iter
            indices = np.arange(len_train)

            self.random_state_.shuffle(indices)

            for start_idx in range(0, len_train, self.batch_size):
                end_idx = start_idx + self.batch_size
                batch_indices = indices[start_idx:end_idx]

                x_batch = np.array(X[batch_indices], requires_grad=False, dtype=float)
                y_batch = np.array(y[batch_indices], requires_grad=False)

                self.weights_, self.bias_, _, _ = opt.step(
                    self.cost, self.weights_, self.bias_, x_batch, y_batch
                )

            print(
                "Epoch: {:5d} | Training Cost: {:0.7f}".format(
                    n_iter, self.cost(self.weights_, self.bias_, X, y)
                )
            )

        # Return the classifier
        return self

    def transform_y(self, y_true):
        unique_labels = np.sort(np.unique(y_true))

        mapping = {int(unique_labels[0]): -1, int(unique_labels[1]): 1}
        self.inverse_mapping_ = {-1: unique_labels[0], 1: unique_labels[1]}

        # Aplicar mapeamento
        y_mapped = np.array([mapping[int(y)] for y in y_true])

        return y_mapped

    def inverse_transform_y(self, y_mapped):
        y_mapped = np.array([self.inverse_mapping_[int(y)] for y in y_mapped])
        return y_mapped

    def get_n_qubits(self, X):
        return np.ceil(np.log2(X.shape[1])).astype(int)

    def predict(self, X):
        check_is_fitted(self, ["weights_", "bias_"])
        X = validate_data(self, X, reset=False)
        X = np.array(X, requires_grad=False, dtype=float)
        output = np.sign(self.variational_classifier(self.weights_, self.bias_, X))
        output = self.inverse_transform_y(output)
        return output

    def predict_proba(self, X):
        check_is_fitted(self, ["weights_", "bias_"])
        X = validate_data(self, X, reset=False)
        X = np.array(X, requires_grad=False, dtype=float)
        output = self.variational_classifier(self.weights_, self.bias_, X)
        output = (1 + output) / 2
        return np.row_stack([1 - output, output])

    def variational_classifier(
        self, weights: np.ndarray, bias: float, x: np.ndarray
    ) -> np.ndarray:
        """
        Computes the output of the variational quantum classifier in a vectorized manner.

        Args:
            weights (np.ndarray): The weights for the quantum gates.
            bias (float): The bias term to be added to the circuit's output.
            x (np.ndarray): The input features (or batch of features) to be embedded into the quantum circuit.

        Returns:
            np.ndarray: The outputs of the variational quantum classifier.
        """

        @qml.qnode(self.device_, interface="autograd")
        def quantum_circuit(weights: np.ndarray, x: np.ndarray) -> float:
            """
            Implements the quantum circuit for variational classification.

            Args:
                weights (np.ndarray): The weights for the quantum gates.
                x (np.ndarray): The input features to be embedded into the quantum circuit.

            Returns:
                float: The expectation value of the PauliZ operator applied to the circuit.
            """
            wires = range(self.n_qubits_)
            qml.AmplitudeEmbedding(
                features=x, wires=wires, normalize=True, pad_with=0.0
            )
            qml.StronglyEntanglingLayers(weights, wires=wires)
            # Combine the PauliZ observables for all wires
            result = reduce(operator.matmul, [qml.PauliZ(i) for i in wires])
            return qml.expval(result)

        return quantum_circuit(weights, x) + bias

    @staticmethod
    def square_loss(
        labels: Union[np.ndarray, List[float]],
        predictions: Union[np.ndarray, List[float]],
    ) -> float:
        """
        Calculates the mean squared error loss between the labels and predictions.

        Args:
            labels (np.ndarray or List[float]): The true labels.
            predictions (np.ndarray or List[float]): The predicted values.

        Returns:
            float: The mean squared error loss.
        """
        labels = np.array(labels)
        predictions = np.array(predictions)
        return np.mean((labels - predictions) ** 2)

    @staticmethod
    def log_loss(
        labels: Union[np.ndarray, List[int]],
        predictions: Union[np.ndarray, List[float]],
    ) -> float:
        """
        Calculates the log loss (cross-entropy loss) between the true labels and the predicted probabilities.

        Args:
            labels (np.ndarray or List[int]): The true labels (e.g., 0 or 1 for binary classification).
            predictions (np.ndarray or List[float]): The predicted probabilities for the positive class.

        Returns:
            float: The log loss.
        """
        labels = np.array(labels)
        predictions = np.array(predictions)
        # To avoid log(0), clip predictions to a small value
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        return -np.mean(
            labels * np.log(predictions) + (1 - labels) * np.log(1 - predictions)
        )

    def cost(
        self, weights: np.ndarray, bias: float, x_batch: np.ndarray, y_batch: np.ndarray
    ) -> float:
        """
        Calculates the cost (mean squared error loss) for a batch of data using the variational quantum classifier.

        Args:
            weights (np.ndarray): The weights for the quantum gates.
            bias (float): The bias term to be added to the circuit's output.
            x_batch (np.ndarray): A batch of input features for the quantum circuit.
            y_batch (np.ndarray): The true labels for the batch.

        Returns:
            float: The mean squared error loss for the batch.
        """
        predictions = self.variational_classifier(weights, bias, x_batch)
        # return self.log_loss(y_batch, predictions)
        return self.square_loss(y_batch, predictions)

    @staticmethod
    def accuracy(
        labels: Union[np.ndarray, List[float]],
        predictions: Union[np.ndarray, List[float]],
    ) -> float:
        """
        Calculates the accuracy of the predictions compared to the true labels.

        Args:
            labels (np.ndarray or List[float]): The true labels.
            predictions (np.ndarray or List[float]): The predicted values.

        Returns:
            float: The accuracy expressed as the fraction of correct predictions.
        """
        labels = np.array(labels)
        predictions = np.array(predictions)
        return float(np.mean(np.abs(labels - predictions) < 1e-5))
