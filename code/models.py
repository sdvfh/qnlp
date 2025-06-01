import json
import operator
import os
import pickle
import uuid
from functools import reduce

import pennylane as qml
from pennylane import NesterovMomentumOptimizer
from pennylane import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import (
    check_array,
    check_is_fitted,
    check_random_state,
    validate_data,
)

import wandb


def create_and_log_artifact(name, data, artifact_type="data"):
    unique_filename = f"{name}_{uuid.uuid4().hex}.json"
    with open(unique_filename, "w") as f:
        json.dump(data, f)
    artifact = wandb.Artifact(name, type=artifact_type)
    artifact.add_file(unique_filename)
    wandb.log_artifact(artifact)
    os.remove(unique_filename)


class BaseQVC(ClassifierMixin, BaseEstimator):
    multi_class = False
    _estimator_type = "classifier"

    def __init__(
        self,
        n_layers=1,
        max_iter=10,
        batch_size=32,
        random_state=None,
        n_qubits_=None,
    ):
        self.n_layers = n_layers
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.random_state = random_state
        self.n_qubits_ = n_qubits_

    def fit(
        self,
        X,
        y,
        testing,
        sample_weight=None,
    ):
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

        self.classes_ = unique_labels(y)
        self.random_state_ = check_random_state(self.random_state)

        opt = NesterovMomentumOptimizer(0.01)

        self.weights_ = self.get_weights()
        self.bias_ = np.array(0.0, requires_grad=True)

        train_cost = self.compute_train_cost(X, y, sample_weight)

        self.loss_curve_ = [train_cost]
        if not testing:
            wandb.log({"loss": float(train_cost)})
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

                if sample_weight is not None:
                    sw_batch = np.array(
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
            if not testing:
                wandb.log({"loss": float(train_cost)})
            print(f"Epoch: {n_iter:5d} | Training Cost: {train_cost:0.7f}")

        return self

    def compute_train_cost(
        self,
        X,
        y,
        sample_weight=None,
    ):
        if sample_weight is not None:
            train_cost = self.cost(self.weights_, self.bias_, X, y, sample_weight)
        else:
            train_cost = self.cost(self.weights_, self.bias_, X, y)
        return train_cost

    def transform_y(self, y_true):
        unique_labels_arr = np.sort(np.unique(y_true))
        mapping = {
            int(unique_labels_arr[0]): -1,
            int(unique_labels_arr[1]): 1,
        }
        self.inverse_mapping_ = {-1: unique_labels_arr[0], 1: unique_labels_arr[1]}
        y_mapped = np.array([mapping[int(y)] for y in y_true])
        return y_mapped

    def inverse_transform_y(self, y_mapped):
        y_original = np.array([self.inverse_mapping_[int(y)] for y in y_mapped])
        return y_original

    def get_n_qubits(self, X):
        return int(np.ceil(np.log2(X.shape[1])))

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
        output = (1 + output) / 2  # Normalize output to [0, 1]
        return np.column_stack([1 - output, output])

    def variational_classifier(self, weights, bias: float, x):
        @qml.qnode(self.device_, interface="autograd")
        def quantum_circuit(weights, x):
            wires = range(self.n_qubits_)
            qml.AmplitudeEmbedding(
                features=x, wires=wires, normalize=True, pad_with=0.0
            )
            self.ansatz(weights, self.n_layers)
            observable = reduce(operator.matmul, [qml.PauliZ(i) for i in wires])
            return qml.expval(observable)

        return quantum_circuit(weights, x) + bias

    @staticmethod
    def square_loss(
        labels,
        predictions,
        sample_weight=None,
    ):
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
        labels,
        predictions,
    ):
        labels = np.array(labels)
        predictions = np.array(predictions)
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        return -np.mean(
            labels * np.log(predictions) + (1 - labels) * np.log(1 - predictions)
        )

    def cost(
        self,
        weights,
        bias,
        x_batch,
        y_batch,
        sample_weight=None,
    ):
        predictions = self.variational_classifier(weights, bias, x_batch)
        return self.square_loss(y_batch, predictions, sample_weight)

    @staticmethod
    def accuracy(
        labels,
        predictions,
    ):
        labels = np.array(labels)
        predictions = np.array(predictions)
        return float(np.mean(np.abs(labels - predictions) < 1e-5))

    def get_weights(self):
        weights = 0.01 * self.random_state_.normal(size=self.get_weights_size())
        return np.array(weights, requires_grad=True)

    def get_weights_size(self):
        raise NotImplementedError(
            "Subclasses must implement the get_weights_size method."
        )

    def ansatz(self, weights, n_layers):
        raise NotImplementedError("Subclasses must implement the ansatz method.")

    def save(self, y_pred, model_has_proba=True):
        if model_has_proba:
            create_and_log_artifact("y_pred", y_pred[:, 1].tolist(), "y_pred.json")
        else:
            create_and_log_artifact("y_pred", y_pred.tolist(), "y_pred.json")
        create_and_log_artifact("weights", self.weights_.tolist(), "weights.json")
        create_and_log_artifact("biases", {"bias": float(self.bias_)}, "biases.json")


class AnsatzSingleRot(BaseQVC):
    _gate = None

    def get_weights_size(self):
        return self.n_layers, self.n_qubits_, 1

    def ansatz(self, weights, n_layers):
        if self._gate is None:
            raise NotImplementedError(
                "Subclasses must define the rotation gate (_gate)."
            )
        for n_layer in range(n_layers):
            for wire in range(self.n_qubits_):
                self._gate(weights[n_layer, wire, 0], wires=wire)


class AnsatzSingleRotX(AnsatzSingleRot):
    _gate = qml.RX


class AnsatzSingleRotY(AnsatzSingleRot):
    _gate = qml.RY


class AnsatzSingleRotZ(AnsatzSingleRot):
    _gate = qml.RZ


class AnsatzRot(BaseQVC):
    def get_weights_size(self):
        return self.n_layers, self.n_qubits_, 3

    def ansatz(self, weights, n_layers):
        for n_layer in range(n_layers):
            for wire in range(self.n_qubits_):
                qml.Rot(
                    weights[n_layer, wire, 0],
                    weights[n_layer, wire, 1],
                    weights[n_layer, wire, 2],
                    wires=wire,
                )


class AnsatzRotCNOT(BaseQVC):
    def get_weights_size(self):
        return self.n_layers, self.n_qubits_, 3

    def ansatz(self, weights, n_layers):
        qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits_))


class AnsatzRotCNOT2(BaseQVC):
    def get_weights_size(self):
        return self.n_layers, self.n_qubits_, 3

    def ansatz(self, weights, n_layers):
        for n_layer in range(n_layers):
            for wire in range(self.n_qubits_):
                qml.Rot(
                    weights[n_layer, wire, 0],
                    weights[n_layer, wire, 1],
                    weights[n_layer, wire, 2],
                    wires=wire,
                )
                qml.CNOT([0, 1])
                qml.CNOT([1, 2])
                qml.CNOT([2, 3])
                qml.CNOT([3, 0])


class AnsatzMaouakiBase(BaseQVC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.n_qubits_ != 4:
            raise NotImplementedError(
                "This ansatz only works for circuit with 4 qubits."
            )


class AnsatzMaouaki15(BaseQVC):
    def get_weights_size(self):
        return self.n_layers, 8

    def ansatz(self, weights, n_layers):
        for n_layer in range(n_layers):
            qml.RY(weights[n_layer, 0], wires=0)
            qml.RY(weights[n_layer, 1], wires=1)
            qml.RY(weights[n_layer, 2], wires=2)
            qml.RY(weights[n_layer, 3], wires=3)

            qml.CNOT(wires=[3, 0])
            qml.CNOT(wires=[2, 3])

            qml.CNOT(wires=[1, 2])
            qml.RY(weights[n_layer, 4], wires=3)

            qml.CNOT(wires=[0, 1])
            qml.RY(weights[n_layer, 5], wires=2)

            qml.CNOT(wires=[3, 2])
            qml.RY(weights[n_layer, 6], wires=1)
            qml.RY(weights[n_layer, 7], wires=0)

            qml.CNOT(wires=[0, 3])

            qml.CNOT(wires=[1, 0])

            qml.CNOT(wires=[2, 1])


class AnsatzMaouaki1(BaseQVC):
    def get_weights_size(self):
        return self.n_layers, 8

    def ansatz(self, weights, n_layers):
        for n_layer in range(n_layers):
            qml.RX(weights[n_layer, 0], wires=0)
            qml.RX(weights[n_layer, 1], wires=1)
            qml.RX(weights[n_layer, 2], wires=2)
            qml.RX(weights[n_layer, 3], wires=3)

            qml.RZ(weights[n_layer, 4], wires=0)
            qml.RZ(weights[n_layer, 5], wires=1)
            qml.RZ(weights[n_layer, 6], wires=2)
            qml.RZ(weights[n_layer, 7], wires=3)


class AnsatzMaouaki9(BaseQVC):
    def get_weights_size(self):
        return self.n_layers, 4

    def ansatz(self, weights, n_layers):
        for n_layer in range(n_layers):
            qml.H(wires=0)
            qml.H(wires=1)
            qml.H(wires=2)
            qml.H(wires=3)

            qml.CZ(wires=[2, 3])

            qml.CZ(wires=[1, 2])
            qml.RX(weights[n_layer, 0], wires=3)

            qml.CZ(wires=[0, 1])
            qml.RX(weights[n_layer, 1], wires=2)

            qml.RX(weights[n_layer, 2], wires=0)
            qml.RX(weights[n_layer, 3], wires=1)


class AnsatzEnt1(BaseQVC):
    def get_weights_size(self):
        return self.n_layers, 4

    def ansatz(self, weights, n_layers):
        for n_layer in range(n_layers):
            qml.H(wires=0)
            qml.H(wires=1)
            qml.H(wires=2)
            qml.H(wires=3)

            qml.RX(weights[n_layer, 0], wires=3)
            qml.RX(weights[n_layer, 1], wires=2)
            qml.RX(weights[n_layer, 2], wires=0)
            qml.RX(weights[n_layer, 3], wires=1)


class AnsatzEnt2(BaseQVC):
    def get_weights_size(self):
        return self.n_layers, 4

    def ansatz(self, weights, n_layers):
        for n_layer in range(n_layers):
            qml.H(wires=0)
            qml.H(wires=1)
            qml.H(wires=2)
            qml.H(wires=3)

            qml.CNOT(wires=[2, 3])

            qml.CNOT(wires=[1, 2])
            qml.RX(weights[n_layer, 0], wires=3)

            qml.CNOT(wires=[0, 1])
            qml.RX(weights[n_layer, 1], wires=2)

            qml.RX(weights[n_layer, 2], wires=0)
            qml.RX(weights[n_layer, 3], wires=1)


class AnsatzEnt22(BaseQVC):
    def get_weights_size(self):
        return self.n_layers, 4

    def ansatz(self, weights, n_layers):
        for n_layer in range(n_layers):
            qml.H(wires=0)
            qml.H(wires=1)
            qml.H(wires=2)
            qml.H(wires=3)

            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            qml.CNOT(wires=[2, 3])
            qml.CNOT(wires=[3, 0])

            qml.RX(weights[n_layer, 0], wires=0)
            qml.RX(weights[n_layer, 1], wires=1)
            qml.RX(weights[n_layer, 2], wires=2)
            qml.RX(weights[n_layer, 3], wires=3)


class AnsatzEnt3(BaseQVC):
    def get_weights_size(self):
        return self.n_layers, 12

    def ansatz(self, weights, n_layers):
        for n_layer in range(n_layers):
            qml.H(wires=0)
            qml.H(wires=1)
            qml.H(wires=2)
            qml.H(wires=3)

            qml.CNOT(wires=[2, 3])

            qml.CNOT(wires=[1, 2])
            qml.Rot(
                weights[n_layer, 0], weights[n_layer, 1], weights[n_layer, 2], wires=3
            )

            qml.CNOT(wires=[0, 1])
            qml.Rot(
                weights[n_layer, 3], weights[n_layer, 4], weights[n_layer, 5], wires=2
            )

            qml.Rot(
                weights[n_layer, 6], weights[n_layer, 7], weights[n_layer, 8], wires=0
            )
            qml.Rot(
                weights[n_layer, 9], weights[n_layer, 10], weights[n_layer, 11], wires=1
            )


class AnsatzEnt32(BaseQVC):
    def get_weights_size(self):
        return self.n_layers, 12

    def ansatz(self, weights, n_layers):
        for n_layer in range(n_layers):
            qml.H(wires=0)
            qml.H(wires=1)
            qml.H(wires=2)
            qml.H(wires=3)

            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            qml.CNOT(wires=[2, 3])
            qml.CNOT(wires=[3, 0])

            qml.Rot(
                weights[n_layer, 0], weights[n_layer, 1], weights[n_layer, 2], wires=0
            )
            qml.Rot(
                weights[n_layer, 3], weights[n_layer, 4], weights[n_layer, 5], wires=1
            )
            qml.Rot(
                weights[n_layer, 6], weights[n_layer, 7], weights[n_layer, 8], wires=2
            )
            qml.Rot(
                weights[n_layer, 9], weights[n_layer, 10], weights[n_layer, 11], wires=3
            )


class AnsatzEnt4(BaseQVC):
    def get_weights_size(self):
        return self.n_layers, 12

    def ansatz(self, weights, n_layers):
        for n_layer in range(n_layers):
            qml.H(wires=0)
            qml.H(wires=1)
            qml.H(wires=2)
            qml.H(wires=3)

            qml.CZ(wires=[2, 3])

            qml.CZ(wires=[1, 2])
            qml.Rot(
                weights[n_layer, 0], weights[n_layer, 1], weights[n_layer, 2], wires=3
            )

            qml.CZ(wires=[0, 1])
            qml.Rot(
                weights[n_layer, 3], weights[n_layer, 4], weights[n_layer, 5], wires=2
            )

            qml.Rot(
                weights[n_layer, 6], weights[n_layer, 7], weights[n_layer, 8], wires=0
            )
            qml.Rot(
                weights[n_layer, 9], weights[n_layer, 10], weights[n_layer, 11], wires=1
            )


class AnsatzMaouaki7(BaseQVC):
    def get_weights_size(self):
        return self.n_layers, 19

    def ansatz(self, weights, n_layers):
        for n_layer in range(n_layers):
            qml.RX(weights[n_layer, 0], wires=0)
            qml.RX(weights[n_layer, 1], wires=1)
            qml.RX(weights[n_layer, 2], wires=2)
            qml.RX(weights[n_layer, 3], wires=3)

            qml.RZ(weights[n_layer, 4], wires=0)
            qml.RZ(weights[n_layer, 5], wires=1)
            qml.RZ(weights[n_layer, 6], wires=2)
            qml.RZ(weights[n_layer, 7], wires=3)

            qml.CRZ(weights[n_layer, 8], wires=[1, 0])
            qml.CRZ(weights[n_layer, 9], wires=[3, 2])

            qml.RX(weights[n_layer, 10], wires=0)
            qml.RX(weights[n_layer, 11], wires=1)
            qml.RX(weights[n_layer, 12], wires=2)
            qml.RX(weights[n_layer, 13], wires=3)

            qml.RZ(weights[n_layer, 14], wires=0)
            qml.RZ(weights[n_layer, 15], wires=1)
            qml.RZ(weights[n_layer, 16], wires=2)
            qml.RZ(weights[n_layer, 17], wires=3)

            qml.CRZ(weights[n_layer, 18], wires=[2, 1])


class AnsatzMaouaki11(BaseQVC):
    def get_weights_size(self):
        return self.n_layers, 12

    def ansatz(self, weights, n_layers):
        for n_layer in range(n_layers):
            qml.RX(weights[n_layer, 0], wires=0)
            qml.RX(weights[n_layer, 1], wires=1)
            qml.RX(weights[n_layer, 2], wires=2)
            qml.RX(weights[n_layer, 3], wires=3)

            qml.RZ(weights[n_layer, 4], wires=0)
            qml.RZ(weights[n_layer, 5], wires=1)
            qml.RZ(weights[n_layer, 6], wires=2)
            qml.RZ(weights[n_layer, 7], wires=3)

            qml.CNOT(wires=[1, 0])
            qml.CNOT(wires=[3, 2])

            qml.RX(weights[n_layer, 8], wires=1)
            qml.RX(weights[n_layer, 9], wires=2)

            qml.RZ(weights[n_layer, 10], wires=1)
            qml.RZ(weights[n_layer, 11], wires=2)

            qml.CNOT(wires=[2, 1])


class AnsatzMaouaki6(BaseQVC):
    def get_weights_size(self):
        return self.n_layers, 28

    def ansatz(self, weights, n_layers):
        for n_layer in range(n_layers):
            qml.RX(weights[n_layer, 0], wires=0)
            qml.RX(weights[n_layer, 1], wires=1)
            qml.RX(weights[n_layer, 2], wires=2)
            qml.RX(weights[n_layer, 3], wires=3)

            qml.RZ(weights[n_layer, 4], wires=0)
            qml.RZ(weights[n_layer, 5], wires=1)
            qml.RZ(weights[n_layer, 6], wires=2)
            qml.RZ(weights[n_layer, 7], wires=3)

            qml.CRX(weights[n_layer, 8], wires=[3, 2])
            qml.CRX(weights[n_layer, 9], wires=[3, 1])
            qml.CRX(weights[n_layer, 10], wires=[3, 0])

            qml.CRX(weights[n_layer, 11], wires=[2, 3])
            qml.CRX(weights[n_layer, 12], wires=[2, 1])
            qml.CRX(weights[n_layer, 13], wires=[2, 0])

            qml.CRX(weights[n_layer, 14], wires=[1, 3])
            qml.CRX(weights[n_layer, 15], wires=[1, 2])
            qml.CRX(weights[n_layer, 16], wires=[1, 0])

            qml.CRX(weights[n_layer, 17], wires=[0, 3])
            qml.CRX(weights[n_layer, 18], wires=[0, 2])
            qml.CRX(weights[n_layer, 19], wires=[0, 1])

            qml.RX(weights[n_layer, 20], wires=0)
            qml.RX(weights[n_layer, 21], wires=1)
            qml.RX(weights[n_layer, 22], wires=2)
            qml.RX(weights[n_layer, 23], wires=3)

            qml.RZ(weights[n_layer, 24], wires=0)
            qml.RZ(weights[n_layer, 25], wires=1)
            qml.RZ(weights[n_layer, 26], wires=2)
            qml.RZ(weights[n_layer, 27], wires=3)


class ScikitBase:
    _model_template = None
    _parameters_template = {}
    _parameters = {}

    def __init__(self, *args, **kwargs):
        self._parameters = {
            "random_state": kwargs.get("random_state"),
            **self._parameters_template,
        }
        self._model = self._model_template(**self._parameters)

    def fit(self, X, y, testing, sample_weight=None):
        self._model.fit(X, y, sample_weight=sample_weight)

    def predict_proba(self, X):
        return self._model.predict_proba(X)

    def predict(self, X):
        return self._model.predict(X)

    def save(self, y_pred, model_has_proba=True):
        if model_has_proba:
            create_and_log_artifact("y_pred", y_pred[:, 1].tolist(), "y_pred.json")
        else:
            create_and_log_artifact("y_pred", y_pred.tolist(), "y_pred.json")
        unique_filename = f"{self.__class__.__name__}_{uuid.uuid4().hex}.pkl"
        with open(unique_filename, "wb") as f:
            pickle.dump(self._model, f)
        artifact = wandb.Artifact(
            self.__class__.__name__, type="model", metadata=self._parameters
        )
        artifact.add_file(unique_filename)
        wandb.log_artifact(artifact)
        os.remove(unique_filename)


class SVMBase(ScikitBase):
    _model_template = SVC


class SVMRBF(SVMBase):
    _parameters_template = {"kernel": "rbf", "probability": True, "verbose": True}


class SVMLinear(SVMBase):
    _parameters_template = {"kernel": "linear", "probability": True, "verbose": True}


class SVMPoly(SVMBase):
    _parameters_template = {"kernel": "poly", "probability": True, "verbose": True}


class MyLogisticRegression(ScikitBase):
    _model_template = LogisticRegression
    _parameters_template = {"max_iter": 1_000_000, "verbose": 5}


class RandomForest(ScikitBase):
    _model_template = RandomForestClassifier
    _parameters_template = {"verbose": 5}


class KNN(ScikitBase):
    _model_template = KNeighborsClassifier

    def __init__(self, *args, **kwargs):
        self._model = self._model_template()

    def fit(self, X, y, testing, sample_weight=None):
        self._model.fit(X, y)


class MLP(ScikitBase):
    _model_template = MLPClassifier
    _parameters_template = {"verbose": True, "hidden_layer_sizes": ()}

    def fit(self, X, y, testing, sample_weight=None):
        self._model.fit(X, y)


class AdaBoostRotCNOT(ScikitBase):
    def __init__(self, *args, **kwargs):
        qvc = AnsatzRotCNOT(*args, **kwargs)
        self._model = AdaBoostClassifier(
            estimator=qvc, n_estimators=100, random_state=kwargs.get("random_state")
        )


class BaggingRotCNOT(ScikitBase):
    def __init__(self, *args, **kwargs):
        qvc = AnsatzRotCNOT(*args, **kwargs)
        self._model = BaggingClassifier(
            estimator=qvc,
            n_estimators=100,
            random_state=kwargs.get("random_state"),
            bootstrap_features=True,
        )


class VotingQVC(ScikitBase):
    voting_method = None

    def __init__(self, *args, **kwargs):
        qvcs = [
            ("single_rot_x", AnsatzSingleRotX(*args, **kwargs)),
            ("single_rot_y", AnsatzSingleRotY(*args, **kwargs)),
            ("single_rot_z", AnsatzSingleRotZ(*args, **kwargs)),
            ("rot", AnsatzRot(*args, **kwargs)),
            ("rotcnot", AnsatzRotCNOT(*args, **kwargs)),
        ]
        self._model = VotingClassifier(estimators=qvcs, voting=self.voting_method)


class SoftVotingQVC(VotingQVC):
    voting_method = "soft"


class HardVotingQVC(VotingQVC):
    voting_method = "hard"


class VotingQVCMaouaki(ScikitBase):
    voting_method = None

    def __init__(self, *args, **kwargs):
        qvcs = [
            ("maouaki1", AnsatzMaouaki1(*args, **kwargs)),
            ("maouaki7", AnsatzMaouaki7(*args, **kwargs)),
            ("maouaki9", AnsatzMaouaki9(*args, **kwargs)),
            ("maouaki11", AnsatzMaouaki11(*args, **kwargs)),
            ("maouaki15", AnsatzMaouaki15(*args, **kwargs)),
        ]
        self._model = VotingClassifier(estimators=qvcs, voting=self.voting_method)


class SoftVotingQVCMaouaki(VotingQVCMaouaki):
    voting_method = "soft"


class HardVotingQVCMaouaki(VotingQVCMaouaki):
    voting_method = "hard"


class VotingQVCAll(ScikitBase):
    voting_method = None

    def __init__(self, *args, **kwargs):
        qvcs = [
            ("maouaki1", AnsatzMaouaki1(*args, **kwargs)),
            ("maouaki7", AnsatzMaouaki7(*args, **kwargs)),
            ("maouaki9", AnsatzMaouaki9(*args, **kwargs)),
            ("maouaki11", AnsatzMaouaki11(*args, **kwargs)),
            ("maouaki15", AnsatzMaouaki15(*args, **kwargs)),
            ("single_rot_x", AnsatzSingleRotX(*args, **kwargs)),
            ("single_rot_y", AnsatzSingleRotY(*args, **kwargs)),
            ("single_rot_z", AnsatzSingleRotZ(*args, **kwargs)),
            ("rot", AnsatzRot(*args, **kwargs)),
            ("rotcnot", AnsatzRotCNOT(*args, **kwargs)),
        ]
        self._model = VotingClassifier(estimators=qvcs, voting=self.voting_method)


class SoftVotingQVCAll(VotingQVCAll):
    voting_method = "soft"


class HardVotingQVCAll(VotingQVCAll):
    voting_method = "hard"


class VotingClassic(ScikitBase):
    voting_method = None

    def __init__(self, *args, **kwargs):
        qvcs = [
            ("svmrbf", SVMRBF(*args, **kwargs)._model),
            ("svmlinear", SVMLinear(*args, **kwargs)._model),
            ("svmpoly", SVMPoly(*args, **kwargs)._model),
            ("logistic", MyLogisticRegression(*args, **kwargs)._model),
            ("randomforest", RandomForest(*args, **kwargs)._model),
            ("knn", KNN(*args, **kwargs)._model),
            ("mlp", MLP(*args, **kwargs)._model),
        ]
        self._model = VotingClassifier(estimators=qvcs, voting=self.voting_method)


class SoftVotingClassic(VotingClassic):
    voting_method = "soft"


class HardVotingClassic(VotingClassic):
    voting_method = "hard"


def get_model_classifier(args, seed):
    model_factory = {
        "singlerotx": AnsatzSingleRotX,
        "singleroty": AnsatzSingleRotY,
        "singlerotz": AnsatzSingleRotZ,
        "rot": AnsatzRot,
        "rotcnot": AnsatzRotCNOT,
        "rotcnot2": AnsatzRotCNOT2,
        "svmrbf": SVMRBF,
        "svmlinear": SVMLinear,
        "svmpoly": SVMPoly,
        "logistic": MyLogisticRegression,
        "randomforest": RandomForest,
        "knn": KNN,
        "mlp": MLP,
        "ensemble_adaboost_rotcnot": AdaBoostRotCNOT,
        "ensemble_bagging_rotcnot": BaggingRotCNOT,
        "ensemble_softvoting_qvc": SoftVotingQVC,
        "ensemble_hardvoting_qvc": HardVotingQVC,
        "maouaki1": AnsatzMaouaki1,
        "maouaki7": AnsatzMaouaki7,
        "maouaki9": AnsatzMaouaki9,
        "maouaki11": AnsatzMaouaki11,
        "maouaki15": AnsatzMaouaki15,
        "ensemble_softvoting_maouaki": SoftVotingQVCMaouaki,
        "ensemble_hardvoting_maouaki": HardVotingQVCMaouaki,
        "ensemble_softvoting_qvc_all": SoftVotingQVCAll,
        "ensemble_hardvoting_qvc_all": HardVotingQVCAll,
        "ensemble_softvoting_classic": SoftVotingClassic,
        "ensemble_hardvoting_classic": HardVotingClassic,
        "ent1": AnsatzEnt1,
        "ent2": AnsatzEnt2,
        "ent22": AnsatzEnt22,
        "ent3": AnsatzEnt3,
        "ent32": AnsatzEnt32,
        "ent4": AnsatzEnt4,
        "maouaki6": AnsatzMaouaki6,
    }
    model_name = args.model_classifier

    return model_factory[model_name](
        n_layers=args.n_layers,
        max_iter=args.epochs,
        batch_size=args.batch_size,
        random_state=seed,
        n_qubits_=args.n_qubits,
    )
