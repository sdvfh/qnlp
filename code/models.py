import json
import operator
import os
import pickle
import uuid
from functools import reduce

import pennylane as qml
from matplotlib import pyplot as plt
from pennylane import numpy as np
from scipy.special import rel_entr
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
    _id = None
    multi_class = False
    _estimator_type = "classifier"

    def __init__(
        self,
        n_layers=1,
        max_iter=10,
        batch_size=32,
        random_state=None,
        n_qubits_=None,
        testing=False,
    ):
        self.n_layers = n_layers
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.random_state = random_state
        self.n_qubits_ = n_qubits_
        self.testing = testing

    def fit(
        self,
        X,
        y,
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

        opt = qml.optimize.AdamOptimizer()

        self.weights_ = self.get_weights()
        self.bias_ = np.array(0.0, requires_grad=True)

        train_cost = self.compute_train_cost(X, y, sample_weight)

        self.loss_curve_ = [train_cost]
        if not self.testing:
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
            if not self.testing:
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

    def probability_haar(self, lower_value, upper_value, N):
        return self.primitive_p_haar(upper_value, N) - self.primitive_p_haar(
            lower_value, N
        )

    @staticmethod
    def primitive_p_haar(fidelity, N):
        return -((1 - fidelity) ** (N - 1))

    def circuit_measures(self, n_layers, with_state_preparation=False):
        wires = list(range(self.n_qubits_))
        dev = qml.device("default.qubit", wires=wires)
        zero_ket = [0] * (2**self.n_qubits_)
        zero_ket[0] = 1

        projector = qml.Projector(zero_ket, wires)

        if with_state_preparation:
            stateprep = AnsatzMottoten(
                n_layers=self.n_layers,
                max_iter=self.max_iter,
                batch_size=self.batch_size,
                random_state=self.random_state,
                n_qubits_=self.n_qubits_,
                testing=self.testing,
            )

            @qml.qnode(dev)
            def expr(params, params_conj):
                params, params_state = params
                params_conj, params_conj_state = params_conj
                stateprep.ansatz(params_state, n_layers)
                self.ansatz(params, n_layers)
                qml.adjoint(self.ansatz)(params_conj, n_layers)
                qml.adjoint(stateprep.ansatz)(params_conj_state, n_layers)
                return qml.expval(projector)

            @qml.qnode(dev)
            def ent(params):
                params, params_state = params
                stateprep.ansatz(params_state, n_layers)
                self.ansatz(params, n_layers)
                return qml.state()

        else:

            @qml.qnode(dev)
            def expr(params, params_conj):
                self.ansatz(params, n_layers)
                qml.adjoint(self.ansatz)(params_conj, n_layers)
                return qml.expval(projector)

            @qml.qnode(dev)
            def ent(params):
                self.ansatz(params, n_layers)
                return qml.state()

        return expr, ent

    def compute_entanglement(self, states):
        m = states.shape[0]
        psi_t = states.reshape((m,) + (2,) * self.n_qubits_)

        distances_sum = np.zeros(m)

        for j in range(self.n_qubits_):
            u = np.take(psi_t, 0, axis=1 + j).reshape(m, -1)
            v = np.take(psi_t, 1, axis=1 + j).reshape(m, -1)

            M1 = u[:, :, None] * v[:, None, :]
            M2 = v[:, :, None] * u[:, None, :]

            D_j = 0.5 * np.abs(M1 - M2) ** 2
            D_j = D_j.sum(axis=(1, 2))

            distances_sum += D_j

        ent_values = (4 / self.n_qubits_) * distances_sum
        return float(ent_values.mean())

    def compute_measures(self, n_bins, n, to_plot, with_state_prep=False):
        random_state = check_random_state(self.random_state)
        n_bins_list = [i / n_bins for i in range(n_bins + 1)]
        prob_dist_haar = [
            self.probability_haar(n_bins_list[i], n_bins_list[i + 1], 2**self.n_qubits_)
            for i in range(n_bins)
        ]
        params_shape = (n,) + self.get_weights_size()

        params, params_conj = random_state.uniform(
            0, 2 * np.pi, size=params_shape
        ), random_state.uniform(0, 2 * np.pi, size=params_shape)

        if with_state_prep:
            stateprep = AnsatzMottoten(
                n_layers=self.n_layers,
                max_iter=self.max_iter,
                batch_size=self.batch_size,
                random_state=self.random_state,
                n_qubits_=self.n_qubits_,
                testing=self.testing,
            )
            params_state_shape = (n,) + stateprep.get_weights_size()

            params_state, params_state_conj = random_state.uniform(
                0, 2 * np.pi, size=params_state_shape
            ), random_state.uniform(0, 2 * np.pi, size=params_state_shape)
            params = list(zip(params, params_state, strict=True))
            params_conj = list(zip(params_conj, params_state_conj, strict=True))

        circ_expr, circ_ent = self.circuit_measures(self.n_layers, with_state_prep)
        fidelities_ansatz = np.array(
            [circ_expr(params[i], params_conj[i]) for i in range(n)]
        )
        if np.ndim(fidelities_ansatz) == 0:
            fidelities_ansatz = np.full(n, fidelities_ansatz)

        prob_dist_ansatz, edges = np.histogram(
            fidelities_ansatz,
            bins=n_bins,
            range=(0, 1),
            density=False,
            weights=np.ones(n) / n,
        )

        div_kl = np.sum(rel_entr(prob_dist_ansatz, prob_dist_haar))

        if to_plot:
            widths = np.diff(edges)
            fig, ax = plt.subplots(dpi=300)
            ax.bar(
                edges[:-1],
                prob_dist_haar,
                width=widths,
                align="edge",
                label="Haar",
                edgecolor="white",
                linewidth=0.1,
            )

            ax.bar(
                edges[:-1],
                prob_dist_ansatz,
                width=widths,
                align="edge",
                label="A",
                edgecolor="white",
                alpha=0.5,
                linewidth=0.1,
            )

            ax.set_xlabel("Fidelity")
            ax.set_ylabel("Probability Density")
            ax.set_xlim(0, 1)
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend()
            plt.show()

        if self.n_qubits_ > 1:
            states = np.array([circ_ent(params[i]) for i in range(n)])
            ent = self.compute_entanglement(states)
        else:
            ent = 0
        return div_kl, ent


class AnsatzMottoten(BaseQVC):
    def get_weights_size(self):
        return 1, 1, int(2**self.n_qubits_)

    def ansatz(self, weights, n_layers):
        wires = range(self.n_qubits_)
        qml.AmplitudeEmbedding(
            features=weights[0, 0, :], wires=wires, normalize=True, pad_with=0.0
        )


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
    _id = 1


class AnsatzSingleRotY(AnsatzSingleRot):
    _gate = qml.RY
    _id = 2


class AnsatzSingleRotZ(AnsatzSingleRot):
    _gate = qml.RZ
    _id = 3


class AnsatzMaouaki1(BaseQVC):
    _id = 4

    def get_weights_size(self):
        return self.n_layers, self.n_qubits_, 2

    def ansatz(self, weights, n_layers):
        for n_layer in range(n_layers):
            for wire in range(self.n_qubits_):
                qml.RX(weights[n_layer, wire, 0], wires=wire)
                qml.RZ(weights[n_layer, wire, 1], wires=wire)


class AnsatzRot(BaseQVC):
    _id = 5

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
    _id = 6

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
            for wire in range(self.n_qubits_):
                qml.CNOT(wires=[wire, (wire + 1) % self.n_qubits_])


class AnsatzEnt1(BaseQVC):
    _id = 7

    def get_weights_size(self):
        return self.n_layers, self.n_qubits_, 1

    def ansatz(self, weights, n_layers):
        for n_layer in range(n_layers):
            for wire in range(self.n_qubits_):
                qml.H(wires=wire)
            for wire in range(self.n_qubits_):
                qml.RX(weights[n_layer, wire, 0], wires=wire)


class AnsatzMaouaki9(BaseQVC):
    _id = 8

    def get_weights_size(self):
        return self.n_layers, self.n_qubits_, 1

    def ansatz(self, weights, n_layers):
        for n_layer in range(n_layers):
            for wire in range(self.n_qubits_):
                qml.H(wires=wire)
            for wire in range(self.n_qubits_ - 1):
                qml.CZ(wires=[wire, wire + 1])
            for wire in range(self.n_qubits_):
                qml.RX(weights[n_layer, wire, 0], wires=wire)


class AnsatzEnt4(BaseQVC):
    _id = 9

    def get_weights_size(self):
        return self.n_layers, self.n_qubits_, 3

    def ansatz(self, weights, n_layers):
        for n_layer in range(n_layers):
            for wire in range(self.n_qubits_):
                qml.H(wires=wire)
            for wire in range(self.n_qubits_ - 1):
                qml.CZ(wires=[wire, wire + 1])
            for wire in range(self.n_qubits_):
                qml.Rot(
                    weights[n_layer, wire, 0],
                    weights[n_layer, wire, 1],
                    weights[n_layer, wire, 2],
                    wires=wire,
                )


class AnsatzEnt2(BaseQVC):
    _id = 10

    def get_weights_size(self):
        return self.n_layers, self.n_qubits_, 1

    def ansatz(self, weights, n_layers):
        for n_layer in range(n_layers):
            for wire in range(self.n_qubits_):
                qml.H(wires=wire)
            for wire in range(self.n_qubits_):
                qml.CNOT(wires=[wire, (wire + 1) % self.n_qubits_])
            for wire in range(self.n_qubits_):
                qml.RX(weights[n_layer, wire, 0], wires=wire)


class AnsatzEnt3(BaseQVC):
    _id = 11

    def get_weights_size(self):
        return self.n_layers, self.n_qubits_, 3

    def ansatz(self, weights, n_layers):
        for n_layer in range(n_layers):
            for wire in range(self.n_qubits_):
                qml.H(wires=wire)
            for wire in range(self.n_qubits_):
                qml.CNOT(wires=[wire, (wire + 1) % self.n_qubits_])
            for wire in range(self.n_qubits_):
                qml.Rot(
                    weights[n_layer, wire, 0],
                    weights[n_layer, wire, 1],
                    weights[n_layer, wire, 2],
                    wires=wire,
                )


class AnsatzMaouakiQuasi7(BaseQVC):
    _id = 12

    def get_weights_size(self):
        return (
            self.n_layers,
            self.n_qubits_,
            self.n_qubits_ + 1,
        )

    def ansatz(self, weights, n_layers):
        for n_layer in range(n_layers):
            for wire in range(self.n_qubits_):
                qml.RX(weights[n_layer, wire, 0], wires=wire)
                qml.RZ(weights[n_layer, wire, 1], wires=wire)

            ent_idx = -1
            for wire in range(0, self.n_qubits_ - 1, 2):
                qml.CRZ(weights[n_layer, wire, ent_idx], wires=[wire, wire + 1])
                ent_idx -= 1

            for wire in range(1, self.n_qubits_ - 1, 2):
                qml.CRZ(weights[n_layer, wire, ent_idx], wires=[wire, wire + 1])
                ent_idx -= 1


class AnsatzMaouaki7(BaseQVC):
    _id = 13

    def get_weights_size(self):
        return (
            self.n_layers,
            self.n_qubits_,
            self.n_qubits_ + 3,
        )

    def ansatz(self, weights, n_layers):
        for n_layer in range(n_layers):
            for wire in range(self.n_qubits_):
                qml.RX(weights[n_layer, wire, 0], wires=wire)
                qml.RZ(weights[n_layer, wire, 1], wires=wire)

            ent_idx = -1
            for wire in range(0, self.n_qubits_ - 1, 2):
                qml.CRZ(weights[n_layer, wire, ent_idx], wires=[wire, wire + 1])
                ent_idx -= 1

            for wire in range(self.n_qubits_):
                qml.RX(weights[n_layer, wire, 2], wires=wire)
                qml.RZ(weights[n_layer, wire, 3], wires=wire)

            for wire in range(1, self.n_qubits_ - 1, 2):
                qml.CRZ(weights[n_layer, wire, ent_idx], wires=[wire, wire + 1])
                ent_idx -= 1


class AnsatzMaouaki15(BaseQVC):
    _id = 14

    def get_weights_size(self):
        return self.n_layers, self.n_qubits_, 2

    def ansatz(self, weights, n_layers):
        for n_layer in range(n_layers):
            for wire in range(self.n_qubits_):
                qml.RY(weights[n_layer, wire, 0], wires=wire)
            for wire in range(self.n_qubits_):
                qml.CNOT(wires=[wire, (wire + 1) % self.n_qubits_])
            for wire in range(self.n_qubits_):
                qml.RY(weights[n_layer, wire, 1], wires=wire)
            for wire in range(self.n_qubits_):
                qml.CNOT(wires=[wire, (wire + 3) % self.n_qubits_])


class AnsatzMaouaki6(BaseQVC):
    _id = 15

    def get_weights_size(self):
        return self.n_layers, self.n_qubits_, self.n_qubits_ + 3

    def ansatz(self, weights, n_layers):
        for n_layer in range(n_layers):
            for wire in range(self.n_qubits_):
                qml.RX(weights[n_layer, wire, 0], wires=wire)
                qml.RZ(weights[n_layer, wire, 1], wires=wire)
            for wire in range(self.n_qubits_):
                qml.CNOT(wires=[wire, (wire + 1) % self.n_qubits_])
            for wire_ctrl in range(self.n_qubits_):
                ent_idx = 2
                for wire_targ in range(self.n_qubits_):
                    if wire_ctrl == wire_targ:
                        continue
                    angle_crx = weights[n_layer, wire_ctrl, ent_idx]
                    qml.CRX(angle_crx, wires=[wire_ctrl, wire_targ])
                    ent_idx += 1
            for wire in range(self.n_qubits_):
                qml.RX(weights[n_layer, wire, -2], wires=wire)
                qml.RZ(weights[n_layer, wire, -1], wires=wire)


class ScikitBase:
    _model_template = None
    _parameters_template = {}
    _parameters = {}
    _id = None

    def __init__(self, *args, **kwargs):
        self._parameters = {
            "random_state": kwargs.get("random_state"),
            **self._parameters_template,
        }
        self._model = self._model_template(**self._parameters)

    def fit(self, X, y, sample_weight=None):
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
    _id = 38
    _parameters_template = {"kernel": "rbf", "probability": True, "verbose": True}


class SVMLinear(SVMBase):
    _id = 36
    _parameters_template = {"kernel": "linear", "probability": True, "verbose": True}


class SVMPoly(SVMBase):
    _id = 37
    _parameters_template = {"kernel": "poly", "probability": True, "verbose": True}


class MyLogisticRegression(ScikitBase):
    _id = 32
    _model_template = LogisticRegression
    _parameters_template = {"max_iter": 1_000_000, "verbose": 5}


class RandomForest(ScikitBase):
    _id = 35
    _model_template = RandomForestClassifier
    _parameters_template = {"verbose": 5}


class KNN(ScikitBase):
    _id = 34
    _model_template = KNeighborsClassifier

    def __init__(self, *args, **kwargs):
        self._model = self._model_template()

    def fit(self, X, y, sample_weight=None):
        self._model.fit(X, y)


class MLP(ScikitBase):
    _id = 33
    _model_template = MLPClassifier
    _parameters_template = {"verbose": True, "hidden_layer_sizes": ()}

    def fit(self, X, y, sample_weight=None):
        self._model.fit(X, y)


class AdaBoostRotCNOT(ScikitBase):
    _id = 16

    def __init__(self, *args, **kwargs):
        qvc = AnsatzRotCNOT(*args, **kwargs)
        self._model = AdaBoostClassifier(
            estimator=qvc, n_estimators=10, random_state=kwargs.get("random_state")
        )


class BaggingRotCNOT(ScikitBase):
    _id = 17

    def __init__(self, *args, **kwargs):
        qvc = AnsatzRotCNOT(*args, **kwargs)
        self._model = BaggingClassifier(
            estimator=qvc,
            n_estimators=10,
            random_state=kwargs.get("random_state"),
        )


class AdaBoostEnt4(ScikitBase):
    _id = 18

    def __init__(self, *args, **kwargs):
        qvc = AnsatzEnt4(*args, **kwargs)
        self._model = AdaBoostClassifier(
            estimator=qvc, n_estimators=10, random_state=kwargs.get("random_state")
        )


class BaggingEnt4(ScikitBase):
    _id = 19

    def __init__(self, *args, **kwargs):
        qvc = AnsatzEnt4(*args, **kwargs)
        self._model = BaggingClassifier(
            estimator=qvc,
            n_estimators=10,
            random_state=kwargs.get("random_state"),
        )


class AdaBoostMaouaki15(ScikitBase):
    _id = 20

    def __init__(self, *args, **kwargs):
        qvc = AnsatzMaouaki15(*args, **kwargs)
        self._model = AdaBoostClassifier(
            estimator=qvc, n_estimators=10, random_state=kwargs.get("random_state")
        )


class BaggingMaouaki15(ScikitBase):
    _id = 21

    def __init__(self, *args, **kwargs):
        qvc = AnsatzMaouaki15(*args, **kwargs)
        self._model = BaggingClassifier(
            estimator=qvc,
            n_estimators=10,
            random_state=kwargs.get("random_state"),
        )


class Voting(ScikitBase):
    voting_method = None

    def __init__(self, *args, **kwargs):
        qvcs = self.get_models(*args, **kwargs)
        self._model = VotingClassifier(estimators=qvcs, voting=self.voting_method)

    def get_models(self, *args, **kwargs):
        raise NotImplementedError("Implement this class")


class SoftVoting(Voting):
    voting_method = "soft"


class HardVoting(Voting):
    voting_method = "hard"


class Voting_1_2_3(Voting):
    def get_models(self, *args, **kwargs):
        return [
            ("single_rot_x", AnsatzSingleRotX(*args, **kwargs)),
            ("single_rot_y", AnsatzSingleRotY(*args, **kwargs)),
            ("single_rot_z", AnsatzSingleRotZ(*args, **kwargs)),
        ]


class SoftVoting_1_2_3(Voting_1_2_3, SoftVoting):
    _id = 22


class HardVoting_1_2_3(Voting_1_2_3, HardVoting):
    _id = 23


class Voting_1_2_3_5(Voting):
    def get_models(self, *args, **kwargs):
        return [
            ("single_rot_x", AnsatzSingleRotX(*args, **kwargs)),
            ("single_rot_y", AnsatzSingleRotY(*args, **kwargs)),
            ("single_rot_z", AnsatzSingleRotZ(*args, **kwargs)),
            ("rot", AnsatzRot(*args, **kwargs)),
        ]


class SoftVoting_1_2_3_5(Voting_1_2_3_5, SoftVoting):
    _id = 24


class HardVoting_1_2_3_5(Voting_1_2_3_5, HardVoting):
    _id = 25


class Voting_1_2_3_5_6(Voting):
    def get_models(self, *args, **kwargs):
        return [
            ("single_rot_x", AnsatzSingleRotX(*args, **kwargs)),
            ("single_rot_y", AnsatzSingleRotY(*args, **kwargs)),
            ("single_rot_z", AnsatzSingleRotZ(*args, **kwargs)),
            ("rot", AnsatzRot(*args, **kwargs)),
            ("rotcnot", AnsatzRotCNOT(*args, **kwargs)),
        ]


class SoftVoting_1_2_3_5_6(Voting_1_2_3_5_6, SoftVoting):
    _id = 26


class HardVoting_1_2_3_5_6(Voting_1_2_3_5_6, HardVoting):
    _id = 27


class Voting_7_8_9_10_11(Voting):
    def get_models(self, *args, **kwargs):
        return [
            ("maouaki9", AnsatzMaouaki9(*args, **kwargs)),
            ("ent1", AnsatzEnt1(*args, **kwargs)),
            ("ent2", AnsatzEnt2(*args, **kwargs)),
            ("ent3", AnsatzEnt3(*args, **kwargs)),
            ("ent4", AnsatzEnt4(*args, **kwargs)),
        ]


class SoftVoting_7_8_9_10_11(Voting_7_8_9_10_11, SoftVoting):
    _id = 28


class HardVoting_7_8_9_10_11(Voting_7_8_9_10_11, HardVoting):
    _id = 29


class Voting_12_14_15(Voting):
    def get_models(self, *args, **kwargs):
        return [
            ("maouakiquasi7", AnsatzMaouakiQuasi7(*args, **kwargs)),
            ("maouaki15", AnsatzMaouaki15(*args, **kwargs)),
            ("maouaki6", AnsatzMaouaki6(*args, **kwargs)),
        ]


class SoftVoting_12_14_15(Voting_12_14_15, SoftVoting):
    _id = 30


class HardVoting_12_14_15(Voting_12_14_15, HardVoting):
    _id = 31


class AdaBoostLogistic(ScikitBase):
    _id = 39

    def __init__(self, *args, **kwargs):
        qvc = MyLogisticRegression(*args, **kwargs)._model
        self._model = AdaBoostClassifier(
            estimator=qvc, n_estimators=10, random_state=kwargs.get("random_state")
        )


class BaggingLogistic(ScikitBase):
    _id = 40

    def __init__(self, *args, **kwargs):
        qvc = MyLogisticRegression(*args, **kwargs)._model
        self._model = BaggingClassifier(
            estimator=qvc,
            n_estimators=10,
            random_state=kwargs.get("random_state"),
        )


class VotingSVM(Voting):
    def get_models(self, *args, **kwargs):
        return [
            ("svmrbf", SVMRBF(*args, **kwargs)._model),
            ("svmpoly", SVMPoly(*args, **kwargs)._model),
            ("svmlinear", SVMLinear(*args, **kwargs)._model),
        ]


class SoftVotingSVM(VotingSVM, SoftVoting):
    _id = 41


class HardVotingSVM(VotingSVM, HardVoting):
    _id = 42


class VotingLogisticMLPKNN(Voting):
    def get_models(self, *args, **kwargs):
        return [
            ("logistic", MyLogisticRegression(*args, **kwargs)._model),
            ("mlp", MLP(*args, **kwargs)._model),
            ("knn", KNN(*args, **kwargs)._model),
        ]


class SoftVotingLogisticMLPKNN(VotingLogisticMLPKNN, SoftVoting):
    _id = 43


class HardVotingLogisticMLPKNN(VotingLogisticMLPKNN, HardVoting):
    _id = 44


def get_model_classifier(args, seed):
    model_factory = {
        "singlerotx": AnsatzSingleRotX,
        "singleroty": AnsatzSingleRotY,
        "singlerotz": AnsatzSingleRotZ,
        "rot": AnsatzRot,
        "rotcnot": AnsatzRotCNOT,
        "maouaki1": AnsatzMaouaki1,
        "maouaki6": AnsatzMaouaki6,
        "maouakiquasi7": AnsatzMaouakiQuasi7,
        "maouaki7": AnsatzMaouaki7,
        "maouaki9": AnsatzMaouaki9,
        "maouaki15": AnsatzMaouaki15,
        "ent1": AnsatzEnt1,
        "ent2": AnsatzEnt2,
        "ent3": AnsatzEnt3,
        "ent4": AnsatzEnt4,
        "svmrbf": SVMRBF,
        "svmlinear": SVMLinear,
        "svmpoly": SVMPoly,
        "logistic": MyLogisticRegression,
        "randomforest": RandomForest,
        "knn": KNN,
        "mlp": MLP,
        "adaboost_rotcnot": AdaBoostRotCNOT,
        "bagging_rotcnot": BaggingRotCNOT,
        "adaboost_ent4": AdaBoostEnt4,
        "bagging_ent4": BaggingEnt4,
        "adaboost_maouaki15": AdaBoostMaouaki15,
        "bagging_maouaki15": BaggingMaouaki15,
        "soft_voting_1_2_3": SoftVoting_1_2_3,
        "hard_voting_1_2_3": HardVoting_1_2_3,
        "soft_voting_1_2_3_5": SoftVoting_1_2_3_5,
        "hard_voting_1_2_3_5": HardVoting_1_2_3_5,
        "soft_voting_1_2_3_5_6": SoftVoting_1_2_3_5_6,
        "hard_voting_1_2_3_5_6": HardVoting_1_2_3_5_6,
        "soft_voting_7_8_9_10_11": SoftVoting_7_8_9_10_11,
        "hard_voting_7_8_9_10_11": HardVoting_7_8_9_10_11,
        "soft_voting_12_14_15": SoftVoting_12_14_15,
        "hard_voting_12_14_15": HardVoting_12_14_15,
        "adaboost_logistic": AdaBoostLogistic,
        "bagging_logistic": BaggingLogistic,
        "soft_voting_svm": SoftVotingSVM,
        "hard_voting_svm": HardVotingSVM,
        "soft_voting_logistic_mlp_knn": SoftVotingLogisticMLPKNN,
        "hard_voting_logistic_mlp_knn": HardVotingLogisticMLPKNN,
    }

    model_name = args.model_classifier

    return model_factory[model_name](
        n_layers=args.n_layers,
        max_iter=args.epochs,
        batch_size=args.batch_size,
        random_state=seed,
        n_qubits_=args.n_qubits,
        testing=args.testing,
    )
