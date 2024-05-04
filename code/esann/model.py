from time import time

import pennylane as qml
import torch
import xgboost as xgb
from pennylane import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from .utils import transform_labels


class Model:
    _default_params = {}

    def __init__(self, datasets: dict, *args, **kwargs):
        self.datasets = datasets

    def run(self, seed: int):
        hyper_params = self._get_hyper_params(seed=seed)
        x_train = self.datasets["train"]["x"]
        y_train = self.datasets["train"]["y"]
        x_test = self.datasets["test"]["x"]
        y_test = self.datasets["test"]["y"]
        scaler = MinMaxScaler(feature_range=(0, np.pi / 2))
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        if hyper_params is None:
            hyper_params = self._default_params
        else:
            hyper_params = {**self._default_params, **hyper_params}
        return self._run(x_train, y_train, x_test, y_test, hyper_params)

    def _run(self, x_train, y_train, x_test, y_test, hyper_params: dict):
        raise NotImplementedError

    def _get_hyper_params(self, seed: int):
        raise NotImplementedError


class SklearnModel(Model):
    _model_template = None
    _default_params = {}

    def _get_hyper_params(self, seed):
        return {"random_state": seed}

    def _run(self, x_train, y_train, x_test, y_test, hyper_params: dict):
        model = self._model_template(**hyper_params)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        return y_pred


class RandomForestModel(SklearnModel):
    _model_template = RandomForestClassifier
    _default_params = {"n_jobs": -1, "verbose": 1}


class SVMModel(SklearnModel):
    _model_template = SVC


class SVMLinearModel(SVMModel):
    _default_params = {"kernel": "linear"}


class SVMRBFModel(SVMModel):
    _default_params = {"kernel": "rbf"}


class SVMPolyModel(SVMModel):
    _default_params = {"kernel": "poly"}


class LogisticRegressionModel(SklearnModel):
    _model_template = LogisticRegression
    _default_params = {"max_iter": 1_000_000}


class DummyModel(SklearnModel):
    _model_template = DummyClassifier
    _default_params = {"strategy": "stratified"}


class XGboostModel(Model):
    _default_params = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "tree_method": "hist",
        "objective": "binary:logistic",
    }

    def _get_hyper_params(self, seed):
        return {
            **self._default_params,
            "seed": seed,
            "seed_per_iteration": True,
        }

    def _run(self, x_train, y_train, x_test, y_test, hyper_params: dict):
        train = xgb.DMatrix(x_train, label=transform_labels(y_train))
        test = xgb.DMatrix(x_test, label=transform_labels(y_test))
        model = xgb.train(hyper_params, train, evals=[(test, "test")])
        y_pred = model.predict(test)
        y_pred = np.where(y_pred > 0.5, 1, -1)
        return y_pred


class QuantumModel(Model):
    _default_params = {"batch_size": np.inf, "n_epochs": 30}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_hyper_params(self, seed):
        np.random.seed(seed)
        return self._default_params

    def _run(self, x_train, y_train, x_test, y_test, hyper_params: dict):
        n_qubits = self._get_n_qubits(x_train.shape[1])
        n_all_qubits = n_qubits + 1
        batch_size = len(x_train)
        n_epochs = hyper_params["n_epochs"]
        device = "default.qubit"
        dev = qml.device(device, wires=n_all_qubits)
        opt = qml.AdamOptimizer(stepsize=0.1)
        weights, bias = self._get_initial_weights_bias(n_features=x_train.shape[1])

        # @qml.transforms.broadcast_expand
        @qml.qnode(dev)
        def circuit(weights, x):
            wires = list(range(n_qubits))
            target_wire = [n_qubits]
            # weights_repeated = np.repeat(weights, len(x), axis=0)
            weights_repeated = weights

            qml.broadcast(qml.Hadamard, wires=wires, pattern="single")

            self._specific_ansatz(weights_repeated, x, wires, target_wire)

            qml.broadcast(qml.Hadamard, wires=wires, pattern="single")
            qml.broadcast(qml.X, wires=wires, pattern="single")

            qml.MultiControlledX(wires=wires + target_wire)
            return qml.probs(wires=target_wire)

        def cost(weights, bias, X, Y):
            predictions = [variational_classifier(weights, bias, x) for x in X]
            # loss = log_loss(Y, predictions)
            loss = square_loss(Y, predictions)
            return loss

        def variational_classifier(weights, bias, x):
            output_neuron = circuit(weights, x)
            return (2 * np.inner(output_neuron, [0, 1]) - 1) + bias

        def log_loss(y_true, y_pred):
            y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
            loss = np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))
            return loss

        def square_loss(labels, predictions):
            return np.mean((labels - np.array(predictions)) ** 2)

        y_pred = None
        for it in range(n_epochs):
            old = time()
            batch_index = np.random.randint(0, len(x_train), (batch_size,))
            x_batch = x_train[batch_index]
            y_batch = y_train[batch_index]

            (weights, bias, _, _), cost_value = opt.step_and_cost(
                cost, weights, bias, x_batch, y_batch
            )
            y_pred = [np.sign(variational_classifier(weights, bias, x)) for x in x_test]
            f1 = f1_score(y_test, y_pred)
            acc = balanced_accuracy_score(y_test, y_pred)
            print(
                "Iter: {:5d} | Cost: {:0.7f} | Accuracy: {:0.4f} | F1: {:0.4f} | Time: {:0.4f}".format(
                    it + 1, cost_value, acc, f1, time() - old
                )
            )
        return y_pred

    @staticmethod
    def _get_n_qubits(n_features):
        raise NotImplementedError

    def _encode_vector(self, input_vector, control_wires):
        raise NotImplementedError

    def _specific_ansatz(self, weights, x, wires, target_wire):
        raise NotImplementedError

    def _get_initial_weights_bias(self, n_features):
        weights = np.random.uniform(
            low=0, high=np.pi / 2, size=(n_features,), requires_grad=True
        )
        bias = np.array(0.0, requires_grad=True)
        return weights, bias

    @staticmethod
    def _transform_decimal_to_binary(decimal, n_qubits):
        binary_num = []
        while decimal > 0:
            binary_num.append(decimal % 2)
            decimal //= 2
        padding_length = n_qubits - len(binary_num)
        binary_num = binary_num + [0] * padding_length  # Pad at the beginning
        return binary_num


class ContinuousNeuronQuantumModel(QuantumModel):
    @staticmethod
    def _get_n_qubits(n_features):
        return int(np.ceil(np.log2(n_features)))

    def _encode_vector(self, input_vector, control_wires):
        n_qubits = len(control_wires)
        n_feature = 1
        for n_qubit in range(n_qubits):
            n_gates = 2**n_qubit
            actual_control_wires = control_wires.copy()
            actual_control_wires.remove(control_wires[n_qubit])
            for i in range(n_gates):
                control_value_binary = self._transform_decimal_to_binary(
                    i, n_qubits - 1
                )
                ctrl_phase_shift = qml.ctrl(
                    qml.PhaseShift,
                    control=actual_control_wires,
                    control_values=control_value_binary,
                )
                ctrl_phase_shift(
                    input_vector[n_feature] - input_vector[0],
                    wires=control_wires[n_qubit],
                )
                n_feature += 1

    def _specific_ansatz(self, weights, x, wires, target_wire):
        self._encode_vector(x, wires)
        self._encode_vector(weights, wires)


class ParametricNeuronQuantumModel(QuantumModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tau = kwargs["tau"]
        self._delta = kwargs["delta"]

    @staticmethod
    def _get_n_qubits(n_features):
        return n_features

    def _encode_vector(self, input_vector, control_wires):
        for i in range(len(control_wires)):
            qml.PhaseShift(input_vector[i], i)

    def _specific_ansatz(self, weights, x, wires, target_wire):
        self._encode_vector(self._tau * x, wires)
        self._encode_vector(-self._tau * weights, wires)
        self._encode_vector(self._delta * np.ones(x.shape, requires_grad=False), wires)


def construct_parametric(tau, delta):
    return lambda datasets: ParametricNeuronQuantumModel(datasets, tau=tau, delta=delta)


class ModelHandler:
    models = {
        "Dummy": DummyModel,
        "LR": LogisticRegressionModel,
        "SVMLinear": SVMLinearModel,
        "SVMRBF": SVMRBFModel,
        "SVMPoly": SVMPolyModel,
        "RF": RandomForestModel,
        "XGB": XGboostModel,
        "PCDQN_0.1_0.1": construct_parametric(tau=0.1, delta=np.pi / 2 * 0.1),
        "PCDQN_0.1_0.5": construct_parametric(tau=0.1, delta=np.pi / 2 * 0.5),
        "PCDQN_0.1_1.0": construct_parametric(tau=0.1, delta=np.pi / 2 * 1.0),
        "PCDQN_0.5_0.1": construct_parametric(tau=0.5, delta=np.pi / 2 * 0.1),
        "PCDQN_0.5_0.5": construct_parametric(tau=0.5, delta=np.pi / 2 * 0.5),
        "PCDQN_0.5_1.0": construct_parametric(tau=0.5, delta=np.pi / 2 * 1.0),
        "PCDQN_1.0_0.1": construct_parametric(tau=1.0, delta=np.pi / 2 * 0.1),
        "PCDQN_1.0_0.5": construct_parametric(tau=1.0, delta=np.pi / 2 * 0.5),
        "PCDQN_1.0_1.0": construct_parametric(tau=1.0, delta=np.pi / 2 * 1.0),
        "CDQN": ContinuousNeuronQuantumModel,
    }

    def __init__(self):
        self.model = None
        self.seed = None

    def load(self, model_name: str, datasets: dict, seed: int):
        self.model = self.models[model_name](datasets)
        self.seed = seed

    def run(self):
        return self.model.run(seed=self.seed)
