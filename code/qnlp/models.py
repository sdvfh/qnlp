from functools import partial

import pennylane as qml
import torch
import xgboost as xgb
from hyperopt import fmin, hp, tpe
from hyperopt.fmin import generate_trials_to_calculate
from joblib import Parallel, delayed
from pennylane import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import SVC


class Model:
    def __init__(self, path, n_repetitions, testing):
        self._path = path
        self._n_repetitions = n_repetitions
        self.testing = testing
        self._n_epochs = 1 if self.testing else 100

    def run(self, df):
        raise NotImplementedError

    def _run(self, dataset, seed):
        raise NotImplementedError

    # def _save_state(self):
    #     states = {
    #         "cpu_rng_state": torch.get_rng_state(),
    #         "gpu_rng_state": torch.cuda.get_rng_state(),
    #         "numpy_rng_state": np.random.get_state(),
    #         "py_rng_state": random.getstate(),
    #     }
    #     torch.save(states, self._path["data"] / "states.pth")

    # def _load_state(self):
    #     if not (self._path["data"] / "states.pth").exists():
    #         return
    #     states = torch.load(self._path["data"] / "states.pth")
    #     torch.set_rng_state(states["cpu_rng_state"])
    #     torch.cuda.set_rng_state(states["gpu_rng_state"])
    #     np.random.set_state(states["numpy_rng_state"])
    #     random.setstate(states["py_rng_state"])


class ClassicalModel(Model):
    def run(self, df):
        return Parallel(n_jobs=self._n_repetitions)(
            delayed(self._run)(df, seed) for seed in range(self._n_repetitions)
        )

    def _run(self, dataset, seed):
        raise NotImplementedError


class HyBridTorchModel(torch.nn.Module):
    def __init__(self, n_features, n_qubits, tau, delta):
        super().__init__()
        self._n_features = n_features
        self._n_qubits = n_qubits
        self.layer_1 = torch.nn.Linear(self._n_features, self._n_qubits)
        self.quantum_layers_0 = [
            self.create_quantum_layer(tau, delta) for _ in range(self._n_qubits)
        ]
        self.quantum_layers_1 = self.create_quantum_layer(tau, delta)
        self.biases = torch.nn.Parameter(torch.zeros(n_qubits + 1), requires_grad=True)
        self.classical_activation = torch.sigmoid

    def forward(self, x):
        x = self.classical_activation(self.layer_1(x)) * np.pi / 2
        x_q_0 = torch.stack(
            [
                self.quantum_layers_0[n_qubit](x)[:, 1] + self.biases[n_qubit]
                for n_qubit in range(self._n_qubits)
            ],
            dim=1,
        )
        x_q_1 = self.quantum_layers_1(x_q_0)[:, 1] + self.biases[self._n_qubits]
        return x_q_1

    def create_quantum_layer(self, tau, delta):
        weight_shapes = {
            "weights": (1, self._n_qubits),
        }
        circuit = partial(self.circuit, tau, delta, self._n_qubits)
        dev = qml.device("default.qubit", wires=self._n_qubits + 1)
        circuit = qml.qnode(dev, interface="torch", diff_method="backprop")(circuit)
        circuit = qml.transforms.broadcast_expand(circuit)
        return qml.qnn.TorchLayer(
            circuit, weight_shapes, init_method={"weights": self.create_weight}
        )

    def create_weight(self, *args, **kwargs):
        return torch.rand(size=(1, self._n_qubits)) * np.pi / 2

    @staticmethod
    def circuit(tau, delta, n_qubits, inputs, weights):
        target_wire = [n_qubits]
        wires = list(range(n_qubits))
        vector_data = tau * (inputs - weights) + delta

        qml.broadcast(qml.Hadamard, wires=wires, pattern="single")
        for wire in range(len(wires)):
            qml.PhaseShift(vector_data[:, wire], wires=wire)
        qml.broadcast(qml.Hadamard, wires=wires, pattern="single")
        qml.broadcast(qml.X, wires=wires, pattern="single")
        qml.MultiControlledX(wires=wires + target_wire)
        return qml.probs(wires=target_wire)


class HybridModel(Model):
    def run(self, df):
        rng = np.random.default_rng(0)
        objective = self._get_objective(df)
        space = {
            "tau": hp.uniform("tau", -5, 5),
            "delta": hp.uniform("delta", 0, 2 * np.pi),
        }
        trials = generate_trials_to_calculate([{"tau": 1, "delta": 0}])
        max_trials = 2 if self.testing else 49
        best_hypers = fmin(
            fn=objective,
            space=space,
            trials=trials,
            algo=partial(tpe.suggest, n_startup_jobs=20),
            max_evals=max_trials,
            trials_save_file=self._path["data"] / "model.hyperopt",
            rstate=rng,
        )
        return Parallel(n_jobs=self._n_repetitions)(
            delayed(self._run_hybrid)(
                df,
                seed,
                best_hypers["tau"],
                best_hypers["delta"],
                n_qubits=8,
                batch=16,
                dataset_evaluation="test",
            )
            for seed in range(self._n_repetitions)
        )

    def _get_objective(self, df):
        def objective(args):
            tau = args["tau"]
            delta = args["delta"]
            results = Parallel(n_jobs=self._n_repetitions)(
                delayed(self._run_hybrid)(
                    df, seed, tau, delta, n_qubits=8, batch=16, dataset_evaluation="dev"
                )
                for seed in range(self._n_repetitions)
            )
            return -np.mean(results)

        return objective

    def _run_hybrid(
        self, dataset, seed, tau, delta, n_qubits, batch, dataset_evaluation
    ):
        n_features = dataset["train"]["embeddings"].shape[1]
        model = HyBridTorchModel(n_features, n_qubits, tau, delta)
        opt = torch.optim.Adam(model.parameters())
        loss_fn = torch.nn.MSELoss()
        if dataset_evaluation == "dev":
            x_train = torch.tensor(
                dataset["train"]["embeddings"], dtype=torch.float32, requires_grad=False
            )
            y_train = torch.tensor(
                dataset["train"]["labels"], dtype=torch.float32, requires_grad=False
            )
        else:
            x_train = np.concatenate(
                (dataset["train"]["embeddings"], dataset["dev"]["embeddings"])
            )
            y_train = np.concatenate(
                (dataset["train"]["labels"], dataset["dev"]["labels"])
            )
            x_train = torch.tensor(x_train, dtype=torch.float32, requires_grad=False)
            y_train = torch.tensor(y_train, dtype=torch.float32, requires_grad=False)
        for i in range(self._n_epochs):
            idx_batch = np.random.choice(len(x_train), batch)
            x_train_batch = x_train[idx_batch,]
            y_train_batch = y_train[idx_batch,]

            opt.zero_grad()
            y_pred_batch = model(x_train_batch)
            loss = loss_fn(y_train_batch, y_pred_batch)
            print(f"Epoch {i} - Loss: {loss.item()}")
            loss.backward()
            opt.step()
        x_test = torch.tensor(
            dataset[dataset_evaluation]["embeddings"], dtype=torch.float32
        )
        y_test = (
            torch.tensor(dataset[dataset_evaluation]["labels"], dtype=torch.float32)
            .detach()
            .numpy()
        )
        y_test_pred = (model(x_test) > 0.5).detach().numpy()
        if dataset_evaluation == "dev":
            acc = balanced_accuracy_score(y_test, y_test_pred)
            return acc
        else:
            return seed, y_test, y_test_pred


class SKLearnModel(ClassicalModel):
    _model_template = None
    _default_params = {}

    def _run(self, dataset, seed):
        if self._model_template is None:
            raise NotImplementedError
        params = self._default_params.copy()
        params["random_state"] = seed
        x_train = np.concatenate(
            (dataset["train"]["embeddings"], dataset["dev"]["embeddings"])
        )
        y_train = np.concatenate((dataset["train"]["labels"], dataset["dev"]["labels"]))
        x_test = dataset["test"]["embeddings"]
        y_test = dataset["test"]["labels"]
        model = self._model_template(**params)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        return seed, y_test, y_pred


class RandomForestModel(SKLearnModel):
    _model_template = RandomForestClassifier
    _default_params = {"n_jobs": 1, "verbose": 1}


class SVMModel(SKLearnModel):
    _model_template = SVC


class SVMLinearModel(SVMModel):
    _default_params = {"kernel": "linear"}


class SVMPolyModel(SVMModel):
    _default_params = {"kernel": "rbf"}


class SVMRBFModel(SVMModel):
    _default_params = {"kernel": "poly"}


class LogisticRegressionModel(SKLearnModel):
    _model_template = LogisticRegression
    _default_params = {"max_iter": 1_000_000}


class DummyModel(SKLearnModel):
    _model_template = DummyClassifier
    _default_params = {"strategy": "stratified"}


class XGBoostModel(ClassicalModel):
    _default_params = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "tree_method": "hist",
        "objective": "binary:logistic",
        "seed_per_iteration": True,
    }

    def _run(self, dataset, seed):
        params = self._default_params.copy()
        params["seed"] = seed
        x_train = np.concatenate(
            (dataset["train"]["embeddings"], dataset["dev"]["embeddings"])
        )
        y_train = np.concatenate((dataset["train"]["labels"], dataset["dev"]["labels"]))
        x_test = dataset["test"]["embeddings"]
        y_test = dataset["test"]["labels"]
        train = xgb.DMatrix(x_train, label=y_train)
        test = xgb.DMatrix(x_test, label=y_test)
        model = xgb.train(params, train, evals=[(test, "test")])
        y_pred = model.predict(test)
        y_pred = (y_pred > 0.5).astype(int)
        return seed, y_test, y_pred


models = {
    # "random_forest": RandomForestModel,
    # "svm_linear": SVMLinearModel,
    # "svm_poly": SVMPolyModel,
    # "svm_rbf": SVMRBFModel,
    # "logistic_regression": LogisticRegressionModel,
    # "dummy": DummyModel,
    # "xgboost": XGBoostModel,
    "hybrid": HybridModel,
}
