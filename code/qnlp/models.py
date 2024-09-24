import pickle
from functools import partial

import pennylane as qml
import torch
import xgboost as xgb
from hyperopt import STATUS_OK, fmin, hp, tpe
from hyperopt.fmin import generate_trials_to_calculate
from joblib import Parallel, delayed
from pennylane import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import SVC
from torch.nn import Linear, Module, MSELoss, Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader


class Model:
    def __init__(self, path, n_repetitions, testing):
        self._path = path
        self._n_repetitions = n_repetitions
        self.testing = testing

    def run(self, df):
        raise NotImplementedError

    def _run(self, dataset, seed):
        raise NotImplementedError


class ClassicalModel(Model):
    def run(self, df):
        return Parallel(n_jobs=self._n_repetitions)(
            delayed(self._run)(df, seed) for seed in range(self._n_repetitions)
        )

    def _run(self, dataset, seed):
        raise NotImplementedError


class NNModel(ClassicalModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._n_epochs = 1 if self.testing else 50
        self._max_trials = 3 if self.testing else 50
        self._batch = 16


class NNClassicalTorchModel(torch.nn.Module):
    def __init__(self, n_features, n_qubits):
        super().__init__()
        self.layer_1 = torch.nn.Linear(n_features, n_qubits)
        self.layer_2 = torch.nn.Linear(n_qubits, n_qubits)
        self.layer_3 = torch.nn.Linear(n_qubits, 1)
        self.activation = torch.sigmoid

    def forward(self, x):
        x = self.activation(self.layer_1(x)) * np.pi / 2
        x = self.activation(self.layer_2(x))
        x = self.activation(self.layer_3(x))
        return x[:, 0]


class NNClassicalModel(NNModel):
    def _run(self, dataset, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        x_train = np.concatenate(
            (dataset["train"]["embeddings"], dataset["dev"]["embeddings"])
        )
        x_train = torch.tensor(x_train, dtype=torch.float32, requires_grad=False)
        y_train = np.concatenate((dataset["train"]["labels"], dataset["dev"]["labels"]))
        y_train = torch.tensor(y_train, dtype=torch.float32, requires_grad=False)
        x_test = dataset["test"]["embeddings"]
        x_test = torch.tensor(x_test, dtype=torch.float32, requires_grad=False)
        y_test = dataset["test"]["labels"]
        train_loader = DataLoader(
            zip((x_train, y_train), strict=True), batch_size=self._batch, shuffle=True
        )
        model = NNClassicalTorchModel(x_train.shape[1], n_qubits=8)
        opt = Adam(model.parameters())
        loss_fn = MSELoss()

        for i in range(self._n_epochs):
            for x_train_batch, y_train_batch in train_loader:
                opt.zero_grad()
                y_pred_batch = model(x_train_batch)
                loss = loss_fn(y_train_batch, y_pred_batch)
                print(f"Seed {seed} - Epoch {i} - Loss: {loss.item()}")
                loss.backward()
                opt.step()
            with torch.no_grad():
                y_pred = (model(x_test) > 0.5).detach().numpy()
                acc = balanced_accuracy_score(y_test, y_pred)
                print(f"Seed {seed} - Epoch {i} - Balanced Accuracy: {acc}")
        y_test_pred = (model(x_test) > 0.5).detach().numpy()
        return seed, y_test, y_test_pred


class HyBridTorchModel(Module):
    def __init__(self, n_features, n_qubits, tau, delta):
        super().__init__()
        self._n_features = n_features
        self._n_qubits = n_qubits
        self.layer_1 = Linear(self._n_features, self._n_qubits)
        self.quantum_layers_0 = [
            self.create_quantum_layer(tau, delta) for _ in range(self._n_qubits)
        ]
        self.weights_quantum_layers_0 = torch.nn.ParameterList(
            [layer.weights for layer in self.quantum_layers_0]
        )
        self.quantum_layers_1 = self.create_quantum_layer(tau, delta)
        self.biases = Parameter(torch.zeros(n_qubits + 1), requires_grad=True)
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
        layer = qml.qnn.TorchLayer(
            circuit, weight_shapes, init_method={"weights": self.create_weight}
        )
        return layer

    def create_weight(self, *args, **kwargs):
        weight = torch.rand(size=(1, self._n_qubits)) * np.pi / 2
        return weight

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


class HybridModel(NNModel):
    def run(self, df):
        if (self._path["hybrid"] / "hyperopt_rng.pth").exists():
            rng_state = torch.load(self._path["hybrid"] / "hyperopt_rng.pth")
            rng = np.random.default_rng()
            rng.bit_generator.state = rng_state
        else:
            rng = np.random.default_rng(0)

        objective = self._get_objective(df)
        space = {
            "tau": hp.uniform("tau", -5, 5),
            "delta": hp.uniform("delta", 0, 2 * np.pi),
        }
        if (self._path["hybrid"] / "model.hyperopt").exists():
            with open(self._path["hybrid"] / "model.hyperopt", "rb") as f:
                trials = pickle.load(f)
        else:
            trials = generate_trials_to_calculate([{"tau": 1, "delta": 0}])

        n_trials = len(trials.trials)
        if n_trials < self._max_trials:
            while n_trials < self._max_trials:
                i = len(trials.trials)
                best_hypers = fmin(
                    fn=objective,
                    space=space,
                    trials=trials,
                    algo=partial(tpe.suggest, n_startup_jobs=20),
                    max_evals=i + 1,
                    trials_save_file=self._path["hybrid"] / "model.hyperopt",
                    rstate=rng,
                )
                torch.save(
                    rng.bit_generator.state, self._path["hybrid"] / "hyperopt_rng.pth"
                )
                with open(self._path["hybrid"] / "model.hyperopt", "wb") as f:
                    pickle.dump(trials, f)
                n_trials = len(trials.trials)
        else:
            best_hypers = fmin(
                fn=objective,
                space=space,
                trials=trials,
                algo=partial(tpe.suggest, n_startup_jobs=20),
                max_evals=self._max_trials,
                trials_save_file=self._path["hybrid"] / "model.hyperopt",
                rstate=rng,
            )
        return Parallel(n_jobs=self._n_repetitions)(
            delayed(self._run_hybrid)(
                df,
                seed,
                best_hypers["tau"],
                best_hypers["delta"],
                n_qubits=8,
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
                    df, seed, tau, delta, n_qubits=8, dataset_evaluation="dev"
                )
                for seed in range(self._n_repetitions)
            )
            acc_list = []
            for _, y_test, y_test_pred in results:
                acc = balanced_accuracy_score(y_test, y_test_pred)
                acc_list.append(acc)
            return {
                "loss": -np.mean(acc_list),
                "status": STATUS_OK,
                "raw_results": results,
            }

        return objective

    def _run_hybrid(self, dataset, seed, tau, delta, n_qubits, dataset_evaluation):
        torch.manual_seed(seed)
        np.random.seed(seed)

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
        x_test = torch.tensor(
            dataset[dataset_evaluation]["embeddings"], dtype=torch.float32
        )
        y_test = (
            torch.tensor(dataset[dataset_evaluation]["labels"], dtype=torch.float32)
            .detach()
            .numpy()
        )
        n_features = dataset["train"]["embeddings"].shape[1]
        model = HyBridTorchModel(n_features, n_qubits, tau, delta)
        train_loader = DataLoader(
            list(zip(x_train, y_train, strict=True)),
            batch_size=self._batch,
            shuffle=True,
        )
        opt = Adam(model.parameters())
        loss_fn = MSELoss()

        for i in range(self._n_epochs):
            print(f"Seed {seed} - Epoch {i}")
            for o, (x_train_batch, y_train_batch) in enumerate(train_loader):
                if o % 100 == 0:
                    print(f"Seed {seed} - Epoch {i} - Batch {o}")
                opt.zero_grad()
                y_pred_batch = model(x_train_batch)
                loss = loss_fn(y_train_batch, y_pred_batch)
                loss.backward()
                opt.step()
            with torch.no_grad():
                y_pred = (model(x_test) > 0.5).detach().numpy()
                acc = balanced_accuracy_score(y_test, y_pred)
                print(f"Seed {seed} - Epoch {i} - Balanced Accuracy: {acc}")
        y_test_pred = (model(x_test) > 0.5).detach().numpy()
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
    "random_forest": RandomForestModel,
    "svm_linear": SVMLinearModel,
    "svm_poly": SVMPolyModel,
    "svm_rbf": SVMRBFModel,
    "logistic_regression": LogisticRegressionModel,
    "dummy": DummyModel,
    "xgboost": XGBoostModel,
    "nn_classical": NNClassicalModel,
    "nn_hybrid": HybridModel,
}
