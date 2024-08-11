import numpy as np
import torch
import xgboost as xgb
from joblib import Parallel, delayed
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


class Model:
    def __init__(self, n_repetitions):
        self._n_repetitions = n_repetitions

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


class HybridModel(Model):
    def run(self, df):
        pass


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
    # "hybrid": HybridModel,
}
