import numpy as np
import torch
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from xgboost.callback import EarlyStopping

from .utils import PATIENCE, transform_labels


class Model:
    _default_params = {}

    def __init__(self, datasets: dict):
        self.datasets = datasets

    def run_train_valid(self, seed: int):
        x_train = self.datasets["train"]["x"]
        y_train = self.datasets["train"]["y"]
        x_test = self.datasets["valid"]["x"]
        y_test = self.datasets["valid"]["y"]
        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        return self._run_train_valid(x_train, y_train, x_test, y_test, seed)

    def run_train_test(self, hyper_params: dict):
        x_train = self.datasets["train"]["x"]
        y_train = self.datasets["train"]["y"]
        x_valid = self.datasets["valid"]["x"]
        y_valid = self.datasets["valid"]["y"]
        x_test = self.datasets["test"]["x"]
        y_test = self.datasets["test"]["y"]
        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train)
        x_valid = scaler.transform(x_valid)
        x_test = scaler.transform(x_test)
        x_train = np.concatenate((x_train, x_valid), axis=0)
        y_train = np.concatenate((y_train, y_valid), axis=0)
        if hyper_params is None:
            hyper_params = self._default_params
        else:
            hyper_params = {**self._default_params, **hyper_params}
        return self._run_train_test(x_train, y_train, x_test, y_test, hyper_params)

    def _run_train_valid(self, x_train, y_train, x_test, y_test, seed):
        raise NotImplementedError

    def _run_train_test(self, x_train, y_train, x_test, y_test, hyper_params: dict):
        raise NotImplementedError


class SklearnModel(Model):
    _model_template = None
    _default_params = {}

    def _run_train_valid(self, x_train, y_train, x_test, y_test, seed):
        return {"random_state": seed}

    def _run_train_test(self, x_train, y_train, x_test, y_test, hyper_params: dict):
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


class XGboostModel(Model):
    _default_params = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "tree_method": "hist",
        "objective": "binary:logistic",
    }

    def _run_train_valid(self, x_train, y_train, x_test, y_test, seed):
        train = xgb.DMatrix(x_train, label=transform_labels(y_train))
        test = xgb.DMatrix(x_test, label=transform_labels(y_test))

        params = {
            **self._default_params,
            "seed": seed,
            "seed_per_iteration": True,
        }

        callback = [
            EarlyStopping(
                rounds=PATIENCE,
                min_delta=1e-4,
                save_best=True,
                maximize=False,
                data_name="valid",
                metric_name="logloss",
            )
        ]

        model = xgb.train(
            params,
            train,
            num_boost_round=1_000_000,
            evals=[(train, "train"), (test, "valid")],
            callbacks=callback,
        )

        return {
            **params,
            "num_boost_round": model.best_iteration + 2,
        }

    def _run_train_test(self, x_train, y_train, x_test, y_test, hyper_params: dict):
        train = xgb.DMatrix(x_train, label=transform_labels(y_train))
        test = xgb.DMatrix(x_test, label=transform_labels(y_test))
        num_boost_round = hyper_params.pop("num_boost_round")
        model = xgb.train(
            hyper_params, train, num_boost_round=num_boost_round, evals=[(test, "test")]
        )
        y_pred = model.predict(test)
        y_pred = np.where(y_pred > 0.5, 1, -1)
        return y_pred


class QuantumModel(Model):
    pass


class ContinuousNeuronQuantumModel(QuantumModel):
    pass


class ParametricNeuronQuantumModel(QuantumModel):
    pass


class ModelHandler:
    # models = {"CNQ": ContinuousNeuronQuantumModel, "PNQ": ParametricNeuronQuantumModel}

    models = {
        "RF": RandomForestModel,
        "SVMLinear": SVMLinearModel,
        "SVMRBF": SVMRBFModel,
        "SVMPoly": SVMPolyModel,
        "LR": LogisticRegressionModel,
        "XGB": XGboostModel,
    }

    def __init__(self):
        self.model = None
        self.seed = None

    def load(self, model_name: str, datasets: dict, seed: int):
        self.model = self.models[model_name](datasets)
        self.seed = seed

    def run(self):
        hyper_params = self.model.run_train_valid(seed=self.seed)
        return self.model.run_train_test(hyper_params=hyper_params)
