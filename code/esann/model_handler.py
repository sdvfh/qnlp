import numpy as np
from sklearn.ensemble import RandomForestClassifier


class Model:
    _default_params = {}

    def __init__(self, datasets: dict):
        self.datasets = datasets

    def run_train_valid(self, seed: int):
        x_train = self.datasets["train"]["x"]
        y_train = self.datasets["train"]["y"]
        x_test = self.datasets["valid"]["x"]
        y_test = self.datasets["valid"]["y"]
        return self._run_train_valid(x_train, y_train, x_test, y_test, seed)

    def run_train_test(self, hyper_params: dict):
        x_train = np.row_stack(
            (self.datasets["train"]["x"], self.datasets["valid"]["x"])
        )
        y_train = np.concatenate(
            (self.datasets["train"]["y"], self.datasets["valid"]["y"])
        )
        x_test = self.datasets["test"]["x"]
        y_test = self.datasets["test"]["y"]
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
    pass


class SVMLinearModel(SVMModel):
    pass


class SVMRBFModel(SVMModel):
    pass


class SVMPolyModel(SVMModel):
    pass


class LogisticRegressionModel(SklearnModel):
    pass


class XGboostModel(Model):
    pass


class QuantumModel(Model):
    pass


class ContinuousNeuronQuantumModel(QuantumModel):
    pass


class ParametricNeuronQuantumModel(QuantumModel):
    pass


class ModelHandler:
    # models = {"RF": RandomForestModel, "SVMLinear": SVMLinearModel, "SVMRBF": SVMRBFModel,
    #           "SVMPoly": SVMPolyModel, "LR": LogisticRegressionModel, "XGB": XGboostModel,
    #           "CNQ": ContinuousNeuronQuantumModel, "PNQ": ParametricNeuronQuantumModel}

    models = {"RF": RandomForestModel}

    def __init__(self):
        self.model = None
        self.seed = None

    def load(self, model_name: str, datasets: dict, seed: int):
        self.model = self.models[model_name](datasets)
        self.seed = seed

    def run(self):
        hyper_params = self.model.run_train_valid(seed=self.seed)
        return self.model.run_train_test(hyper_params=hyper_params)
