from joblib import Parallel, delayed


class Model:
    def __init__(self, df):
        self._df = df
        self._n_repetitions = len(df)

    def run(self):
        raise NotImplementedError

    def _run(self):
        raise NotImplementedError


class ClassicalModel(Model):
    def run(self):
        return Parallel(n_jobs=self._n_repetitions)(
            delayed(self._run)(repetition) for repetition in range(self._n_repetitions)
        )

    def _run(self):
        pass


class SKLearnModel(ClassicalModel):
    pass


class RandomForestModel(SKLearnModel):
    pass


class SVMModel(SKLearnModel):
    pass


class SVMLinearModel(SVMModel):
    pass


class SVMPolyModel(SVMModel):
    pass


class SVMRBFModel(SVMModel):
    pass


class LogisticRegressionModel(SKLearnModel):
    pass


class DummyModel(SKLearnModel):
    pass


class XGBoostModel(ClassicalModel):
    pass


class HybridModel(Model):
    pass


models = {
    "random_forest": RandomForestModel,
    # "svm": SVMModel,
    # "svm_linear": SVMLinearModel,
    # "svm_poly": SVMPolyModel,
    # "svm_rbf": SVMRBFModel,
    # "logistic_regression": LogisticRegressionModel,
    # "dummy": DummyModel,
    # "xgboost": XGBoostModel,
    # "hybrid": HybridModel,
}
