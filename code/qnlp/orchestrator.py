import random
from pathlib import Path

import numpy as np
import torch
from joblib import Parallel, delayed
from pytreebank import load_sst


class QNLP:
    def __init__(self):
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)
        self._path = None
        self._df = None
        self._repetitions = 30
        self._define_path()

    def run(self):
        self._get_data()
        self._process_data()
        self._run_models()
        self._agg_metrics()
        print("End.")

    def _define_path(self):
        self._path = {"root": Path(__file__).parent.parent.parent.resolve()}
        self._path["data"] = self._path["root"] / "data"

    def _get_data(self):
        self._df = load_sst(self._path["data"] / "sst")

    def _process_data(self):
        pass

    def _run_models(self):
        pass

    def _agg_metrics(self):
        pass
