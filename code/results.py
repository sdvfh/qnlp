import gzip
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def read_results() -> pd.DataFrame:
    """Read all results from the results folder and return a DataFrame."""
    result_files = Path("../results").rglob("*.pkl.gz")
    results_list = []
    for result_file in result_files:
        with gzip.open(result_file, "rb") as f:
            data = pickle.load(f)
        data["symbols"] = {symbol.name: symbol for symbol in data["symbols"]}
        results_list.append(data)
    return pd.DataFrame(results_list)


results = read_results()
