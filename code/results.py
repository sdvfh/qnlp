import pickle
from pathlib import Path

root_path: Path = Path(__file__).parent.parent.resolve()
data_path: Path = root_path / "data"
results_path: Path = root_path / "results"


file_results_path = results_path.rglob("*.pkl")

results = []

for file in file_results_path:
    with open(file, "rb") as f:
        metrics = pickle.load(f)
        metrics["file"] = file.stem
        results.append(metrics)
