import csv
from pathlib import Path

from qvc import models

# Configuration variables.
n_layers_list = [1, 10]
n_qubits = 10  # Number of qubits is now a variable.
samples = 1024
truncate_dim = 32

# Define the root path and CSV file path.
root_path: Path = Path(__file__).parent.parent.resolve()
csv_file_path: Path = root_path / "results" / str(truncate_dim) / "measures.csv"
csv_file_path.parent.mkdir(parents=True, exist_ok=True)

# Read existing entries from the CSV file, if it exists.
existing_entries = set()
if csv_file_path.exists():
    with open(csv_file_path, "r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Convert n_layers and n_qubits to integers for comparison.
            key = (row["Model"], int(row["n_layers"]), int(row["n_qubits"]))
            existing_entries.add(key)

# Open the CSV file in append mode (or write mode if it doesn't exist).
mode = "a" if csv_file_path.exists() else "w"
with open(csv_file_path, mode, newline="") as csvfile:
    writer = csv.writer(csvfile)
    # Write header if creating a new file.
    if mode == "w":
        writer.writerow(["Model", "n_layers", "n_qubits", "Haar", "Meyer-Wallach"])

    # Iterate over the specified configurations.
    for n_layers in n_layers_list:
        for model in models:
            key = (model.__name__, n_layers, n_qubits)
            if key in existing_entries:
                print(
                    f"Skipping {model.__name__} with {n_layers} layers and {n_qubits} qubits (already exists)."
                )
                continue
            qvc = model(n_layers=n_layers, random_state=0)
            haar, meyer_wallach = qvc.measures(n_qubits=n_qubits, samples=samples)
            print(
                f"{model.__name__} with {n_layers} layers and {n_qubits} qubits: "
                f"Haar = {haar}, Meyer-Wallach = {meyer_wallach}"
            )
            writer.writerow([model.__name__, n_layers, n_qubits, haar, meyer_wallach])
