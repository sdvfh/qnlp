import gzip
import pickle
from itertools import product
from pathlib import Path
from time import time
from typing import List, Tuple

import numpy as np
import torch
from lambeq import (
    AtomicType,
    BobcatParser,
    Dataset,
    IQPAnsatz,
    PennyLaneModel,
    RemoveCupsRewriter,
    Sim4Ansatz,
    Sim14Ansatz,
    Sim15Ansatz,
    StronglyEntanglingAnsatz,
    UnifyCodomainRewriter,
)
from lambeq.backend.tensor import Diagram
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class MyModel(PennyLaneModel):
    def forward(self, x: list[Diagram]) -> torch.Tensor:
        evaluated_circuits = self.get_diagram_output(x)
        return evaluated_circuits.flatten(start_dim=1)[:, 0]


class Experiment:
    def __init__(
        self,
        level,
        ansatz,
        dim_noun,
        dim_sentence,
        dim_prepositional_phrase,
        n_layer,
        seed,
    ):
        self.level = level
        self.ansatz = ansatz
        self.dim_noun = dim_noun
        self.dim_sentence = dim_sentence
        self.dim_prepositional_phrase = dim_prepositional_phrase
        self.n_layer = n_layer
        self.seed = seed

        self._datasets = {}
        self._model = None
        self._other_infos = {
            "level": level,
            "ansatz": ansatz.__name__,
            "dim_noun": dim_noun,
            "dim_sentence": dim_sentence,
            "dim_prepositional_phrase": dim_prepositional_phrase,
            "n_layer": n_layer,
            "seed": seed,
            "device": None,
            "n_qubits_circuits": None,
            "epoch_infos": None,
            "y_pred": None,
            "metrics": None,
            "symbols": None,
        }
        self._other_infos["experiment_id"] = (
            f"{level}_{ansatz.__name__}_"
            f"{dim_noun}_{dim_sentence}_"
            f"{dim_prepositional_phrase}_"
            f"{n_layer}_{seed}"
        )

    def run(self):
        if Path(f"../results/{self._other_infos['experiment_id']}.pkl.gz").exists():
            print(f"Experiment {self._other_infos['experiment_id']} already exists.")
            return
        self._print_experiment_info()
        self._set_seed()
        self._read_data()
        self._sentences2diagrams()
        self._transform_diagrams()
        self._create_circuits()
        try:
            self._create_model()
        except ValueError as e:
            print(f"Error: {e}")
            return
        self._train_model()
        self._save_results()

    def _print_experiment_info(self):
        print(
            f"Dataset level: {self.level} | "
            f"Ansatz name: {self.ansatz.__name__} | "
            f"Dimension of noun: {self.dim_noun} | "
            f"Dimension of sentence: {self.dim_sentence} | "
            f"Dimension of prepositional phrase: {self.dim_prepositional_phrase} | "
            f"Number of layers: {self.n_layer} | "
            f"Seed: {self.seed}"
        )

    def _set_seed(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def _read_data(self):
        for dataset in ["train", "test"]:
            data, targets = self._read_files(
                f"../data/chatgpt/{self.level}/{dataset}.txt"
            )
            self._datasets[dataset] = {"data": data, "targets": targets}

    @staticmethod
    def _read_files(filename: str) -> Tuple[List[str], np.ndarray]:
        """Read data from a file and return sentences and targets."""
        data, targets = [], []
        with open(filename, "r") as file:
            for line in file:
                label = int(line[0])
                sentence = line[1:].strip()
                data.append(sentence)
                targets.append(label)
        return data, np.array(targets)

    def _sentences2diagrams(self):
        parser = BobcatParser(verbose="text")
        for dataset in ["train", "test"]:
            self._datasets[dataset]["diagrams"] = parser.sentences2diagrams(
                self._datasets[dataset]["data"]
            )

    def _transform_diagrams(self):
        self._transform_diagrams_remove_cup()
        self._transform_diagrams_unify_codomain()

    def _transform_diagrams_remove_cup(self):
        remove_cups = RemoveCupsRewriter()
        for dataset in ["train", "test"]:
            self._datasets[dataset]["diagrams"] = [
                remove_cups(diagram).normal_form()
                for diagram in self._datasets[dataset]["diagrams"]
            ]

    def _transform_diagrams_unify_codomain(self):
        unify_diagram = UnifyCodomainRewriter()
        for dataset in ["train", "test"]:
            self._datasets[dataset]["diagrams"] = [
                unify_diagram(diagram)
                for diagram in self._datasets[dataset]["diagrams"]
            ]

    def _create_circuits(self):
        for dataset in ["train", "test"]:
            self._datasets[dataset]["circuits"] = [
                self.ansatz(
                    {
                        AtomicType.NOUN: self.dim_noun,
                        AtomicType.SENTENCE: self.dim_sentence,
                        AtomicType.PREPOSITIONAL_PHRASE: self.dim_prepositional_phrase,
                    },
                    n_layers=self.n_layer,
                    n_single_qubit_params=3,
                )(diagram)
                for diagram in self._datasets[dataset]["diagrams"]
            ]

    def _create_model(self):
        all_circuits = (
            self._datasets["train"]["circuits"] + self._datasets["test"]["circuits"]
        )
        n_qubits_circuits = [circuit.to_tk().n_qubits for circuit in all_circuits]
        if max(n_qubits_circuits) > 28:
            raise ValueError("Circuits have more than 28 qubits.")

        if np.mean(n_qubits_circuits) > 12:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = "cpu"

        print(f"Device: {device}")
        self._other_infos["device"] = device
        self._other_infos["n_qubits_circuits"] = n_qubits_circuits

        self._model = MyModel.from_diagrams(
            all_circuits, probabilities=True, normalize=True
        )
        self._model.initialise_weights()
        self._model = self._model.double().to(device)

    def _train_model(self):
        x_train = self._datasets["train"]["circuits"]
        y_train = self._datasets["train"]["targets"]
        x_test = self._datasets["test"]["circuits"]

        train_dataset = Dataset(x_train, y_train, batch_size=BATCH_SIZE)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=0.1)

        epoch_infos = {}
        for i in range(EPOCHS):
            batch_infos = {}
            epoch_loss = 0
            epoch_old_time = time()
            for o, (circuits, labels) in enumerate(train_dataset):
                batch_old_time = time()
                total_batches = len(train_dataset) // BATCH_SIZE
                optimizer.zero_grad()
                predicted = self._model(circuits)
                labels = torch.DoubleTensor(labels).to(self._other_infos["device"])
                loss = torch.nn.functional.binary_cross_entropy(
                    torch.flatten(predicted), labels
                )
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
                batch_final_time = time()
                current_batch = f"{o + 1}/{total_batches}"
                print(
                    f"Epoch: {i:^3} "
                    f"| Batch: {current_batch:^4} "
                    f"| Loss: {loss.item():^6.2f} "
                    f"| Time: {batch_final_time - batch_old_time:^6.2f}"
                )
                batch_info = {
                    "Loss": loss.item(),
                    "Processing time": batch_final_time - batch_old_time,
                }
                batch_infos[o] = batch_info
            epoch_infos[i] = batch_infos
            epoch_final_time = time()
            print(
                f"Epoch: {i + 1:^3} "
                f"| Train loss: {epoch_loss:^6.2f} "
                f"| Time: {epoch_final_time - epoch_old_time:^6.2f}"
            )
        self._other_infos["epoch_infos"] = epoch_infos
        self._other_infos["y_pred"] = self._model(x_test).detach().cpu().numpy()

    def _save_results(self):
        y_test = self._datasets["test"]["targets"]
        y_pred = self._other_infos["y_pred"]
        y_pred_round = (y_pred > 0.5).astype(int)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred_round),
            "f1": f1_score(y_test, y_pred_round),
            "precision": precision_score(y_test, y_pred_round),
            "recall": recall_score(y_test, y_pred_round),
            "roc_auc": roc_auc_score(y_test, y_pred),
        }
        print(f"Metrics: {metrics}")
        self._other_infos["metrics"] = metrics

        symbols = {
            symbol: float(value.detach().cpu().numpy())
            for (symbol, value) in self._model.symbol_weight_map.items()
        }
        self._other_infos["symbols"] = symbols

        with gzip.open(
            f"../results/{self._other_infos['experiment_id']}.pkl.gz", "wb"
        ) as file:
            pickle.dump(self._other_infos, file)


BATCH_SIZE = 20
EPOCHS = 200
N_REPETITIONS = 30

levels = ["easy", "medium", "hard"]
anstaze = [
    IQPAnsatz,
    StronglyEntanglingAnsatz,
    Sim4Ansatz,
    Sim14Ansatz,
    Sim15Ansatz,
]
dim_noun = dim_sentence = dim_prepositional_phrase = list(range(1, 6))
n_layers = [1, 2, 4, 8, 16, 32, 64]

experiments = product(
    levels, anstaze, dim_noun, dim_sentence, dim_prepositional_phrase, n_layers
)

for (
    level,
    ansatz,
    dim_noun,
    dim_sentence,
    dim_prepositional_phrase,
    n_layer,
) in experiments:
    for seed in range(N_REPETITIONS):
        experiment = Experiment(
            level=level,
            ansatz=ansatz,
            dim_noun=dim_noun,
            dim_sentence=dim_sentence,
            dim_prepositional_phrase=dim_prepositional_phrase,
            n_layer=n_layer,
            seed=seed,
        )
        experiment.run()
