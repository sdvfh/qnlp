from pennylane import numpy as np
from sentence_transformers import SentenceTransformer


class Dataset:
    def __init__(self, paths, *args, **kwargs):
        self.paths = paths
        self.dataset_name = None
        self.datasets = {}

    def get_embeddings(self, model_transformer, n_features):
        model = SentenceTransformer(
            model_transformer, truncate_dim=n_features, trust_remote_code=True
        )
        for dataset in ["train", "test"]:
            self.datasets[dataset]["embeddings"] = model.encode(
                self.datasets[dataset]["data"]
            )

    def get_sets(self):
        x_train = np.array(self.datasets["train"]["embeddings"], requires_grad=False)
        y_train = np.array(self.datasets["train"]["targets"], requires_grad=False)
        x_test = np.array(self.datasets["test"]["embeddings"], requires_grad=False)
        y_test = np.array(self.datasets["test"]["targets"], requires_grad=False)
        return x_train, y_train, x_test, y_test

    def read_dataset(self):
        for dataset in ["train", "test"]:
            file_path = self.paths["data"] / self.dataset_name / f"{dataset}.txt"
            data, targets = self.read_files(file_path)
            self.datasets[dataset] = {"data": data, "targets": targets}

    @staticmethod
    def read_files(filename):
        if not filename.exists():
            raise FileNotFoundError(f"The file {filename} does not exist.")

        try:
            with open(filename, "r") as file:
                lines = file.readlines()

            data_targets = []
            for line in lines:
                stripped_line: str = line.strip()
                if not stripped_line:  # Skip empty lines
                    continue
                if not stripped_line[0].isdigit():
                    raise ValueError(
                        f"The line does not start with an integer label: {line}"
                    )
                label: int = int(stripped_line[0])
                text: str = stripped_line[1:].strip()
                data_targets.append((text, label))

            if not data_targets:
                raise ValueError(f"No valid data found in file {filename}.")

            data, targets = zip(*data_targets, strict=True)
        except IOError as e:
            raise IOError(
                f"An error occurred while reading the file {filename}: {e}"
            ) from None
        except ValueError as e:
            raise ValueError(f"Error processing file {filename}: {e}") from None

        return list(data), np.array(targets)


class ChatGPTDataset(Dataset):
    def __init__(self, paths, level):
        super().__init__(paths, level)
        self.dataset_name = f"chatgpt_{level}"


class SST(Dataset):
    def __init__(self, paths):
        super().__init__(paths)
        self.dataset_name = "sst"


def read_dataset(dataset, model_transformer, n_features, paths):
    dataset_factory = {
        "chatgpt_easy": lambda: ChatGPTDataset(paths, level="easy"),
        "chatgpt_medium": lambda: ChatGPTDataset(paths, level="medium"),
        "chatgpt_hard": lambda: ChatGPTDataset(paths, level="hard"),
        "sst": lambda: SST(paths),
    }

    if dataset not in dataset_factory:
        raise ValueError(f"Dataset {dataset!r} not found.")

    dataset_class = dataset_factory[dataset]()
    dataset_class.read_dataset()
    dataset_class.get_embeddings(model_transformer, n_features)
    return dataset_class.get_sets()
