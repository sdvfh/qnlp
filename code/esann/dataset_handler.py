from urllib.request import urlopen
from .utils import get_path, GLOBAL_SEED
import tarfile
import urllib.request
import shutil
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
import random
import numpy as np
from sklearn.model_selection import train_test_split


class DatasetProcessor:
    _folder_name = None

    def __init__(self):
        self._path = get_path()
        self._df = pd.DataFrame()

        self._path["folder_root_dataset"] = self._path["data"] / self._folder_name
        self._path["folder_root_dataset"].mkdir(parents=True, exist_ok=True)

        self._path["folder_sentence_embeddings"] = self._path["folder_root_dataset"] / "sentence_embeddings"
        self._path["folder_sentence_embeddings"].mkdir(parents=True, exist_ok=True)
        self._path["x_data"] = (
                self._path["folder_root_dataset"] / f"x_data.npy")
        self._path["y_data"] = (
                self._path["folder_root_dataset"] / f"y_data.npy")

    def download(self):
        raise NotImplementedError()

    def load(self):
        raise NotImplementedError()

    def process(self):
        if self.df_processing_done():
            self._df = pd.DataFrame()

            with open(self._path["x_data"], "rb") as file:
                x_data = np.load(file)
            with open(self._path["y_data"], "rb") as file:
                y_data = np.load(file)
            return x_data, y_data

        torch.use_deterministic_algorithms(True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = BertTokenizer.from_pretrained("bert-large-uncased", use_fast=True, device=device)
        model = BertModel.from_pretrained("bert-large-uncased").to(device)

        for idx, line in self._df.iterrows():
            file_path = self._path["folder_sentence_embeddings"] / f"{idx}.npy"
            if file_path.exists():
                continue

            random.seed(GLOBAL_SEED)
            np.random.seed(GLOBAL_SEED)
            torch.manual_seed(GLOBAL_SEED)

            text_tokenized = tokenizer(
                line["sentence"],
                # TODO: study the max_length parameter
                max_length=64,
                add_special_tokens=True,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                outputs = model(**text_tokenized)
            sentence_embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            with open(file_path, "wb") as file:
                np.save(file, sentence_embedding)
            print(f"Processed {idx} of {len(self._df)}")

        embeddings = []
        for idx, line in self._df.iterrows():
            file_path = self._path["folder_sentence_embeddings"] / f"{idx}.npy"
            with open(file_path, "rb") as file:
                embeddings.append(np.load(file))

        embeddings = np.row_stack(embeddings)
        with open(self._path["x_data"], "wb") as file:
            np.save(file, embeddings)

        with open(self._path["y_data"], "wb") as file:
            np.save(file, self._df["label"].values)

        self._df = pd.DataFrame()
        return embeddings, self._df["label"].values

    def df_processing_done(self):
        return self._path["x_data"].exists() and self._path["y_data"].exists()

class MovieReviewProcessor(DatasetProcessor):
    _folder_name = "movie_review"
    _url = "https://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz"

    def __init__(self):
        super().__init__()
        self._path["file_path_compressed"] = self._path["folder_root_dataset"] / "review_polarity.tar.gz"
        self._path["folder_path_uncompressed"] = self._path["folder_root_dataset"] / "review_polarity"
        self._path["folder_dataset"] = self._path["folder_path_uncompressed"] / "txt_sentoken"
        self._path["file_all_sentences"] = self._path["folder_root_dataset"] / "all_sentences.snappy.parquet"

    def download(self):
        if not self._path["file_path_compressed"].exists():
            # From https://stackoverflow.com/questions/7243750/download-file-from-web-in-python-3
            with urllib.request.urlopen(self._url) as response, open(self._path["file_path_compressed"], 'wb') as file:
                shutil.copyfileobj(response, file)

        if not self._path["folder_path_uncompressed"].exists():
            with tarfile.open(self._path["file_path_compressed"], "r:gz") as tar:
                tar.extractall(self._path["folder_path_uncompressed"])

    def load(self):
        if not self._path["file_all_sentences"].exists():
            df = []
            for file in self._path["folder_dataset"].rglob("*.txt"):
                label = 1 if file.parts[-2] == "pos" else -1
                text = file.read_text().splitlines()
                for sentence in text:
                    df.append([sentence, label])
            self._df = pd.DataFrame(df, columns=["sentence", "label"])
            self._df.to_parquet(self._path["file_all_sentences"])
        else:
            self._df = pd.read_parquet(self._path["file_all_sentences"])


class DatasetHandler:
    datasets = {"MR": MovieReviewProcessor}

    def __init__(self):
        self.dataset = {"original": {"x": None, "y": None}}

    def load(self, dataset_initials: str):
        df_processor = self.datasets[dataset_initials]()
        if not df_processor.df_processing_done():
            df_processor.download()
            df_processor.load()
        self.dataset["original"]["x"], self.dataset["original"]["y"] = df_processor.process()

    def split_train_valid_test(self, seed: int):
        x_train_valid, x_test, y_train_valid, y_test = train_test_split(
            self.dataset["original"]["x"],
            self.dataset["original"]["y"],
            test_size=1 / 4,
            random_state=seed,
            stratify=self.dataset["original"]["y"],
        )
        x_train, x_valid, y_train, y_valid = train_test_split(
            x_train_valid,
            y_train_valid,
            test_size=1 / 3,
            random_state=seed,
            stratify=y_train_valid,
        )
        datasets = {
            "train": {"x": x_train, "y": y_train},
            "valid": {"x": x_valid, "y": y_valid},
            "test": {"x": x_test, "y": y_test}
        }
        self.dataset.update(datasets)
