import shutil
import tarfile
import urllib.request

import nltk
import numpy as np
import pandas as pd
import torch
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from transformers import BertModel, BertTokenizer

from .utils import N_FEATURES, WORD_COUNT_MIN, get_path


class DatasetProcessor:
    _folder_name = None

    def __init__(self, seed):
        self.seed = seed
        self._path = get_path()
        self._df = pd.DataFrame()

        self._path["folder_root_dataset"] = self._path["data"] / self._folder_name
        self._path["folder_root_dataset"].mkdir(parents=True, exist_ok=True)

        self._path["folder_root_dataset_seed"] = (
            self._path["data"] / self._folder_name / "seeds" / str(self.seed)
        )
        self._path["folder_root_dataset_seed"].mkdir(parents=True, exist_ok=True)

        self._path["folder_sentence_embeddings"] = (
            self._path["folder_root_dataset_seed"] / "sentence_embeddings"
        )
        self._path["folder_sentence_embeddings"].mkdir(parents=True, exist_ok=True)
        self._path["x_data"] = self._path["folder_root_dataset_seed"] / "x_data.npy"
        self._path["y_data"] = self._path["folder_root_dataset_seed"] / "y_data.npy"

    def download(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

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
        tokenizer = BertTokenizer.from_pretrained(
            "bert-large-uncased", use_fast=True, device=device
        )
        model = BertModel.from_pretrained("bert-large-uncased").to(device)

        for idx, line in self._df.iterrows():
            file_path = self._path["folder_sentence_embeddings"] / f"{idx}.npy"
            if file_path.exists():
                continue

            torch.manual_seed(self.seed)
            text_tokenized = tokenizer(
                line["sentence"],
                max_length=256,
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
            if idx % 1000 == 0:
                print(f"Processed {idx} of {len(self._df)}")

        embeddings = []
        for idx, _ in self._df.iterrows():
            file_path = self._path["folder_sentence_embeddings"] / f"{idx}.npy"
            with open(file_path, "rb") as file:
                embeddings.append(np.load(file))

        embeddings = np.row_stack(embeddings)
        with open(self._path["x_data"], "wb") as file:
            np.save(file, embeddings)

        labels = self._df["label"].values
        with open(self._path["y_data"], "wb") as file:
            np.save(file, labels)

        self._df = pd.DataFrame()
        return embeddings, labels

    def df_processing_done(self):
        return self._path["x_data"].exists() and self._path["y_data"].exists()

    def get_dataset(self):
        return {"original": {"x": self._df["sentence"], "y": self._df["label"]}}


class MovieReviewProcessor(DatasetProcessor):
    _folder_name = "movie_review"
    _url = "https://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._path["file_path_compressed"] = (
            self._path["folder_root_dataset"] / "review_polarity.tar.gz"
        )
        self._path["folder_path_uncompressed"] = (
            self._path["folder_root_dataset"] / "review_polarity"
        )
        self._path["folder_dataset"] = (
            self._path["folder_path_uncompressed"] / "txt_sentoken"
        )
        self._path["file_all_sentences"] = (
            self._path["folder_root_dataset"] / "all_sentences.snappy.parquet"
        )

    def download(self):
        if not self._path["file_path_compressed"].exists():
            # From https://stackoverflow.com/questions/7243750/download-file-from-web-in-python-3
            with urllib.request.urlopen(self._url) as response, open(
                self._path["file_path_compressed"], "wb"
            ) as file:
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
            self._df["word_count"] = (
                self._df["sentence"]
                .str.replace(r"[^\w\s]", "", regex=True)
                .str.split()
                .str.len()
            )
            self._df = self._df[self._df["word_count"] >= WORD_COUNT_MIN].reset_index(
                drop=True
            )
            self._df.to_parquet(self._path["file_all_sentences"])
        else:
            self._df = pd.read_parquet(self._path["file_all_sentences"])


class LambeqProcessor(DatasetProcessor):
    _folder_name = "lambeq"
    _url = "https://raw.githubusercontent.com/CQCL/lambeq/main/docs/examples/datasets"
    _files_names = {
        "train": "mc_train_data.txt",
        "dev": "mc_test_data.txt",
        "test": "mc_dev_data.txt",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def download(self):
        self._path["dataset_paths"] = {}
        for dataset_type, file_name in self._files_names.items():
            remote_file_path = self._url + "/" + file_name
            local_file_path = self._path["folder_root_dataset"] / file_name
            self._path["dataset_paths"][dataset_type] = local_file_path
            if not local_file_path.exists():
                with urllib.request.urlopen(remote_file_path) as response, open(
                    local_file_path, "wb"
                ) as file:
                    shutil.copyfileobj(response, file)

    def load(self):
        datasets = {}
        for dataset_type, file_path in self._path["dataset_paths"].items():
            labels, sentences = self.read_data(file_path)
            datasets[dataset_type] = pd.DataFrame(
                {"sentence": sentences, "label": labels}
            )
        self._df = datasets

    def process(self):
        nltk.download("wordnet", quiet=True)
        nltk.download("punkt", quiet=True)
        wnl = WordNetLemmatizer()
        for dataset_type in self._df:
            self._df[dataset_type]["words"] = (
                self._df[dataset_type]["sentence"]
                .str.replace(r"[^\w\s]", "", regex=True)
                .str.split(" ")
            )
            self._df[dataset_type]["words"] = self._df[dataset_type]["words"].apply(
                lambda x: [wnl.lemmatize(word) for word in x]
            )
            self._df[dataset_type]["sentence"] = self._df[dataset_type]["words"].apply(
                lambda x: " ".join(x)
            )

        train = self._df["train"]
        test = pd.concat([self._df["dev"], self._df["test"]])

        tfidf_transformer = TfidfVectorizer(
            stop_words="english", lowercase=True, max_features=16
        )
        x_train = tfidf_transformer.fit_transform(train["sentence"]).toarray()
        x_test = tfidf_transformer.transform(test["sentence"]).toarray()

        y_train = np.int64(train["label"].apply(lambda x: x[1]))
        y_test = np.int64(test["label"].apply(lambda x: x[1]))

        y_train = np.where(y_train == 1, 1, -1)
        y_test = np.where(y_test == 1, 1, -1)

        self._df = {
            "train": {"x": x_train, "y": y_train},
            "test": {"x": x_test, "y": y_test},
        }

    def get_dataset(self):
        return self._df

    @staticmethod
    def read_data(filename):
        labels, sentences = [], []
        with open(filename) as f:
            for line in f:
                t = float(line[0])
                labels.append([t, 1 - t])
                sentences.append(line[1:].strip())
        return labels, sentences


class DatasetHandler:
    datasets = {
        # "MR": MovieReviewProcessor,
        "Lambeq": LambeqProcessor
    }

    def __init__(self):
        self.dataset = {"original": {"x": None, "y": None}}
        self.seed = None

    def load(self, dataset_name: str, seed: int):
        self.seed = seed
        df_processor = self.datasets[dataset_name](self.seed)
        if not df_processor.df_processing_done():
            df_processor.download()
            df_processor.load()
            df_processor.process()
            self.dataset = df_processor.get_dataset()

    def split_train_test(self):
        if not isinstance(self.dataset, dict):
            x_train, x_test, y_train, y_test = train_test_split(
                self.dataset["original"]["x"],
                self.dataset["original"]["y"],
                test_size=1 / 4,
                random_state=self.seed,
                stratify=self.dataset["original"]["y"],
            )

            pca = PCA(n_components=N_FEATURES, random_state=self.seed)
            x_train = pca.fit_transform(x_train)
            x_test = pca.transform(x_test)

            datasets = {
                "train": {"x": x_train, "y": y_train},
                "test": {"x": x_test, "y": y_test},
            }
            self.dataset.update(datasets)
