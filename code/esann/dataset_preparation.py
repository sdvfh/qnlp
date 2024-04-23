from urllib.request import urlopen
from .utils import get_path, GLOBAL_SEED
import tarfile
import urllib.request
import shutil
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
import pickle
import random
import numpy as np

class DatasetProcessor:
    _folder_name = None

    def __init__(self):
        self._path = get_path()
        self._df = pd.DataFrame()

        self._path["folder_root_dataset"] = self._path["data"] / self._folder_name
        self._path["folder_root_dataset"].mkdir(parents=True, exist_ok=True)

        self._path["folder_sentence_embeddings"] = self._path["folder_root_dataset"] / "sentence_embeddings"
        self._path["folder_sentence_embeddings"].mkdir(parents=True, exist_ok=True)
        self._path["file_unified_sentence_embeddings"] = (
                self._path["folder_sentence_embeddings"] / f"{self._folder_name}.pkl")

    def download(self):
        raise NotImplementedError()

    def load(self):
        raise NotImplementedError()

    def process(self):
        if self._path["file_unified_sentence_embeddings"].exists():
            return
        torch.use_deterministic_algorithms(True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = BertTokenizer.from_pretrained("bert-large-uncased", use_fast=True, device=device)
        model = BertModel.from_pretrained("bert-large-uncased").to(device)

        for idx, line in self._df.iterrows():
            file_path = self._path["folder_sentence_embeddings"] / f"{idx}.pkl"
            if file_path.exists():
                continue

            random.seed(0)
            np.random.seed(0)
            torch.manual_seed(0)
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
                pickle.dump(sentence_embedding, file)
            print(f"Processed {idx} of {len(self._df)}")

    def create_embeddings(self):
        raise NotImplementedError()


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


class DatasetPreparation:
    _options = {"MR": MovieReviewProcessor}

    def load(self, dataset_initials: str):
        df_processor = self._options[dataset_initials]()
        df_processor.download()
        df_processor.load()
        df_processor.process()
        df_processor.create_embeddings()
