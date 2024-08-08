import logging
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from pytreebank import load_sst
from transformers import BertModel, BertTokenizer


def transform_binary_label(label):
    if label == 2:
        return -1
    elif label < 2:
        return 0
    else:
        return 1


class QNLP:
    def __init__(self, testing=False):
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(42)
        random.seed(42)
        np.random.seed(42)

        self.testing = testing
        self._path = None

        if self.testing:
            self._repetitions = 3
        else:
            self._repetitions = 30
        self._define_log()
        self._define_path()

    def run(self):
        self._process_data()
        self._run_models()
        self._agg_metrics()
        print("End.")

    def _define_log(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.FileHandler("log.txt"), logging.StreamHandler()],
        )

    def _define_path(self):
        self._path = {"root": Path(__file__).parent.parent.parent.resolve()}
        self._path["data"] = self._path["root"] / "data"

    def _process_data(self):
        logging.info("Getting data.")
        df = load_sst(self._path["data"] / "sst")

        logging.info("Processing data.")
        self._path["sst_processed"] = self._path["data"] / "sst_processed"

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = BertTokenizer.from_pretrained(
            "bert-large-uncased", use_fast=True, device=device
        )
        bert = BertModel.from_pretrained("bert-large-uncased").to(device)

        for seed in range(self._repetitions):
            for dataset_name in df:
                dataset = df[dataset_name]
                dataset_path = self._path["sst_processed"] / str(seed) / dataset_name
                if (dataset_path / "values.pth").exists():
                    continue
                dataset_path.mkdir(parents=True, exist_ok=True)
                for i, sentence in enumerate(dataset):
                    sentence_path = dataset_path / f"{i}.pth"
                    if sentence_path.exists():
                        continue
                    else:
                        self._load_state()
                    if self.testing and i == 25:
                        break
                    logging.info(
                        "Seed: %d, Dataset: %s, Sentence number: %d",
                        seed,
                        dataset_name,
                        i,
                    )
                    label, sentence = sentence.to_labeled_lines()[0]
                    label = transform_binary_label(label)
                    embedding = self._get_embedding(tokenizer, bert, device, sentence)
                    self._save_embedding(sentence_path, sentence, label, embedding)
                    self._save_state()
                self._agg_dataset(dataset_path)

    def _get_embedding(self, tokenizer, bert, device, sentence):
        sentence_tokenized = tokenizer(
            sentence,
            max_length=128,
            add_special_tokens=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = bert(**sentence_tokenized)
        embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        return embedding

    def _save_embedding(self, sentence_path, sentence, label, embedding):
        sentence_info = {
            "sentence": sentence,
            "label": label,
            "embedding": embedding,
        }
        torch.save(sentence_info, sentence_path)

    def _save_state(self):
        states = {
            "cpu_rng_state": torch.get_rng_state(),
            "gpu_rng_state": torch.cuda.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "py_rng_state": random.getstate(),
        }
        torch.save(states, self._path["data"] / "states.pth")

    def _load_state(self):
        if not (self._path["data"] / "states.pth").exists():
            return
        states = torch.load(self._path["data"] / "states.pth")
        torch.set_rng_state(states["cpu_rng_state"])
        torch.cuda.set_rng_state(states["gpu_rng_state"])
        np.random.set_state(states["numpy_rng_state"])
        random.setstate(states["py_rng_state"])

    def _agg_dataset(self, dataset_path):
        sentences = dataset_path.glob("*.pth")
        sentences = pd.DataFrame([torch.load(sentence) for sentence in sentences])
        to_delete = sentences[sentences["label"] == -1]
        logging.info(
            "Deleting %d sentences in %s dataset.",
            len(to_delete),
            dataset_path.parts[-1],
        )
        sentences = sentences.drop(to_delete.index)
        labels = sentences["label"].values
        embeddings = np.row_stack(sentences["embedding"].values)
        values = {
            "embeddings": embeddings,
            "labels": labels,
        }
        shutil.rmtree(dataset_path)
        dataset_path.mkdir()
        torch.save(values, dataset_path / "values.pth")

    def _run_models(self):
        pass

    def _agg_metrics(self):
        pass
