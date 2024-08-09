import logging
import random
import shutil
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytreebank import load_sst
from scipy.stats import wilcoxon
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import MinMaxScaler
from transformers import BertModel, BertTokenizer

from .models import models


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

        self.testing = testing
        self._path = None

        self._repetitions = 30
        self._df = {repetition: {} for repetition in range(self._repetitions)}
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
        self._path["models"] = self._path["data"] / "models"
        self._path["models"].mkdir(parents=True, exist_ok=True)

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
            self._set_seed(seed)
            scaler = MinMaxScaler(feature_range=(0, np.pi / 2))
            for dataset_name in df:
                dataset = df[dataset_name]
                dataset_path = self._path["sst_processed"] / str(seed) / dataset_name
                values_dataset_path = dataset_path / "values.pth"
                if values_dataset_path.exists():
                    self._df[seed][dataset_name] = torch.load(values_dataset_path)
                    continue
                dataset_path.mkdir(parents=True, exist_ok=True)
                for i, sentence in enumerate(dataset):
                    sentence_path = dataset_path / f"{i}.pth"
                    if sentence_path.exists():
                        continue
                    else:
                        self._load_state()
                    if self.testing and i == 100:
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
                self._agg_dataset(seed, scaler, dataset_path, values_dataset_path)

    def _set_seed(self, seed):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

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

    def _agg_dataset(self, seed, scaler, dataset_path, values_dataset_path):
        sentences = dataset_path.glob("*.pth")
        dataset_name = dataset_path.parts[-1]
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
        if dataset_name == "train":
            embeddings = scaler.fit_transform(embeddings)
        else:
            embeddings = scaler.transform(embeddings)
        values = {
            "embeddings": embeddings,
            "labels": labels,
        }
        shutil.rmtree(dataset_path)
        dataset_path.mkdir()
        self._df[seed][dataset_name] = values
        torch.save(values, values_dataset_path)

    def _run_models(self):
        self._load_state()
        for model in models:
            model_path = self._path["models"] / f"{model}.pth"
            if model_path.exists():
                continue
            logging.info("Running model %s.", model)
            self._model = models[model]()
            metrics = self._model.run(self._df)
            self._save_state()
            self._save_metrics(model, model_path, metrics)

    @staticmethod
    def _save_metrics(model, model_path, metrics):
        results = {}
        for seed, y_test, y_pred in metrics:
            results[seed] = {
                "accuracy": balanced_accuracy_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
            }
        results = pd.DataFrame(results).T
        results["model"] = model
        torch.save(results, model_path)

    def _agg_metrics(self):
        agg_metrics_path = self._path["data"] / "agg_metrics.csv"
        if agg_metrics_path.exists():
            return
        metrics = {}
        for model in self._path["models"].glob("*.pth"):
            model_name = model.stem
            metrics[model_name] = torch.load(model)
        metrics = pd.concat(metrics).sort_index()
        agg_metrics = metrics.groupby("model").agg(["mean", "std"])
        agg_metrics.to_csv(agg_metrics_path)
        self._run_wilcoxon(metrics)

    def _run_wilcoxon(self, metrics):
        wilcoxon_results_path = self._path["data"] / "wilcoxon_results.csv"
        models = metrics["model"].unique().tolist()
        metrics_names = metrics.columns.tolist()
        metrics_names.remove("model")
        wilcoxon_results = []
        for model_1, model_2 in combinations(models, 2):
            for metric in metrics_names:
                model_1_metrics = metrics.loc[metrics["model"] == model_1, metric]
                model_2_metrics = metrics.loc[metrics["model"] == model_2, metric]
                wilcoxon_value = wilcoxon(
                    model_1_metrics, model_2_metrics, zero_method="zsplit"
                )
                wilcoxon_results.append(
                    {
                        "model_1": model_1,
                        "model_2": model_2,
                        "metric": metric,
                        "wilcoxon_value": wilcoxon_value.statistic,
                        "wilcoxon_pvalue": wilcoxon_value.pvalue,
                    }
                )
        wilcoxon_results = pd.DataFrame(wilcoxon_results)
        wilcoxon_results.to_csv(wilcoxon_results_path)
