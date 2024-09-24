import logging
from itertools import combinations
from pathlib import Path

import pandas as pd
import torch
from pennylane import numpy as np
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

        self._n_repetitions = 30
        self._df = {}
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
        self._logging = logging.getLogger()

    def _define_path(self):
        self._path = {"root": Path(__file__).parent.parent.parent.resolve()}
        self._path["data"] = self._path["root"] / "data"
        self._path["models"] = self._path["data"] / "models"
        self._path["models"].mkdir(parents=True, exist_ok=True)
        self._path["sst_processed"] = self._path["data"] / "sst_processed"
        self._path["sst_processed"].mkdir(parents=True, exist_ok=True)
        self._path["hybrid"] = self._path["data"] / "hybrid"
        self._path["hybrid"].mkdir(parents=True, exist_ok=True)

    def _process_data(self):
        bert_model = "bert-base-cased"
        if (self._path["sst_processed"] / "sst_processed.pth").exists():
            self._df = torch.load(self._path["sst_processed"] / "sst_processed.pth")
            return
        self._logging.info("Getting data")
        sst = load_sst(self._path["data"] / "sst")
        df = {}

        self._logging.info("Processing data")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = BertTokenizer.from_pretrained(
            bert_model, use_fast=True, device=device
        )
        bert = BertModel.from_pretrained(bert_model).to(device)
        for dataset_name in sst:
            dataset = sst[dataset_name]
            self._logging.info("Dataset: %s", dataset_name)
            sentences, labels = self._get_sentences(dataset)
            embeddings = self._get_embeddings(tokenizer, bert, device, sentences)
            df[dataset_name] = {
                "embeddings": embeddings,
                "labels": labels,
            }
        self._save_dataset(df)
        self._df = df

    def _get_sentences(self, dataset):
        sentences = []
        labels = []
        for i, sentence in enumerate(dataset):
            if self.testing and i == 100:
                break
            label, sentence = sentence.to_labeled_lines()[0]
            label = transform_binary_label(label)
            labels.append(label)
            sentences.append(sentence)
        labels = np.array(labels)
        return sentences, labels

    def _get_embeddings(self, tokenizer, bert, device, sentences):
        sentence_tokenized = tokenizer(
            sentences,
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

    def _save_dataset(self, df):
        for dataset_name in df:
            dataset = df[dataset_name]
            self._logging.info(
                "The %s dataset has %d sentences", dataset_name, len(dataset["labels"])
            )
            to_remove = np.argwhere(dataset["labels"] == -1)
            dataset["labels"] = np.delete(dataset["labels"], to_remove)
            dataset["embeddings"] = np.delete(dataset["embeddings"], to_remove, axis=0)
            self._logging.info(
                "Deleting %d sentences in %s dataset", len(to_remove), dataset_name
            )
            self._logging.info(
                "The %s dataset has now %d sentences",
                dataset_name,
                len(dataset["labels"]),
            )

        scaler = MinMaxScaler(feature_range=(0, np.pi / 2))
        df["train"]["embeddings"] = scaler.fit_transform(df["train"]["embeddings"])
        df["dev"]["embeddings"] = scaler.transform(df["dev"]["embeddings"])
        df["test"]["embeddings"] = scaler.transform(df["test"]["embeddings"])
        torch.save(df, self._path["sst_processed"] / "sst_processed.pth")

    def _run_models(self):
        for model in models:
            model_path = self._path["models"] / f"{model}.pth"
            if model_path.exists():
                continue
            self._logging.info("Running model %s.", model)
            self._model = models[model](self._path, self._n_repetitions, self.testing)
            metrics = self._model.run(self._df)
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
