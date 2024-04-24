from .dataset_handler import DatasetHandler
from .metric_handler import MetricHandler
from .model_handler import ModelHandler
from .utils import N_REPETITIONS


class Orchestrator:
    def __init__(self):
        self.df_handler = None
        self.model_handler = None
        self.metric_handler = None

    def run(self):
        self.df_handler = DatasetHandler()
        self.model_handler = ModelHandler()
        self.metric_handler = MetricHandler()
        for dataset_name in self.df_handler.datasets:
            for seed in range(N_REPETITIONS):
                for model_name in self.model_handler.models:
                    self.metric_handler.load(dataset_name, model_name, seed)
                    if self.metric_handler.metric_exists():
                        continue
                    self.df_handler.load(dataset_name, seed)
                    self.df_handler.split_train_valid_test()
                    print(f"Dataset: {dataset_name}")
                    print(f"Seed: {seed}")
                    print(f"Model: {model_name}")
                    self.model_handler.load(model_name, self.df_handler.dataset, seed)
                    pred = self.model_handler.run()
                    metrics = self.metric_handler.get_metrics(
                        pred, self.df_handler.dataset["test"]["y"]
                    )
                    self.metric_handler.save_metrics(metrics)
