from .dataset_handler import DatasetHandler
from .model_handler import ModelHandler
from .utils import N_REPETITIONS

class Orchestrator:
    def __init__(self):
        self.df_handler = None

    def run(self):
        self.df_handler = DatasetHandler()
        self.model_handler = ModelHandler()
        for dataset in self.df_handler.datasets:
            print(f"Dataset: {dataset}")
            self.df_handler.load(dataset)
            for seed in range(N_REPETITIONS):
                print(f"Seed: {seed}")
                self.df_handler.split_train_valid_test(seed)
