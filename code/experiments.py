from pathlib import Path

from custom_classifier import AnsatzSingleRotX
from pennylane import numpy as np
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, VotingClassifier
from utils import get_embeddings, read_dataset

# Constants and Hyperparameters
LEVELS = ["easy", "medium", "hard"]
TYPES_DATASETS = ["train", "test"]
EPOCHS = 100
BATCH_SIZE = 5
n_layers = 1
seed = 0

# Paths and random number generator
paths = {"data": Path(__file__).parent.parent / "data"}

# Load and embed dataset for each level
dfs = {level: read_dataset(paths["data"], level) for level in LEVELS}
dfs = get_embeddings(dfs, LEVELS, TYPES_DATASETS)

# Use the "easy" level for training and testing in this example
x_train = np.array(dfs["easy"]["train"]["embeddings"], requires_grad=False)
y_train = np.array(dfs["easy"]["train"]["targets"], requires_grad=False)
x_test = np.array(dfs["easy"]["test"]["embeddings"], requires_grad=False)
y_test = np.array(dfs["easy"]["test"]["targets"], requires_grad=False)

qvc1 = AnsatzSingleRotX(
    n_layers=n_layers, max_iter=EPOCHS, batch_size=BATCH_SIZE, random_state=seed
)
# qvc2 = Ansatz1(n_layers=10, max_iter=1, batch_size=20, random_state=0)
# ensemble = AdaBoostClassifier(estimator=qvc1, n_estimators=10, random_state=0)
# ensemble = BaggingClassifier(
#     estimator=qvc1, n_estimators=2, random_state=0, max_features=0.8, max_samples=0.8
# )
# estimators = [("qvc1", qvc1), ("qvc2", qvc2)]
# ensemble = VotingClassifier(estimators=estimators, voting="soft")
# ensemble = VotingClassifier(estimators=estimators, voting="hard")
ensemble = qvc1
ensemble.fit(x_train, y_train)
y_pred = ensemble.predict(x_test)
print(y_pred)
