from typing import List, Tuple

import numpy as np
import torch
from lambeq import (
    AtomicType,
    BobcatParser,
    Dataset,
    IQPAnsatz,
    PennyLaneModel,
    RemoveCupsRewriter,
)
from lambeq.backend.tensor import Diagram


class MyModel(PennyLaneModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x: list[Diagram]) -> torch.Tensor:
        evaluated_circuits = self.get_diagram_output(x)
        return evaluated_circuits[:, 1]


def read_data(filename: str) -> Tuple[List[str], np.ndarray]:
    """Read data from a file and return sentences and targets."""
    data, targets = [], []
    with open(filename, "r") as file:
        for line in file:
            label = int(line[0])
            sentence = line[1:].strip()
            # target = np.array([label, 1 - label], dtype=np.float32)
            data.append(sentence)
            targets.append(label)
    return data, np.array(targets)


def accuracy(circs, labels):
    predicted = model(circs)
    return (
        torch.round(torch.flatten(predicted)) == torch.DoubleTensor(labels)
    ).sum().item() / len(circs)


# set constants
BATCH_SIZE = 1
EPOCHS = 5

# Set random seed for reproducibility
np.random.seed(0)

# Read training and testing data
train_data, train_targets = read_data("../data/chatgpt/easy_train.txt")
test_data, test_targets = read_data("../data/chatgpt/easy_test.txt")

# Parse sentences into diagrams
parser = BobcatParser(device=0)
raw_train_diagrams = parser.sentences2diagrams(train_data)
raw_test_diagrams = parser.sentences2diagrams(test_data)

remove_cups = RemoveCupsRewriter()

train_diagrams = [remove_cups(diagram) for diagram in raw_train_diagrams]
test_diagrams = [remove_cups(diagram) for diagram in raw_test_diagrams]

info_layers = {}
# for n_layer in [1, 2, 4, 8, 16, 32, 64]:
for n_layer in [1]:
    # Create ansatz and circuits
    ansatz = IQPAnsatz(
        {
            AtomicType.NOUN: 1,
            AtomicType.SENTENCE: 1,
            AtomicType.PREPOSITIONAL_PHRASE: 1,
        },
        n_layers=n_layer,
        n_single_qubit_params=3,
    )
    train_circuits = [ansatz(diagram) for diagram in train_diagrams]
    test_circuits = [ansatz(diagram) for diagram in test_diagrams]

    # Build vocabulary
    # all_circuits = train_circuits + dev_circuits + test_circuits
    all_circuits = train_circuits + test_circuits

    # Initialise our model by passing in the diagrams, so that we have trainable parameters for each token
    model = MyModel.from_diagrams(all_circuits, probabilities=True, normalize=True)
    model.initialise_weights()
    model = model.double()

    # initialise datasets and optimizers as in PyTorch
    train_pair_dataset = Dataset(train_circuits, train_targets, batch_size=BATCH_SIZE)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    print()
    print(f"Training with {n_layer} layers")

    for i in range(EPOCHS):
        epoch_loss = 0
        for circuits, labels in train_pair_dataset:
            optimizer.zero_grad()
            predicted = model(circuits)
            # use BCELoss as our outputs are probabilities, and labels are binary
            loss = torch.nn.functional.binary_cross_entropy(
                torch.flatten(predicted), torch.DoubleTensor(labels)
            )
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

        # evaluate on dev set every 1 epoch
        # dev_acc = accuracy(dev_circuits, dev_targets)

        print(f"Epoch: {i:^3} | Train loss: {epoch_loss:.2f^6}")

    test_acc = accuracy(test_circuits, test_targets)
    info_layers[n_layer] = test_acc

    print(f"Final test accuracy: {test_acc}")

print(info_layers)
