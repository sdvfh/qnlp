from typing import List, Tuple

import numpy as np
import numpy.random as npr
from jax import grad, jit
from jax import numpy as jnp
from lambeq import AtomicType, BobcatParser, IQPAnsatz, RemoveCupsRewriter
from lambeq.backend.numerical_backend import set_backend
from sympy import default_sort_key

set_backend("jax")


def read_data(filename: str) -> Tuple[List[str], jnp.ndarray]:
    """Read data from a file and return sentences and targets."""
    data, targets = [], []
    with open(filename, "r") as file:
        for line in file:
            label = int(line[0])
            sentence = line[1:].strip()
            target = jnp.array([label, 1 - label], dtype=jnp.float32)
            data.append(sentence)
            targets.append(target)
    return data, jnp.array(targets)


def sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    """Apply the sigmoid function element-wise."""
    return 1 / (1 + jnp.exp(-x))


def compute_loss(tensors: List[jnp.ndarray], circuits, targets) -> float:
    """Compute the cross-entropy loss over the training data."""
    np_circuits = [circuit.lambdify(*vocab)(*tensors) for circuit in circuits]
    predictions = sigmoid(jnp.array([c.eval() for c in np_circuits]))
    loss = -jnp.mean(jnp.sum(targets * jnp.log(predictions + 1e-10), axis=1))
    return jnp.float32(loss)


def update_tensors(
    tensors: List[jnp.ndarray], gradients: List[jnp.ndarray], learning_rate: float
) -> List[jnp.ndarray]:
    """Update tensors using the computed gradients."""
    return [
        tensor - learning_rate * grad
        for tensor, grad in zip(tensors, gradients, strict=True)
    ]


# Set random seed for reproducibility
npr.seed(0)

# Read training and testing data
train_data, train_targets = read_data("../data/lambeq/mc_train_data.txt")
test_data, test_targets = read_data("../data/lambeq/mc_test_data.txt")

# Parse sentences into diagrams
parser = BobcatParser()
raw_train_diagrams = parser.sentences2diagrams(train_data)
raw_test_diagrams = parser.sentences2diagrams(test_data)

remove_cups = RemoveCupsRewriter()

train_diagrams = [remove_cups(diagram) for diagram in raw_train_diagrams]
test_diagrams = [remove_cups(diagram) for diagram in raw_test_diagrams]

# Create ansatz and circuits
ansatz = IQPAnsatz(
    {AtomicType.NOUN: 1, AtomicType.SENTENCE: 1}, n_layers=1, n_single_qubit_params=3
)
train_circuits = [ansatz(diagram) for diagram in train_diagrams]
test_circuits = [ansatz(diagram) for diagram in test_diagrams]

# Build vocabulary
all_circuits = train_circuits + test_circuits
vocab = sorted(
    {sym for circ in all_circuits for sym in circ.free_symbols}, key=default_sort_key
)

# Initialize tensors
tensors = [jnp.array(npr.uniform(low=0.0, high=2 * jnp.pi)) for symbol in vocab]

# JIT-compile loss and gradient functions
compiled_loss = jit(lambda t: compute_loss(t, train_circuits, train_targets))
compiled_grad = jit(grad(compiled_loss))

# Training loop
epochs = 90
learning_rate = 1.0
training_losses = []

for epoch in range(1, epochs + 1):
    gradients = compiled_grad(tensors)
    tensors = update_tensors(tensors, gradients, learning_rate)
    loss_value = float(compiled_loss(tensors))
    training_losses.append(loss_value)

    if epoch % 10 == 0 or epoch == epochs:
        print(f"Epoch {epoch} - Loss: {loss_value:.4f}")

# Evaluation on the test set
np_test_circuits = [circuit.lambdify(*vocab)(*tensors) for circuit in test_circuits]
test_predictions = sigmoid(jnp.array([c.eval() for c in np_test_circuits]))
predicted_labels = jnp.argmax(jnp.float32(test_predictions), axis=1)
true_labels = jnp.argmax(test_targets, axis=1)
accuracy = jnp.mean(predicted_labels == true_labels)

print(f"Accuracy on test set: {accuracy:.2%}")
