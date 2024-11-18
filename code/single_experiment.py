# Import libraries
import numpy.random as npr
from lambeq import AtomicType, BobcatParser, StronglyEntanglingAnsatz
from lambeq.backend.numerical_backend import set_backend
from sympy import default_sort_key

# Set backend for lambeq
set_backend("jax")

# Set numpy seed
npr.seed(0)

# Example sentence
sentence = "Alice runs."
print(sentence)

# Create BobcatParser object
parser = BobcatParser()
diagram = parser.sentence2diagram(sentence)

# Define ansatz
ansatz = StronglyEntanglingAnsatz(
    {AtomicType.NOUN: 1, AtomicType.SENTENCE: 1}, n_layers=1, n_single_qubit_params=1
)

# Create circuit
circuit = ansatz(diagram)
circuit.draw(figsize=(14, 7))

# Create vocabulary from circuit
vocab = sorted(circuit.free_symbols, key=default_sort_key)
print("Vocabulary: ", vocab)

# Add angles to vocabulary
values_angles = [(npr.uniform(low=0.0, high=1),) for symbol in vocab]
dict_angles = dict(zip(vocab, values_angles, strict=True))
print("Vocabulary with angles: ", dict_angles)
