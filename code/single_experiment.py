# Import libraries
import numpy.random as npr
from lambeq import AtomicType, BobcatParser, StronglyEntanglingAnsatz
from lambeq.backend.numerical_backend import set_backend
from sympy import default_sort_key

# Set backend for lambeq
set_backend("numpy")

# Set numpy seed
npr.seed(0)

# Example sentence
sentence = "Alice runs."
print("Sentence: ", sentence)

# Create BobcatParser object
parser = BobcatParser()
diagram = parser.sentence2diagram(sentence)
diagram.draw()

# Define ansatz
ansatz = StronglyEntanglingAnsatz(
    {AtomicType.NOUN: 1, AtomicType.SENTENCE: 1}, n_layers=1, n_single_qubit_params=3
)

# Create circuit
circuit = ansatz(diagram)
circuit.draw(figsize=(14, 7))

# Create vocabulary from circuit
vocab = sorted(circuit.free_symbols, key=default_sort_key)
print()
print("Vocabulary: ", vocab)

# Add angles to vocabulary
values_angles = [npr.uniform(low=0.0, high=1) for symbol in vocab]
dict_angles = dict(zip(vocab, values_angles, strict=True))
print()
print("Vocabulary with angles: ", dict_angles)

# Lambeq to PennyLane circuit
qml_circ = circuit.to_pennylane(
    probabilities=True,
    backend_config=None,
    diff_method="best",
)

# Initialise device and circuit
qml_circ.initialise_device_and_circuit()
qml_circ.initialise_concrete_params(dict_angles)

# Draw circuit
print()
print("Circuit:")
qml_circ.draw()

# Eval circuit
print()
print("Lambeq eval: ", qml_circ.eval())

# How to recreate the circuit
# list(zip(qml_circ._ops, qml_circ._concrete_params, qml_circ._wires))
