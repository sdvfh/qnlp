from typing import Any, Callable, Dict, List, Tuple, Union

import pennylane as qml
import scipy.linalg as la
from pennylane import numpy as np
from pennylane.operation import Operation

DatasetType = Dict[str, Union[List[str], np.ndarray, Any]]
DataDict = Dict[str, DatasetType]
SizeType = Union[int, Tuple["SizeType", ...]]


def random_unitary(n: int) -> np.ndarray:
    """
    Returns a Haar-distributed random unitary matrix from U(N).

    Args:
        n (int): The dimension of the unitary matrix.

    Returns:
        np.ndarray: A Haar-distributed random unitary matrix of shape (N, N).
    """
    # Generate a random complex matrix.
    z = np.random.randn(n, n) + 1.0j * np.random.randn(n, n)
    q, r = la.qr(z)
    d = np.diag(np.diagonal(r) / np.abs(np.diagonal(r)))
    return np.dot(q, d)


def haar_integral(num_qubits: int, samples: int) -> np.ndarray:
    """
    Computes the Haar integral for a specified number of samples by approximating
    the average density matrix over random unitary transformations.

    Args:
        num_qubits (int): The number of qubits (the Hilbert space dimension is 2**num_qubits).
        samples (int): The number of samples to average over.

    Returns:
        np.ndarray: The averaged density matrix computed from Haar-distributed unitaries.
    """
    n: int = 2**num_qubits
    zero_state: np.ndarray = np.zeros(n, dtype=complex)
    zero_state[0] = 1.0

    density_matrices = []
    for _ in range(samples):
        u = random_unitary(n)
        # Embed the zero state into the unitary transformation.
        a = np.matmul(zero_state, u).reshape(-1, 1)
        density_matrices.append(np.kron(a, a.conj().T))

    return np.mean(density_matrices, axis=0)


def pqc_integral(
    num_qubits: int,
    ansatze: Callable[[np.ndarray, int], Operation],
    size: SizeType,
    samples: int,
) -> np.ndarray:
    """
    Computes the integral of a Parameterized Quantum Circuit (PQC) over uniformly sampled parameters.

    This function samples parameters uniformly from the interval [-π, π], applies the given ansatze to prepare
    a quantum state, and computes the density matrix |ψ⟩⟨ψ| for that state. The result is the average density
    matrix over the specified number of samples.

    Args:
        num_qubits (int): The number of qubits in the circuit.
        ansatze (Callable[[np.ndarray, int], None]): A function that applies the ansatz circuit given parameters
            and the number of qubits.
        size (int): The number of parameters for the ansatz.
        samples (int): The number of samples over which to average.

    Returns:
        np.ndarray: The averaged density matrix of shape (2**num_qubits, 2**num_qubits).
    """
    # Initialize a PennyLane device for simulation.
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev, interface="autograd")
    def circuit(params: np.ndarray) -> np.ndarray:
        ansatze(params, num_qubits)
        return qml.state()

    # Collect density matrices in a list for each sample.
    density_matrices = []
    for _ in range(samples):
        parameters = np.random.uniform(-np.pi, np.pi, size)
        state = circuit(parameters).reshape(-1, 1)
        density = np.kron(state, state.conj().T)
        density_matrices.append(density)

    # Compute the mean of all density matrices along axis 0.
    return np.mean(density_matrices, axis=0)


def ansatz1(params: np.array, num_qubits: int) -> Operation:
    return qml.templates.StronglyEntanglingLayers(params, wires=range(num_qubits))


haar_value = haar_integral(10, 2048)
pqc_value = pqc_integral(10, ansatz1, (20, 10, 3), 2048)
print(np.linalg.norm(haar_value - pqc_value))
