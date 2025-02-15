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
    ansatze: Callable[[np.ndarray, int], Union[Operation, None]],
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


def meyer_wallach(
    circuit: Callable[[np.ndarray, int], Union[Operation, None]],
    num_qubits: int,
    size: SizeType,
    sample: int = 1024,
) -> float:
    """
    Computes the Meyer–Wallach entanglement measure for a given parameterized quantum circuit
    using PennyLane. This optimized version passes all parameter samples at once to the ansatz
    and leverages NumPy's vectorized operations for density matrix construction and purity calculation.

    Procedure:
      1. Generate a batch of 'sample' parameter sets with shape (sample, *size) sampled uniformly from [-π, π].
      2. Evaluate the circuit on all parameter sets simultaneously to obtain a batch of state vectors.
      3. Construct the batch of density matrices ρ = |ψ⟩⟨ψ| using a vectorized outer product.
      4. For each qubit, compute the reduced density matrices (via partial trace) for all samples at once.
      5. Calculate the purity of each reduced density matrix and average over all qubits.
      6. The Meyer–Wallach measure is 2 times the average over samples of (1 - average purity).

    Args:
        circuit: A function implementing the ansatz, which accepts parameters and the number of qubits.
        num_qubits (int): The number of qubits in the circuit.
        size (int or Tuple): The number (or shape) of parameters required by the ansatz.
        sample (int): The number of samples (executions with random parameters) for averaging.

    Returns:
        float: The Meyer–Wallach entanglement measure.
    """
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev, interface="autograd")
    def state_circuit(params: np.ndarray) -> np.ndarray:
        # Apply the ansatz; the initial state is |0...0> by default.
        circuit(params, num_qubits)
        return qml.state()

    # Generate all parameter samples at once: shape (sample, *size)
    parameters = np.random.uniform(-np.pi, np.pi, (sample, *np.atleast_1d(size)))
    # Evaluate the circuit in batch mode to obtain state vectors.
    # Assumes that state_circuit supports batch processing (returns an array of shape (sample, dim)).
    states = state_circuit(parameters)
    if states.shape != (sample, 2**num_qubits):
        states = np.repeat(states[np.newaxis, :], sample, axis=0)

    # Compute the batch of density matrices via vectorized outer product:
    # density_matrices[i] = |ψ_i⟩⟨ψ_i| for each sample i.
    density_matrices = np.einsum("bi,bj->bij", states, states.conj())

    qb = list(range(num_qubits))
    # Initialize an array to accumulate purity values for each sample.
    purity_sum = np.zeros(sample, dtype=complex)

    # Loop over qubit indices (usually a small number) to compute reduced density matrices.
    for j in qb:
        indices_to_trace = qb.copy()
        indices_to_trace.remove(j)
        # Compute the reduced density matrices for qubit j in batch.
        # qml.math.partial_trace accepts a batch of density matrices with shape (sample, dim, dim).
        reduced_density = qml.math.partial_trace(
            density_matrices, indices=indices_to_trace
        )
        # 'reduced_density' now has shape (sample, 2, 2).
        # Compute the purity for each sample: Tr((ρ_reduced)²).
        prod = np.matmul(reduced_density, reduced_density)
        purity_j = np.trace(prod, axis1=1, axis2=2)
        purity_sum += purity_j

    # Average the purity over all qubits for each sample.
    avg_purity_per_sample = purity_sum / num_qubits
    # The entanglement measure for each sample is 1 minus the average purity.
    entanglement_values = 1 - avg_purity_per_sample.real
    # The Meyer–Wallach measure is 2 times the average entanglement value over all samples.
    measure = 2 * np.mean(entanglement_values)
    return measure


def ansatz1(params: np.array, num_qubits: int) -> Operation:
    return qml.templates.StronglyEntanglingLayers(params, wires=range(num_qubits))


def max_entangled_ansatz(params: np.ndarray, num_qubits: int) -> None:
    """
    A test ansatz for 2 qubits that prepares a maximally entangled Bell state.
    This ansatz ignores the parameters and deterministically prepares the state:
      |Φ⁺⟩ = (|00⟩ + |11⟩) / √2,
    which yields the maximum Meyer–Wallach measure for 2 qubits.
    """
    # Ensure this ansatz is used only for 2 qubits.
    if num_qubits != 2:
        raise ValueError("max_entangled_ansatz is defined only for 2 qubits.")
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])


# Example 1
ansatz = ansatz1
num_qubits = 10
size = (20, 10, 3)
samples = 2048

# Example 2: full MW measure for 2 qubits
# ansatz = max_entangled_ansatz
# num_qubits = 2
# # Dummy parameter shape; since the ansatz ignores parameters, we can use a minimal shape.
# size = (1,)
# samples = 2048

haar_value = haar_integral(num_qubits, samples)
pqc_value = pqc_integral(num_qubits, ansatz, size, samples)
haar = np.linalg.norm(haar_value - pqc_value)
print("Haar: ", haar)

mw = meyer_wallach(ansatz, num_qubits, size, samples)
print("Meyer-Wallach: ", mw)
