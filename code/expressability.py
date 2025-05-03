import pennylane as qml
from matplotlib import pyplot as plt
from pennylane import numpy as np
from scipy.special import rel_entr


def create_circuit(n_qubits, ansatz):
    wires = list(range(n_qubits))
    dev = qml.device("default.qubit", wires=wires)
    zero_ket = [0] * (2**n_qubits)
    zero_ket[0] = 1

    projector = qml.Projector(zero_ket, wires)

    @qml.qnode(dev)
    def circuit(params, params_conj):
        ansatz(params)
        qml.adjoint(ansatz)(params_conj)
        return qml.expval(projector)

    return circuit


def ansatz_two_qubits(params):
    qml.H(wires=0)
    qml.H(wires=1)
    qml.RZ(params[:, 0], wires=0)
    qml.RX(params[:, 1], wires=1)


def ansatz_three_qubits(params):
    qml.H(wires=0)
    qml.H(wires=1)
    qml.H(wires=2)
    qml.RZ(params[:, 0], wires=0)
    qml.RX(params[:, 1], wires=1)
    qml.RY(params[:, 2], wires=1)


def ansatz_four_qubits(params):
    qml.H(wires=0)
    qml.H(wires=1)
    qml.H(wires=2)
    qml.H(wires=3)
    qml.RZ(params[:, 0], wires=0)
    qml.RX(params[:, 1], wires=1)
    qml.RY(params[:, 2], wires=1)
    qml.RY(params[:, 3], wires=1)


def ansatz_a(params):
    qml.H(wires=0)
    qml.RZ(params[:, 0], wires=0)


def ansatz_b(params):
    qml.H(wires=0)
    qml.RZ(params[:, 0], wires=0)
    qml.RX(params[:, 1], wires=0)


def ansatz_idle(params):
    return


def ansatz_u(params):
    qml.Rot(params[:, 0], params[:, 1], params[:, 2], wires=0)


def probability_haar(lower_value, upper_value, N):
    return primitive_p_haar(upper_value, N) - primitive_p_haar(lower_value, N)


def primitive_p_haar(fidelity, N):
    return -((1 - fidelity) ** (N - 1))


def random_su2(rng, size):
    phi = rng.uniform(0, 2 * np.pi, size=size)
    omega = rng.uniform(0, 2 * np.pi, size=size)
    z = rng.uniform(0, 1, size=size)
    theta = np.arccos(1 - 2 * z)
    return np.column_stack((phi, theta, omega))


seed = 0
# N = 5000
N = 1_000_000
n_qubits = 1
n_bins = 75
# n_bins = int(np.ceil(np.sqrt(N)))

rng = np.random.default_rng(seed)

n_bins_list = [i / n_bins for i in range(n_bins + 1)]
prob_dist_haar = [
    probability_haar(n_bins_list[i], n_bins_list[i + 1], 2**n_qubits)
    for i in range(n_bins)
]

# my_ansatz = ansatz_a
my_ansatz = ansatz_u
params_shape = (N, 3)

# params, params_conj = rng.uniform(0, 2 * np.pi, size=params_shape), rng.uniform(0, 2 * np.pi, size=params_shape)
params, params_conj = random_su2(rng, N), random_su2(rng, N)

my_circuit = create_circuit(n_qubits, my_ansatz)
fidelities_ansatz = my_circuit(params, params_conj)

if np.ndim(fidelities_ansatz) == 0:
    fidelities_ansatz = np.full(N, fidelities_ansatz)

# hist: densidade p(F) em cada bin
# edges: tamanho n_bins+1 com as fronteiras
prob_dist_ansatz, edges = np.histogram(
    fidelities_ansatz,
    bins=n_bins,
    range=(0, 1),
    density=False,
    weights=np.ones(N) / N,
)

divergence_kl = np.sum(rel_entr(prob_dist_ansatz, prob_dist_haar))

print(f"Divergence KL is {divergence_kl:0.4f}")

widths = np.diff(edges)
# 2. Plote como barras: cada barra começa em edges[i] e tem largura edges[i+1]−edges[i]
fig, ax = plt.subplots(dpi=300)
ax.bar(
    edges[:-1],  # borda esquerda de cada barra
    prob_dist_haar,  # probabilidades de Haar em cada bin
    width=widths,  # largura dos bins
    align="edge",  # alinha cada barra à borda esquerda :contentReference[oaicite:1]{index=1}
    label="Haar",
    edgecolor="white",
    linewidth=0.1,
)

ax.bar(
    edges[:-1],  # posições à esquerda de cada barra
    prob_dist_ansatz,  # alturas (densidade)
    width=widths,  # largura de cada bin
    align="edge",  # alinha cada barra à sua borda esquerda
    label="A",
    edgecolor="white",
    alpha=0.5,  # semitransparência para enxergar o histograma atrás
    linewidth=0.1,
)

# 3. Rótulos e ajustes estéticos
ax.set_xlabel("Fidelity")  # eixo x = fidelidade
ax.set_ylabel("Probability Density")  # eixo y = densidade
ax.set_xlim(0, 1)  # garante que fique em [0,1]
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend()
plt.show()
