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
    def expr(params, params_conj):
        ansatz(params)
        qml.adjoint(ansatz)(params_conj)
        return qml.expval(projector)

    @qml.qnode(dev)
    def ent(params):
        ansatz(params)
        return qml.state()

    return expr, ent


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


def random_su2(rng, size):
    phi = rng.uniform(0, 2 * np.pi, size=size)
    omega = rng.uniform(0, 2 * np.pi, size=size)
    z = rng.uniform(0, 1, size=size)
    theta = np.arccos(1 - 2 * z)
    return np.column_stack((phi, theta, omega))


def ansatz_16(params):
    qml.RX(params[:, 0], wires=0)
    qml.RX(params[:, 1], wires=1)
    qml.RX(params[:, 2], wires=2)
    qml.RX(params[:, 3], wires=3)

    qml.RZ(params[:, 4], wires=0)
    qml.RZ(params[:, 5], wires=1)
    qml.RZ(params[:, 6], wires=2)
    qml.RZ(params[:, 7], wires=3)

    qml.CRZ(params[:, 8], wires=[1, 0])
    qml.CRZ(params[:, 9], wires=[3, 2])
    qml.CRZ(params[:, 10], wires=[2, 1])


def ansatz_17(params):
    qml.RX(params[:, 0], wires=0)
    qml.RX(params[:, 1], wires=1)
    qml.RX(params[:, 2], wires=2)
    qml.RX(params[:, 3], wires=3)

    qml.RZ(params[:, 4], wires=0)
    qml.RZ(params[:, 5], wires=1)
    qml.RZ(params[:, 6], wires=2)
    qml.RZ(params[:, 7], wires=3)

    qml.CRX(params[:, 8], wires=[1, 0])
    qml.CRX(params[:, 9], wires=[3, 2])
    qml.CRX(params[:, 10], wires=[2, 1])


def ansatz_14(params):
    qml.RY(params[:, 0], wires=0)
    qml.RY(params[:, 1], wires=1)
    qml.RY(params[:, 2], wires=2)
    qml.RY(params[:, 3], wires=3)

    qml.CRX(params[:, 4], wires=[3, 0])
    qml.CRX(params[:, 5], wires=[2, 3])
    qml.CRX(params[:, 6], wires=[1, 2])
    qml.CRX(params[:, 7], wires=[0, 1])

    qml.RY(params[:, 8], wires=0)
    qml.RY(params[:, 9], wires=1)
    qml.RY(params[:, 10], wires=2)
    qml.RY(params[:, 11], wires=3)

    qml.CRX(params[:, 12], wires=[3, 2])
    qml.CRX(params[:, 13], wires=[0, 3])
    qml.CRX(params[:, 14], wires=[1, 0])
    qml.CRX(params[:, 15], wires=[2, 1])


def probability_haar(lower_value, upper_value, N):
    return primitive_p_haar(upper_value, N) - primitive_p_haar(lower_value, N)


def primitive_p_haar(fidelity, N):
    return -((1 - fidelity) ** (N - 1))


def iota_j(psi: np.ndarray, j: int, b: int) -> np.ndarray:
    """
    Eq. (18): projeta o qubit j em |b> e remove esse eixo.
    - psi: vetcor de estado de tamanho 2**n
    - j: índice do qubit (0-based)
    - b: valor 0 ou 1 para projeção
    Retorna: vector de tamanho 2**(n-1).
    """
    # 1) Descobre número de qubits n
    n = int(np.log2(psi.size))
    # 2) Torna psi em tensor shape=(2,)*n
    psi_tensor = psi.reshape((2,) * n)  # :contentReference[oaicite:0]{index=0}
    # 3) Seleciona fatia psi_tensor[b,...] no eixo j
    phi = np.take(
        psi_tensor, indices=b, axis=j
    )  # :contentReference[oaicite:1]{index=1}
    # 4) Flatten para vector
    return phi.reshape(-1)


def compute_distance_ent(u, v):
    norm_u2 = np.linalg.norm(u) ** 2
    norm_v2 = np.linalg.norm(v) ** 2
    overlap = np.vdot(u, v)
    return norm_u2 * norm_v2 - np.abs(overlap) ** 2


def compute_entanglement_vectorized(states: np.ndarray, n_qubits: int) -> float:
    M = states.shape[0]
    # reshape to (M, 2, 2, ..., 2) with one axis per qubit
    psi_t = states.reshape((M,) + (2,) * n_qubits)

    # accumulate distances for each state
    distances_sum = np.zeros(M)

    for j in range(n_qubits):
        # select substate when qubit j = 0 or 1
        u = np.take(psi_t, indices=0, axis=1 + j).reshape(M, -1)
        v = np.take(psi_t, indices=1, axis=1 + j).reshape(M, -1)

        # vectorized norms and overlaps for all M states at once
        norm_u2 = np.sum(np.abs(u) ** 2, axis=1)  # shape (M,)
        norm_v2 = np.sum(np.abs(v) ** 2, axis=1)  # shape (M,)
        overlap = np.sum(np.conj(u) * v, axis=1)  # shape (M,)

        # distance D_j for each state
        distances_sum += norm_u2 * norm_v2 - np.abs(overlap) ** 2

    # Meyer–Wallach Q for each state, then mean
    ent_values = (4 / n_qubits) * distances_sum  # shape (M,)
    return np.mean(ent_values)


seed = 0
N = 5_000
# N = 1_000_000
n_qubits = 4
n_bins = 75
# n_bins = int(np.ceil(np.sqrt(N)))
to_plot = True

rng = np.random.default_rng(seed)

n_bins_list = [i / n_bins for i in range(n_bins + 1)]
prob_dist_haar = [
    probability_haar(n_bins_list[i], n_bins_list[i + 1], 2**n_qubits)
    for i in range(n_bins)
]

# my_ansatz = ansatz_a
my_ansatz = ansatz_14
params_shape = (N, 16)

params, params_conj = rng.uniform(0, 2 * np.pi, size=params_shape), rng.uniform(
    0, 2 * np.pi, size=params_shape
)
# params, params_conj = random_su2(rng, N), random_su2(rng, N)

circ_expr, circ_ent = create_circuit(n_qubits, my_ansatz)
fidelities_ansatz = circ_expr(params, params_conj)

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

if to_plot:
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

if n_qubits == 1:
    ent_final = 0
else:
    states = circ_ent(params)
    ent_final = compute_entanglement_vectorized(states, n_qubits)

print(f"Entanglement value is {ent_final:0.4f}")
# ents = []
# for i in range(len(states)):
#     sample = states[i]
#     distances = []
#     for j_qubit in range(n_qubits):
#         distance = compute_distance_ent(iota_j(sample, j_qubit, 0), iota_j(sample, j_qubit, 1))
#         distances.append(distance)
#     ent = (4 / n_qubits) * np.sum(distances)
#     ents.append(ent)
# ent_final = np.mean(ents)
