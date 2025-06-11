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


# Ansatz definitions 1 to 19
def ansatz_1(params):
    # RX(0-3), RZ(4-7)
    for i in range(4):
        qml.RX(params[:, i], wires=i)
    for i in range(4):
        qml.RZ(params[:, 4 + i], wires=i)


def ansatz_2(params):
    ansatz_1(params)
    qml.CNOT(wires=[3, 2])
    qml.CNOT(wires=[2, 1])
    qml.CNOT(wires=[1, 0])


def ansatz_3(params):
    ansatz_1(params)
    qml.CRZ(params[:, 8], wires=[3, 2])
    qml.CRZ(params[:, 9], wires=[2, 1])
    qml.CRZ(params[:, 10], wires=[1, 0])


def ansatz_4(params):
    ansatz_1(params)
    qml.CRX(params[:, 8], wires=[3, 2])
    qml.CRX(params[:, 9], wires=[2, 1])
    qml.CRX(params[:, 10], wires=[1, 0])


def ansatz_5(params):
    ansatz_1(params)
    pairs = [
        (3, 0),
        (3, 1),
        (3, 2),
        (2, 0),
        (2, 1),
        (2, 3),
        (1, 0),
        (1, 2),
        (1, 3),
        (0, 1),
        (0, 2),
        (0, 3),
    ]
    for idx, (c, t) in enumerate(pairs, start=8):
        qml.CRZ(params[:, idx], wires=[c, t])
    # second RX/RZ
    for i in range(4):
        qml.RX(params[:, 20 + i], wires=i)
    for i in range(4):
        qml.RZ(params[:, 24 + i], wires=i)


def ansatz_6(params):
    ansatz_1(params)
    pairs = [
        (3, 0),
        (3, 1),
        (3, 2),
        (2, 0),
        (2, 1),
        (2, 3),
        (1, 0),
        (1, 2),
        (1, 3),
        (0, 1),
        (0, 2),
        (0, 3),
    ]
    for idx, (c, t) in enumerate(pairs, start=8):
        qml.CRX(params[:, idx], wires=[c, t])
    for i in range(4):
        qml.RX(params[:, 20 + i], wires=i)
    for i in range(4):
        qml.RZ(params[:, 24 + i], wires=i)


def ansatz_7(params):
    ansatz_1(params)
    qml.CRZ(params[:, 8], wires=[1, 0])
    qml.CRZ(params[:, 9], wires=[3, 2])
    for i in range(4):
        qml.RX(params[:, 10 + i], wires=i)
    for i in range(4):
        qml.RZ(params[:, 14 + i], wires=i)
    qml.CRZ(params[:, 18], wires=[2, 1])


def ansatz_8(params):
    ansatz_1(params)
    qml.CRX(params[:, 8], wires=[1, 0])
    qml.CRX(params[:, 9], wires=[3, 2])
    for i in range(4):
        qml.RX(params[:, 10 + i], wires=i)
    for i in range(4):
        qml.RZ(params[:, 14 + i], wires=i)
    qml.CRX(params[:, 18], wires=[2, 1])


def ansatz_9(params):
    for i in range(4):
        qml.H(wires=i)
    qml.CZ(wires=[3, 2])
    qml.CZ(wires=[2, 1])
    qml.CZ(wires=[1, 0])
    for i in range(4):
        qml.RX(params[:, i], wires=i)


def ansatz_10(params):
    for i in range(4):
        qml.RY(params[:, i], wires=i)
    qml.CZ(wires=[3, 2])
    qml.CZ(wires=[2, 1])
    qml.CZ(wires=[1, 0])
    qml.CZ(wires=[3, 0])
    for i in range(4):
        qml.RY(params[:, 4 + i], wires=i)


def ansatz_11(params):
    for i in range(4):
        qml.RY(params[:, i], wires=i)
    for i in range(4):
        qml.RZ(params[:, 4 + i], wires=i)
    qml.CNOT(wires=[1, 0])
    qml.CNOT(wires=[3, 2])
    qml.RY(params[:, 8], wires=1)
    qml.RY(params[:, 9], wires=2)
    qml.RZ(params[:, 10], wires=1)
    qml.RZ(params[:, 11], wires=2)
    qml.CNOT(wires=[2, 1])


def ansatz_12(params):
    for i in range(4):
        qml.RY(params[:, i], wires=i)
    for i in range(4):
        qml.RZ(params[:, 4 + i], wires=i)
    qml.CZ(wires=[1, 0])
    qml.CZ(wires=[3, 2])
    qml.RY(params[:, 8], wires=1)
    qml.RY(params[:, 9], wires=2)
    qml.RZ(params[:, 10], wires=1)
    qml.RZ(params[:, 11], wires=2)
    qml.CZ(wires=[2, 1])


def ansatz_13(params):
    for i in range(4):
        qml.RY(params[:, i], wires=i)
    for idx, (c, t) in enumerate([(3, 0), (2, 3), (1, 2), (0, 1)], start=4):
        qml.CRZ(params[:, idx], wires=[c, t])
    for i in range(4):
        qml.RY(params[:, 8 + i], wires=i)
    for idx, (c, t) in enumerate([(3, 2), (0, 3), (1, 0), (2, 1)], start=12):
        qml.CRZ(params[:, idx], wires=[c, t])


def ansatz_14(params):
    for i in range(4):
        qml.RY(params[:, i], wires=i)
    for idx, (c, t) in enumerate([(3, 0), (2, 3), (1, 2), (0, 1)], start=4):
        qml.CRX(params[:, idx], wires=[c, t])
    for i in range(4):
        qml.RY(params[:, 8 + i], wires=i)
    for idx, (c, t) in enumerate([(3, 2), (0, 3), (1, 0), (2, 1)], start=12):
        qml.CRX(params[:, idx], wires=[c, t])


def ansatz_15(params):
    for i in range(4):
        qml.RY(params[:, i], wires=i)
    for c, t in [(3, 0), (2, 3), (1, 2), (0, 1)]:
        qml.CNOT(wires=[c, t])
    for i in range(4):
        qml.RY(params[:, 4 + i], wires=i)
    for c, t in [(3, 2), (0, 3), (1, 0), (2, 1)]:
        qml.CNOT(wires=[c, t])


def ansatz_16(params):
    ansatz_1(params)
    qml.CRZ(params[:, 8], wires=[1, 0])
    qml.CRZ(params[:, 9], wires=[3, 2])
    qml.CRZ(params[:, 10], wires=[2, 1])


def ansatz_17(params):
    ansatz_1(params)
    qml.CRX(params[:, 8], wires=[1, 0])
    qml.CRX(params[:, 9], wires=[3, 2])
    qml.CRX(params[:, 10], wires=[2, 1])


def ansatz_18(params):
    ansatz_1(params)
    for idx, (c, t) in enumerate([(3, 0), (2, 3), (1, 2), (0, 1)], start=8):
        qml.CRZ(params[:, idx], wires=[c, t])


def ansatz_19(params):
    ansatz_1(params)
    for idx, (c, t) in enumerate([(3, 0), (2, 3), (1, 2), (0, 1)], start=8):
        qml.CRX(params[:, idx], wires=[c, t])


def probability_haar(lower_value, upper_value, N):
    return primitive_p_haar(upper_value, N) - primitive_p_haar(lower_value, N)


def primitive_p_haar(fidelity, N):
    return -((1 - fidelity) ** (N - 1))


def compute_entanglement(states, n_qubits):
    M = states.shape[0]
    psi_t = states.reshape((M,) + (2,) * n_qubits)

    distances_sum = np.zeros(M)

    for j in range(n_qubits):
        u = np.take(psi_t, 0, axis=1 + j).reshape(M, -1)
        v = np.take(psi_t, 1, axis=1 + j).reshape(M, -1)

        M1 = u[:, :, None] * v[:, None, :]
        M2 = v[:, :, None] * u[:, None, :]

        D_j = 0.5 * np.abs(M1 - M2) ** 2
        D_j = D_j.sum(axis=(1, 2))

        distances_sum += D_j

    ent_values = (4 / n_qubits) * distances_sum
    return float(ent_values.mean())


def compute_expressability_entanglement(
    ansatz, n_params, n_qubits, n_bins, to_plot, N, seed
):
    rng = np.random.default_rng(seed)

    n_bins_list = [i / n_bins for i in range(n_bins + 1)]
    prob_dist_haar = [
        probability_haar(n_bins_list[i], n_bins_list[i + 1], 2**n_qubits)
        for i in range(n_bins)
    ]

    params_shape = (N, n_params)

    params, params_conj = rng.uniform(0, 2 * np.pi, size=params_shape), rng.uniform(
        0, 2 * np.pi, size=params_shape
    )
    # params, params_conj = random_su2(rng, N), random_su2(rng, N)

    circ_expr, circ_ent = create_circuit(n_qubits, ansatz)
    fidelities_ansatz = circ_expr(params, params_conj)

    if np.ndim(fidelities_ansatz) == 0:
        fidelities_ansatz = np.full(N, fidelities_ansatz)

    prob_dist_ansatz, edges = np.histogram(
        fidelities_ansatz,
        bins=n_bins,
        range=(0, 1),
        density=False,
        weights=np.ones(N) / N,
    )

    div_kl = np.sum(rel_entr(prob_dist_ansatz, prob_dist_haar))

    if to_plot:
        widths = np.diff(edges)
        fig, ax = plt.subplots(dpi=300)
        ax.bar(
            edges[:-1],
            prob_dist_haar,
            width=widths,
            align="edge",
            label="Haar",
            edgecolor="white",
            linewidth=0.1,
        )

        ax.bar(
            edges[:-1],
            prob_dist_ansatz,
            width=widths,
            align="edge",
            label="A",
            edgecolor="white",
            alpha=0.5,
            linewidth=0.1,
        )

        ax.set_xlabel("Fidelity")
        ax.set_ylabel("Probability Density")
        ax.set_xlim(0, 1)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        plt.show()

    if n_qubits > 1:
        states = circ_ent(params)
        ent = compute_entanglement(states, n_qubits)
    else:
        ent = 0
    print(
        f"Circuit {ansatz.__name__}: Divergence KL = {div_kl:.4f}, Entanglement = {ent:.4f}"
    )


seed = 0
N = 5_000
n_qubits = 4
n_bins = 75
to_plot = False

ansatz_list = [
    (ansatz_1, 8),
    (ansatz_2, 8),
    (ansatz_3, 11),
    (ansatz_4, 11),
    (ansatz_5, 28),
    (ansatz_6, 28),
    (ansatz_7, 19),
    (ansatz_8, 19),
    (ansatz_9, 4),
    (ansatz_10, 8),
    (ansatz_11, 12),
    (ansatz_12, 12),
    (ansatz_13, 16),
    (ansatz_14, 16),
    (ansatz_15, 8),
    (ansatz_16, 11),
    (ansatz_17, 11),
    (ansatz_18, 12),
    (ansatz_19, 12),
]

for ansatz, num_params in ansatz_list:
    compute_expressability_entanglement(
        ansatz, num_params, n_qubits, n_bins, to_plot, N, seed
    )
