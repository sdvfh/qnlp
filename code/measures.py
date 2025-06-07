from models import AnsatzSingleRotX

qvc = AnsatzSingleRotX(
    n_layers=1, max_iter=40, batch_size=20, random_state=0, n_qubits_=4
)

print(qvc.compute_measures(n_bins=75, n=5000, to_plot=True))
