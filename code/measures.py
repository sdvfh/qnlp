from pathlib import Path

import pandas as pd
from models import (
    AnsatzEnt1,
    AnsatzEnt2,
    AnsatzEnt3,
    AnsatzEnt4,
    AnsatzMaouaki1,
    AnsatzMaouaki6,
    AnsatzMaouaki7,
    AnsatzMaouaki9,
    AnsatzMaouaki15,
    AnsatzMaouakiQuasi7,
    AnsatzMottoten,
    AnsatzRot,
    AnsatzRotCNOT,
    AnsatzSingleRotX,
    AnsatzSingleRotY,
    AnsatzSingleRotZ,
)

# ────────────────────────────── parâmetros ────────────────────────────────────
MODELS = {
    "stateprep": AnsatzMottoten,
    "singlerotx": AnsatzSingleRotX,
    "singleroty": AnsatzSingleRotY,
    "singlerotz": AnsatzSingleRotZ,
    "rot": AnsatzRot,
    "rotcnot": AnsatzRotCNOT,
    "maouaki1": AnsatzMaouaki1,
    "maouaki6": AnsatzMaouaki6,
    "maouakiquasi7": AnsatzMaouakiQuasi7,
    "maouaki7": AnsatzMaouaki7,
    "maouaki9": AnsatzMaouaki9,
    "maouaki15": AnsatzMaouaki15,
    "ent1": AnsatzEnt1,
    "ent2": AnsatzEnt2,
    "ent3": AnsatzEnt3,
    "ent4": AnsatzEnt4,
}

N_LAYERS = [1, 10]
WITH_STATE_PREP = [False, True]

RESULTS_DIR = Path(__file__).with_suffix("").parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

# ────────────────────────────── execução ──────────────────────────────────────
rows = []

for state_prep in WITH_STATE_PREP:
    for n_layer in N_LAYERS:
        for name, model_cls in MODELS.items():
            # ▸ condição especial para o AnsatzMottoten
            if name == "stateprep" and not (n_layer == 1 and state_prep is False):
                continue  # pula todas as outras combinações
            qvc = model_cls(
                n_layers=n_layer,
                max_iter=40,
                batch_size=20,
                random_state=0,
                n_qubits_=4,
            )
            exp, ent = qvc.compute_measures(
                n_bins=75, n=5000, to_plot=False, with_state_prep=state_prep
            )
            rows.append(
                {
                    "model": name,
                    "n_layers": n_layer,
                    "with_state_prep": state_prep,
                    "exp": exp,
                    "ent": ent,
                }
            )
            print(
                f"[layers={n_layer} | stateprep={state_prep}] {name}: "
                f"exp={exp:.4f}, ent={ent:.4f}"
            )

# ────────────────────────────── salvamento ───────────────────────────────────
df = pd.DataFrame(rows)
out_file = RESULTS_DIR / "measures_all.csv"
df.to_csv(out_file, index=False)
print(f"\n✔️  Resultados consolidados salvos em: {out_file.resolve()}")
