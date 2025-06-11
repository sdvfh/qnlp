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

models = {
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
n_layers = {1: {}, 10: {}}
measures = {}
for with_state_prep in [False, True]:
    for n_layer in n_layers:
        for model in models:
            qvc = models[model](
                n_layers=n_layer,
                max_iter=40,
                batch_size=20,
                random_state=0,
                n_qubits_=4,
            )
            exp, ent = qvc.compute_measures(
                n_bins=75, n=5000, to_plot=False, with_state_prep=with_state_prep
            )
            measures[model] = {"exp": exp, "ent": ent}
            print(f"With {n_layer} layers, model {model}: {measures[model]}")
        n_layers[n_layer] = measures
        if with_state_prep:
            filename = f"measures_{n_layer}_layer_with_state_prep.csv"
        else:
            filename = f"measures_{n_layer}_layer.csv"
        pd.DataFrame(measures).to_csv(filename, index=False)
