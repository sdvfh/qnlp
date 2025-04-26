from pathlib import Path

import pytreebank

root = Path(__file__).parent.parent
dataset_folder = root / "data" / "sst"
dataset_folder.mkdir(parents=True, exist_ok=True)
datasets = pytreebank.load_sst()

for dataset_type in ["train", "test"]:
    dataset = datasets[dataset_type]
    with open(dataset_folder / f"{dataset_type}.txt", "w") as f:
        for sample in dataset:
            label, sentence = sample.to_labeled_lines()[0]
            if label == 3:
                continue
            label = 0 if label < 3 else 1
            f.write(f"{label} {sentence}\n")
