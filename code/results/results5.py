from pathlib import Path

import pandas as pd


class Dataset:
    def __init__(self, paths):
        self.paths = paths
        self.dataset_name = None
        self.datasets = {}

    def read_dataset(self):
        for split in ["train", "test"]:
            file_path = self.paths["root"] / "data" / self.dataset_name / f"{split}.txt"
            data, targets = self.read_files(file_path)
            self.datasets[split] = {"data": data, "targets": targets}

    @staticmethod
    def read_files(filename):
        if not filename.exists():
            raise FileNotFoundError(f"O arquivo {filename} não existe.")
        with open(filename, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        data_targets = []
        for line in lines:
            if not line[0].isdigit():
                raise ValueError(f"Linha sem rótulo inteiro: {line}")
            label = int(line[0])
            text = line[1:].strip()
            data_targets.append((text, label))
        data, targets = zip(*data_targets, strict=True)
        return list(data), list(targets)


class ChatGPTDataset(Dataset):
    def __init__(self, paths, level):
        super().__init__(paths)
        self.dataset_name = f"chatgpt_{level}"


class SST(Dataset):
    def __init__(self, paths):
        super().__init__(paths)
        self.dataset_name = "sst"


def build_summary_dataframe(paths):
    datasets = [
        ("chatgpt_easy", ChatGPTDataset(paths, "easy"), "ChatGPT - Easy"),
        ("chatgpt_medium", ChatGPTDataset(paths, "medium"), "ChatGPT - Medium"),
        ("chatgpt_hard", ChatGPTDataset(paths, "hard"), "ChatGPT - Hard"),
        ("sst", SST(paths), "SST"),
    ]

    records = []
    for _, ds_obj, base_name in datasets:
        ds_obj.read_dataset()
        for split in ["train", "test"]:
            data = ds_obj.datasets[split]["data"]
            targets = ds_obj.datasets[split]["targets"]
            for val, cls in [(1, "Positiva"), (0, "Negativa")]:
                texts = [t for t, y in zip(data, targets, strict=True) if y == val]
                n = len(texts)
                avg_w = (
                    round(sum(len(text.split()) for text in texts) / n, 2) if n else 0.0
                )
                records.append(
                    {
                        "Base": base_name,
                        "Tipo de base": "Treino" if split == "train" else "Teste",
                        "Classe": cls,
                        "N° de exemplos": n,
                        "Média do n° de palavras": avg_w,
                    }
                )

    df = pd.DataFrame(records)
    # ordena as linhas conforme a sua tabela
    ordem = []
    for base in ["ChatGPT - Easy", "ChatGPT - Medium", "ChatGPT - Hard", "SST"]:
        for split in ["Treino", "Teste"]:
            for cls in ["Positiva", "Negativa"]:
                ordem.append((base, split, cls))
    df = df.set_index(["Base", "Tipo de base", "Classe"]).loc[ordem].reset_index()
    return df


if __name__ == "__main__":
    paths = {"root": Path(__file__).parent.parent.resolve()}
    df_resumo = build_summary_dataframe(paths)
    print(df_resumo.to_latex())
