from pathlib import Path

def get_path() -> dict:
    path = {
        "root": Path(__file__).parent.parent.parent,
    }
    path["code"] = path["root"] / "code"
    path["data"] = path["root"] / "data"
    return path

GLOBAL_SEED = 0