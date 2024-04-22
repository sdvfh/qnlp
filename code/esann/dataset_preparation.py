from urllib.request import urlopen
from .utils import get_path
import tarfile
import urllib.request
import shutil
import pandas as pd

class DatasetProcessor:
    def __init__(self):
        self._path = get_path()

    def download(self):
        raise NotImplementedError()

    def load(self):
        raise NotImplementedError()

    def process(self):
        raise NotImplementedError()

    def create_embeddings(self):
        raise NotImplementedError()

class MovieReviewProcessor(DatasetProcessor):
    _url = "https://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz"
    def __init__(self):
        super().__init__()
        self._df = []
        self._path["root_dataset_folder"] = self._path["data"] / "movie_review"
        if not self._path["root_dataset_folder"].exists():
            self._path["root_dataset_folder"].mkdir(parents=True, exist_ok=True)

        self._path["file_path_compressed"] = self._path["root_dataset_folder"] / "review_polarity.tar.gz"
        self._path["folder_path_uncompressed"] = self._path["root_dataset_folder"] / "review_polarity"
        self._path["folder_dataset"] = self._path["folder_path_uncompressed"] / "txt_sentoken"
        self._path["file_all_sentences"] = self._path["root_dataset_folder"] / "all_sentences.snappy.parquet"

    def download(self):
        if not self._path["file_path_compressed"].exists():
            # From https://stackoverflow.com/questions/7243750/download-file-from-web-in-python-3
            with urllib.request.urlopen(self._url) as response, open(self._path["file_path_compressed"], 'wb') as file:
                shutil.copyfileobj(response, file)

        if not self._path["folder_path_uncompressed"].exists():
            with tarfile.open(self._path["file_path_compressed"], "r:gz") as tar:
                tar.extractall(self._path["folder_path_uncompressed"])


    def load(self):
        if not self._path["file_all_sentences"].exists():
            for file in self._path["folder_dataset"].rglob("*.txt"):
                label = 1 if file.parts[-2] == "pos" else -1
                text = file.read_text().splitlines()
                for sentence in text:
                    self._df.append([sentence, label])
            self._df = pd.DataFrame(self._df, columns=["sentence", "label"])
            self._df.to_parquet(self._path["file_all_sentences"])


class DatasetPreparation:
    _options = {"MR": MovieReviewProcessor}



    def load(self, dataset_initials: str):
        df_processor = self._options[dataset_initials]()
        df_processor.download()
        df_processor.load()
        df_processor.process()
        df_processor.create_embeddings()