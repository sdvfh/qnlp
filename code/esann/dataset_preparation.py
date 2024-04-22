from urllib.request import urlopen
from .utils import get_path
import tarfile
import urllib.request
import shutil

class DatasetProcessor:
    def __init__(self):
        self._path = get_path()

    def download(self):
        raise NotImplementedError()

    def load(self):
        raise NotImplementedError()

    def create_embeddings(self):
        raise NotImplementedError()

class MovieReviewProcessor(DatasetProcessor):
    _url = "https://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz"
    def __init__(self):
        super().__init__()
        self.df = None
        self._path["root_dataset_folder"] = self._path["data"] / "movie_review"
        if not self._path["root_dataset_folder"].exists():
            self._path["root_dataset_folder"].mkdir(parents=True, exist_ok=True)

        self._path["file_path_compressed"] = self._path["root_dataset_folder"] / "review_polarity.tar.gz"
        self._path["folder_path_uncompressed"] = self._path["root_dataset_folder"] / "review_polarity"

    def download(self):
        if not self._path["file_path_compressed"].exists():
            # From https://stackoverflow.com/questions/7243750/download-file-from-web-in-python-3
            with urllib.request.urlopen(self._url) as response, open(self._path["file_path_compressed"], 'wb') as file:
                shutil.copyfileobj(response, file)

        if not self._path["folder_path_uncompressed"].exists():
            with tarfile.open(self._path["file_path_compressed"], "r:gz") as tar:
                tar.extractall(self._path["folder_path_uncompressed"])


    def load(self):
        pass

class DatasetPreparation:
    _options = {"MR": MovieReviewProcessor}



    def load(self, dataset_initials: str):
        df_processor = self._options[dataset_initials]()
        df_processor.download()
        df_processor.load()
        df_processor.create_embeddings()