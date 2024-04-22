from urllib.request import urlopen
from .utils import get_path
import tarfile
import urllib.request
import shutil

class DatasetProcessor:
    def __init__(self):
        self._path = get_path()

class MovieReviewProcessor(DatasetProcessor):
    def load(self):
        url = "https://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz"
        root_path = self._path["data"] / "movie_review"
        if not root_path.exists():
            root_path.mkdir(parents=True, exist_ok=True)
        file_path_compressed = root_path / "review_polarity.tar.gz"
        folder_path_uncompressed = root_path / "txt_sentoken"
        if not file_path_compressed.exists():
            # From https://stackoverflow.com/questions/7243750/download-file-from-web-in-python-3
            with urllib.request.urlopen(url) as response, open(file_path_compressed, 'wb') as file:
                shutil.copyfileobj(response, file)

        if not folder_path_uncompressed.exists():
            with tarfile.open(file_path_compressed, "r:gz") as tar:
                tar.extractall(root_path / "review_polarity")



class DatasetPreparation:
    _options = {"MR": MovieReviewProcessor}



    def load(self, dataset_initials: str):
        df_processor = self._options[dataset_initials]()
        df_processor.load()