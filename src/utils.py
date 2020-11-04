# imports
import pandas as pd
import zipfile
import wget
import os

# load environment variables
from dotenv import load_dotenv
load_dotenv()
URL = os.getenv("URL")
ZIP_PATH = os.getenv("ZIP_PATH")
CSV_PATH = os.getenv("CSV_PATH")


def download_and_unzip(url: str = URL, zip_path: str = ZIP_PATH, csv_path: str = CSV_PATH):
    """Downloads the News Aggregator dataset from ```url``` and stores it
    as ```NewsAggregatorDataset.zip``` under ```zip_path```. Extracts the
    "newsCorpora.csv" member from the .zip and stores it under ```csv_path```.

    Args:
        url (str, optional): URL to the dataset. Defaults to URL.
        zip_path (str, optional): Path to the .zip target dir. Defaults to ZIP_PATH.
        csv_path (str, optional): Path to the .csv target file. Defaults to CSV_PATH.
    """

    if not os.path.isdir(zip_path):
        os.makedirs("data/zip")
    if not os.path.isdir(csv_path):
        os.makedirs("data/raw")

    zip_filepath = wget.download(url, out=zip_path)

    with zipfile.ZipFile(zip_filepath, "r") as zf:
        zf.extract(path=csv_path, member="newsCorpora.csv")


def load_dataset_from_csv(csv_path: str = CSV_PATH) -> pd.DataFrame:
    """Loads the News Aggregator dataset from .csv to pd.DataFrame.

    Args:
        csv_path (str, optional): Path to stored data. Defaults to CSV_PATH.

    Returns:
        pd.DataFrame: News Aggregator dataset as pd.DataFrame.
    """

    cols = ["id", "title", "url", "publisher", "category", "story", "hostname", "timestamp"]

    return pd.read_csv(csv_path, delimiter="\t", names=cols, index_col=0)
