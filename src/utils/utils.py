# imports
import zipfile
import logging
import wget
import os
from typing import Tuple, Dict
import pandas as pd
from sklearn.model_selection import train_test_split

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
        zf.extract(path="data/raw", member="newsCorpora.csv")


def load_dataset_from_csv(csv_path: str = CSV_PATH) -> pd.DataFrame:
    """Loads the News Aggregator dataset from .csv to pd.DataFrame.

    Args:
        csv_path (str, optional): Path to stored data. Defaults to CSV_PATH.

    Returns:
        pd.DataFrame: News Aggregator dataset as pd.DataFrame.
    """

    if not os.path.isfile(csv_path):
        logging.info(f"\n{csv_path} not found, downloading and extracting data from {URL}.\n")
        download_and_unzip()

    cols = ["id", "title", "url", "publisher", "category", "story", "hostname", "timestamp"]

    return pd.read_csv(csv_path, delimiter="\t", names=cols, index_col=0)


def get_train_test_data(news: pd.DataFrame, test_size: float) -> Tuple:
    """Splits the news dataset into training and testing.

    Args:
        news (pd.DataFrame): The dataset as pandas DataFrame.
        test_size (float): The size in percent of the test data.

    Returns:
        Tuple: A tuple of size (4,) with x_train, y_train, x_test, y_test
    """

    train, test = train_test_split(news, test_size=test_size)

    x_train, y_train = train["title"].values, train["category"].values
    x_test, y_test = test["title"].values, test["category"].values

    return x_train, y_train, x_test, y_test


def category_mapping() -> Dict:
    """Defines the mapping from encoded labels to decoded categories.

    Returns:
        Dict: Keys are encoded labels and values are decoded categories.
    """

    return {
        "b": "business",
        "t": "science and technology",
        "e": "entertainment",
        "m": "health"
    }
