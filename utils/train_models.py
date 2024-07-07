import pickle
import threading
import pandas as pd
import os
import urllib.request
import zipfile
import time
import utils.status as status
from surprise import Dataset, Reader, SVD, KNNBasic
from surprise.model_selection import train_test_split

from utils.evaluate_models import (
    _evaluate_model_predictions,
    _evaluate_models,
    _evaluate_all,
)


def start_training():
    """
    This function starts the training of the models.
    """
    try:
        # Starten Sie das Training in einem separaten Thread und übergeben Sie die Parameter
        training_thread = threading.Thread(target=_train_models)
        training_thread.start()

        # Geben Sie sofort nach dem Start des Trainings-Threads einen Erfolgstring zurück
        return "Model training started successfully"
    except Exception as e:
        # Wenn ein Fehler auftritt, geben Sie den Fehlerstring zurück
        return f"Model training failed: {str(e)}"


def _train_models():
    status.is_training = True
    print("Model training started")
    try:
        train_ratings = _load_ratings()
        _prepare_content_based_data()
        neighborhood_model = _train_neighborhood_model(train_ratings)
        matrix_factorization_model = _train_matrix_factorization_model(train_ratings)

        _save_model(neighborhood_model, "neighborhood_model")
        _save_model(matrix_factorization_model, "matrix_factorization_model")
    finally:
        status.is_training = False
        print("Model training finished")
    return "Model trained successfully"


def _load_ratings():
    """
    This function loads the ratings from disk.

    Returns:
    - train_set: The training set
    - test_set: The test set
    """
    DATA_FILE = "ml-latest-small"
    DATA_URL = f"https://files.grouplens.org/datasets/movielens/{DATA_FILE}.zip"
    DATA_DIR = "./data"

    data_path = os.path.join(DATA_DIR, f"{DATA_FILE}.zip")

    if not os.path.exists(data_path):
        urllib.request.urlretrieve(DATA_URL, data_path)
        with zipfile.ZipFile(data_path, "r") as zip_ref:
            zip_ref.extractall(DATA_DIR)

    ratings_path = os.path.join(DATA_DIR, DATA_FILE, "ratings.csv")
    ratings_df = pd.read_csv(ratings_path)
    reader = Reader(line_format="user item rating timestamp", sep=",")
    data = Dataset.load_from_df(ratings_df[["userId", "movieId", "rating"]], reader)
    train_set, test_set = train_test_split(data, test_size=0.3, random_state=42)

    # Save the train and test sets as CSV
    train_path = os.path.join(DATA_DIR, DATA_FILE, "train_set_ratings.csv")
    test_path = os.path.join(DATA_DIR, DATA_FILE, "test_set_ratings.csv")
    train_df = pd.DataFrame(
        train_set.all_ratings(), columns=["userId", "movieId", "rating"]
    )
    test_df = pd.DataFrame(test_set, columns=["userId", "movieId", "rating"])
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    return train_set


def _save_model(model, model_name):
    """
    This function saves the trained model to disk.

    Args:
    - model: The trained model to be saved
    - model_name: The name of the model
    """
    if not os.path.exists("./models"):
        os.makedirs("./models")
    model_path = f"./models/{model_name}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)


def _train_neighborhood_model(ratings):
    """
    This function trains a neighborhood model.

    Args:
    - ratings: The ratings data as Surprise dataset

    Returns:
    - neighborhood_model: The trained neighborhood model
    """
    neighborhood_model = KNNBasic(sim_options={"name": "cosine", "user_based": True})
    neighborhood_model.fit(ratings)
    return neighborhood_model


def _train_matrix_factorization_model(ratings):
    """
    This function trains a matrix factorization model.

    Args:
    - ratings: The ratings data as Surprise dataset

    Returns:
    - matrix_factorization_model: The trained matrix factorization model
    """
    matrix_factorization_model = SVD(random_state=42)
    matrix_factorization_model.fit(ratings)
    return matrix_factorization_model


def _prepare_content_based_data():
    """
    This function prepares the data for content-based filtering.
    """
    movies_df = pd.read_csv("./data/ml-latest-small/movies.csv")

    movies_df["year"] = (
        movies_df["title"].str.extract(r"(\d{4})", expand=False).astype(str)
    )
    movies_df["title"] = movies_df["title"].str.replace(
        r"\s*\((\d{4})\)", "", regex=True
    )

    # Save the processed data
    movies_df.to_csv("./data/ml-latest-small/movies_processed.csv", index=False)
