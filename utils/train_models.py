import pickle
import pandas as pd
import os
import urllib.request
import zipfile
from surprise import Dataset, Reader, SVD, KNNBasic, accuracy
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse


def train_models():
    train_ratings, test_ratings = _load_ratings()
    _prepare_content_based_data()
    neighborhood_model = _train_neighborhood_model(train_ratings)
    matrix_factorization_model = _train_matrix_factorization_model(train_ratings)

    _evaluate_models(neighborhood_model, matrix_factorization_model, test_ratings)

    _save_model(neighborhood_model, "neighborhood_model")
    _save_model(matrix_factorization_model, "matrix_factorization_model")
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

    data_path = os.path.join(DATA_DIR, DATA_FILE, ".zip")

    if not os.path.exists(data_path):
        urllib.request.urlretrieve(DATA_URL, data_path)
        with zipfile.ZipFile(data_path, "r") as zip_ref:
            zip_ref.extractall(DATA_DIR)

    ratings_path = os.path.join(DATA_DIR, DATA_FILE, "ratings.csv")
    ratings_df = pd.read_csv(ratings_path)
    reader = Reader(line_format="user item rating timestamp", sep=",")
    data = Dataset.load_from_df(ratings_df[["userId", "movieId", "rating"]], reader)
    train_set, test_set = train_test_split(data, test_size=0.3, random_state=42)
    return train_set, test_set


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
    neighborhood_model = KNNBasic(sim_options={"name": "cosine", "user_based": False})
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


def _evaluate_models(neighborhood_model, matrix_factorization_model, test_ratings):
    """
    This function evaluates the trained models.

    Args:
    - neighborhood_model: The trained neighborhood model
    - matrix_factorization_model: The trained matrix factorization model
    - test_ratings: The test set
    """
    neighborhood_predictions = neighborhood_model.test(test_ratings)
    matrix_factorization_predictions = matrix_factorization_model.test(test_ratings)

    neighborhood_rmse = rmse(neighborhood_predictions)
    matrix_factorization_rmse = rmse(matrix_factorization_predictions)

    neighborhood_mae = accuracy.mae(neighborhood_predictions)
    matrix_factorization_mae = accuracy.mae(matrix_factorization_predictions)

    print(f"Neighborhood Model RMSE: {neighborhood_rmse}")
    print(f"Matrix Factorization Model RMSE: {matrix_factorization_rmse}")
    print(f"Neighborhood Model MAE: {neighborhood_mae}")
    print(f"Matrix Factorization Model MAE: {matrix_factorization_mae}")
    return neighborhood_rmse, matrix_factorization_rmse


def _prepare_content_based_data():
    """
    This function prepares the data for content-based filtering.
    """
    pass
