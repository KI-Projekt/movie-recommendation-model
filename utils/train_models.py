import pickle
from pprint import pprint
import pandas as pd
import os
import urllib.request
import zipfile
from surprise import Dataset, Reader, SVD, KNNBasic, accuracy
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse
from sklearn.model_selection import train_test_split as sk_train_test_split
from utils.make_recommendations import (
    make_matrix_factorization_recommendations,
    make_neighborhood_based_recommendations,
)


def train_models():
    train_ratings, test_ratings = _load_ratings()
    _prepare_content_based_data()
    neighborhood_model = _train_neighborhood_model(train_ratings)
    matrix_factorization_model = _train_matrix_factorization_model(train_ratings)
    _evaluate_model_predictions("matrix_factorization")
    # _evaluate_model_predictions("neighborhood")

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

    print("\nEvaluation results with user in model:")
    print(f"Neighborhood Model RMSE: {neighborhood_rmse}")
    print(f"Matrix Factorization Model RMSE: {matrix_factorization_rmse}")
    print(f"Neighborhood Model MAE: {neighborhood_mae}")
    print(f"Matrix Factorization Model MAE: {matrix_factorization_mae}")
    return neighborhood_rmse, matrix_factorization_rmse


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


def _evaluate_model_predictions(model_type):
    """
    This function evaluates the predictions of a trained model.

    Args:
    - model: The trained model
    - ratings: The test set
    """

    def _get_user_ratings_per_user(ratings_df):
        user_ratings = {}
        for _, row in ratings_df.iterrows():
            user_id = row["userId"]
            movie_id = row["movieId"]
            rating = row["rating"]

            if user_id not in user_ratings:
                user_ratings[user_id] = []

            user_ratings[user_id].append(
                {
                    "movieId": movie_id,
                    "rating": rating,
                    "title": "",
                    "year": 1000,
                    "externalId": "",
                }
            )

        return user_ratings

    ratings_df = pd.read_csv("./data/ml-latest-small/test_set_ratings.csv")
    input_data, test_data = sk_train_test_split(
        ratings_df, test_size=0.3, random_state=42
    )
    if model_type == "neighborhood":
        model = pickle.load(open("./models/neighborhood_model.pkl", "rb"))
    elif model_type == "matrix_factorization":
        model = pickle.load(open("./models/matrix_factorization_model.pkl", "rb"))

    user_ratings_input = _get_user_ratings_per_user(test_data)
    user_ratings_test = _get_user_ratings_per_user(input_data)

    results = []

    for user_id, user_ratings_input in user_ratings_input.items():
        if user_id not in user_ratings_test:
            continue

        movies_to_test = user_ratings_test[user_id]

        if model_type == "neighborhood":
            score = make_neighborhood_based_recommendations(
                user_ratings_input, movies_to_test, model
            )

        elif model_type == "matrix_factorization":
            score = make_matrix_factorization_recommendations(
                user_ratings_input, movies_to_test, model
            )

        user_results = []
        for movie in score:
            rating_for_movie = next(
                (
                    test_movie
                    for test_movie in movies_to_test
                    if test_movie["movieId"] == movie["movieId"]
                    and "rating" in test_movie
                ),
                None,
            )
            user_results.append(
                {
                    "movieId": movie["movieId"],
                    "score": movie["score"],
                    "title": movie["title"],
                    "year": movie["year"],
                    "externalId": movie["externalId"],
                    "rating": rating_for_movie["rating"] * 20,
                }
            )
        results.append({"userId": user_id, "results": user_results})

    def _calculate_rmse(results):
        rmse_score = 0
        mae_score = 0
        ctr = 0
        for user_result in results:
            for movie in user_result["results"]:
                if "rating" in movie:
                    rmse_score += (movie["rating"] - movie["score"]) ** 2
                    mae_score += abs(movie["rating"] - movie["score"])
                    ctr += 1
        rmse_score = (rmse_score / ctr) ** 0.5
        mae_score = mae_score / ctr
        print(f"Evaluation results for model {model_type}:")
        print(f"RMSE: {rmse_score/20}")
        print(f"MAE: {mae_score/20}")
        print("\n")
        return rmse_score, mae_score

    return _calculate_rmse(results)
