import pickle
import threading
import traceback
import pandas as pd
import utils.status as status
from surprise import accuracy
from surprise.accuracy import rmse
from sklearn.model_selection import train_test_split as sk_train_test_split
from utils.make_recommendations import (
    make_matrix_factorization_recommendations,
    make_neighborhood_based_recommendations,
    make_content_based_recommendations,
    make_recommendations,
)


def start_evaluation():
    """
    This function starts the evaluation of the models.
    """
    try:
        training_thread = threading.Thread(target=_evaluate)
        training_thread.start()
        return "Model evaluation started"
    except Exception as e:
        return f"Model evaluation failed: {str(e)}"


def _evaluate():
    """
    This function evaluates the models.
    """
    try:
        status.is_evaluating = True
        _evaluate_models()
        rmse_matrix_factorization, mae_matrix_factorization = (
            _evaluate_model_predictions("matrix_factorization")
        )
        status.rmse_matrix_factorization = rmse_matrix_factorization
        status.mae_matrix_factorization = mae_matrix_factorization

        rmse_neighborhood, mae_neighborhood = _evaluate_model_predictions(
            "neighborhood"
        )
        status.rmse_neighborhood = rmse_neighborhood
        status.mae_neighborhood = mae_neighborhood

        rmse_content_based, mae_content_based = _evaluate_model_predictions(
            "content_based"
        )
        status.rmse_content_based = rmse_content_based
        status.mae_content_based = mae_content_based

        rmse_all, mae_all = _evaluate_all()
        status.rmse_all = rmse_all
        status.mae_all = mae_all

    except Exception as e:
        print(f"Model evaluation failed: {str(e)}")
        print(traceback.format_exc())
    finally:
        status.is_evaluating = False
        print("Model evaluation finished")


def _evaluate_models():
    """
    This function evaluates the trained models with user in model.

    Args:
    - neighborhood_model: The trained neighborhood model
    - matrix_factorization_model: The trained matrix factorization model
    - test_ratings: The test set
    """
    neighborhood_model = pickle.load(open("./models/neighborhood_model.pkl", "rb"))
    matrix_factorization_model = pickle.load(
        open("./models/matrix_factorization_model.pkl", "rb")
    )
    test_ratings = pd.read_csv("./data/ml-latest-small/test_set_ratings.csv")

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


def _evaluate_model_predictions(model_type):
    """
    This function evaluates the predictions of a trained model.

    Args:
    - model: The trained model
    - ratings: The test set
    """

    def _get_user_ratings_per_user(ratings_df):
        """
        This function groups the ratings by user.

        Args:
        - ratings_df: The ratings data
        """
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
            # TODO: Remove from array, that score is calulated correctly
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
        elif model_type == "content_based":
            score = make_content_based_recommendations(
                user_ratings_input, movies_to_test
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
                    rmse_score += ((movie["rating"] - movie["score"]) / 20) ** 2
                    mae_score += abs(movie["rating"] - movie["score"]) / 20
                    ctr += 1
        rmse_score = (rmse_score / ctr) ** 0.5
        mae_score = mae_score / ctr
        print(f"Evaluation results for model {model_type}:")
        print(f"RMSE: {rmse_score}")
        print(f"MAE: {mae_score}")
        print("\n")
        return rmse_score, mae_score

    return _calculate_rmse(results)


def _evaluate_all():
    """
    This function evaluates the predictions of a trained model.
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
                    "externalId": str(movie_id),
                }
            )

        return user_ratings

    ratings_df = pd.read_csv("./data/ml-latest-small/test_set_ratings.csv")
    input_data, test_data = sk_train_test_split(
        ratings_df, test_size=0.3, random_state=42
    )

    user_ratings_input = _get_user_ratings_per_user(test_data)
    user_ratings_test = _get_user_ratings_per_user(input_data)

    results = []

    for user_id, user_ratings_input in user_ratings_input.items():
        if user_id not in user_ratings_test:
            # TODO: Remove from array, that score is calulated correctly
            continue

        movies_to_test = user_ratings_test[user_id]

        score = make_recommendations(user_ratings_input, movies_to_test, True)

        user_results = []
        for movie in score:
            rating_for_movie = next(
                (
                    test_movie
                    for test_movie in movies_to_test
                    if test_movie["externalId"] == movie["externalId"]
                    and "rating" in test_movie
                ),
                None,
            )
            user_results.append(
                {
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
                    rmse_score += ((movie["rating"] - movie["score"]) / 20) ** 2
                    mae_score += abs(movie["rating"] - movie["score"]) / 20
                    ctr += 1
        rmse_score = (rmse_score / ctr) ** 0.5
        mae_score = mae_score / ctr
        print("\nEvaluation results for all models:")
        print(f"RMSE: {rmse_score}")
        print(f"MAE: {mae_score}")
        print("\n")
        return rmse_score, mae_score

    rmse, mae = _calculate_rmse(results)
    print("\nEvaluation results for all models:")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print("\n")

    return rmse, mae
