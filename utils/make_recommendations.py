import pickle
from pprint import pprint
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

import numpy as np
import pandas as pd
from surprise import Dataset, Reader


def make_recommendations(user_ratings_input, cinema_movies_input):
    """
    This function makes movie recommendations for a user.

    Args:
    - user_ratings_input: The user ratings
    - cinema_movies_input: The cinema movies

    Returns:
    - sorted_scores: The sorted scores for the cinema movies
    """
    neighborhood_model, matrix_factorization_model = _get_trained_models()

    user_ratings, cinema_movies = _map_movie_ids(
        user_ratings_input, cinema_movies_input
    )

    scores_content_based = make_content_based_recommendations(
        user_ratings, cinema_movies
    )
    scores_neighborhood_based = make_neighborhood_based_recommendations(
        user_ratings, cinema_movies, neighborhood_model
    )
    scores_matrix_factorization = make_matrix_factorization_recommendations(
        user_ratings, cinema_movies, matrix_factorization_model
    )

    scores = _combine_recommendations(
        scores_content_based,
        scores_neighborhood_based,
        scores_matrix_factorization,
    )

    sorted_scores = sorted(scores, key=lambda x: x["score"], reverse=True)

    return sorted_scores


def make_content_based_recommendations(user_ratings, cinema_movies):
    """
    This function makes recommendations using content-based filtering.

    Args:
    - user_ratings: The user ratings
    - cinema_movies: The cinema movies

    Returns:
    - scores: The scores for the cinema movies
    """
    return [
        {"externalId": "1", "score": 19, "title": "Test", "year": 2011, "movieId": 1},
    ]


def make_neighborhood_based_recommendations(user_ratings, cinema_movies, model):
    """
    This function makes recommendations using neighborhood-based collaborative filtering.

    Args:
    - user_ratings: The user ratings
    - cinema_movies: The cinema movies
    - model: The neighborhood-based model

    Returns:
    - scores: The scores for the cinema movies
    """

    def _find_nearest_neighbors(user_rated_movies, model, n_similar=1):
        """
        Find a similar user based on the given user's ratings using a pre-trained KNNBasic model.

        Parameters:
        user_rated_movies (list of dicts): List of ratings by the user in the form [{'movieId': int, 'rating': float}].
        model (KNNBasic): The pre-trained Surprise KNNBasic model.
        n_similar (int): The number of similar users to find. Default is 1.

        Returns:
        list: List of similar user IDs.
        """
        trainset = model.trainset
        # Ensure the model is user-based and has been trained
        if not model.sim_options["user_based"]:
            raise ValueError("The model must be user-based.")
        if model.sim is None or not model.sim.size:
            raise ValueError("The model has not been trained.")

        # Convert user rated movies to inner ids
        inner_ids = [
            trainset.to_inner_iid(movie["movieId"])
            for movie in user_rated_movies
            if movie["movieId"] in trainset._raw2inner_id_items
        ]

        # Dummy user since KNNBasic does not directly support finding similar users based on their ratings
        # We will use the average similarity of the rated items to all other users as a proxy
        user_similarity = [0] * trainset.n_users
        for movie_inner_id in inner_ids:
            for other_user_id in range(trainset.n_users):
                user_similarity[other_user_id] += model.sim[movie_inner_id][
                    other_user_id
                ]

        # Average the similarity scores
        user_similarity = [sim / len(inner_ids) for sim in user_similarity]

        # Find the top n similar users
        similar_users = sorted(
            range(len(user_similarity)), key=lambda i: user_similarity[i], reverse=True
        )[:n_similar]

        # Convert inner user ids to raw user ids
        similar_users_raw = [
            trainset.to_raw_uid(inner_id) for inner_id in similar_users
        ]

        return similar_users_raw[0]

    nearest_user_id = _find_nearest_neighbors(user_ratings, model, n_similar=1)
    results = []

    for movie in cinema_movies:
        res = model.predict(nearest_user_id, movie["movieId"])
        results.append(
            {
                "movieId": movie["movieId"],
                "score": round(res.est * 20),
                "externalId": movie["externalId"],
                "title": movie["title"],
                "year": movie["year"],
            }
        )

    return results


def make_matrix_factorization_recommendations(user_ratings, cinema_movies, model):
    """
    This function makes recommendations using matrix factorization.

    Args:
    - user_ratings: The user ratings
    - cinema_movies: The cinema movies
    - model: The matrix factorization model

    Returns:
    - scores: The scores for the cinema movies
    """

    def _find_similar_user(user_rated_movies, model, n_similar=1):
        """
        Find a similar user based on the given user's ratings using a pre-trained model.

        Parameters:
        user_rated_movies (list of dicts): List of ratings by the user in the form [{'movieId': int, 'rating': float}].
        model (AlgoBase): The pre-trained Surprise model.
        n_similar (int): The number of similar users to find. Default is 1.

        Returns:
        list: List of similar user IDs.
        """
        # Map movie IDs to the internal item IDs used by the model
        trainset = model.trainset
        temp_user_ratings = [
            (trainset.to_inner_iid(movie["movieId"]), movie["rating"])
            for movie in user_rated_movies
            if movie["movieId"] in trainset._raw2inner_id_items
        ]

        # Get the latent factors for the items rated by the temporary user
        q_i = np.array([model.qi[item_id] for item_id, _ in temp_user_ratings])
        r_ui = np.array([rating for _, rating in temp_user_ratings])

        # Calculate the implicit factors (biases can be included if the model uses them)
        user_factors = np.linalg.lstsq(q_i, r_ui, rcond=None)[0]

        # Calculate the similarity of the temporary user to all other users
        similarities = []
        for other_inner_user_id in trainset.all_users():
            other_user_factors = model.pu[other_inner_user_id]
            similarity = np.dot(user_factors, other_user_factors)
            similarities.append((similarity, trainset.to_raw_uid(other_inner_user_id)))

        # Sort the similarities in descending order and get the top n_similar users
        similarities.sort(reverse=True, key=lambda x: x[0])
        similar_users = [uid for _, uid in similarities[:n_similar]]

        return similar_users[0]

    similar_user = _find_similar_user(user_ratings, model, n_similar=1)
    results = []

    for movie in cinema_movies:
        res = model.predict(similar_user, movie["movieId"])
        results.append(
            {
                "movieId": movie["movieId"],
                "score": round(res.est * 20),
                "externalId": movie["externalId"],
                "title": movie["title"],
                "year": movie["year"],
            }
        )

    return results


def _combine_recommendations(
    scores_content_based,
    scores_neighborhood_based,
    scores_matrix_factorization,
):
    """
    This function combines the scores from the different recommendation methods.

    Args:
    - scores_content_based: The scores from the content-based recommender
    - scores_neighborhood_based: The scores from the neighborhood-based recommender
    - scores_matrix_factorization: The scores from the matrix factorization recommender

    Returns:
    - scores: The combined scores
    """
    print("MATRIX FACTORIZATION")
    pprint(scores_matrix_factorization)
    print("NEIGHBORHOOD BASED")
    pprint(scores_neighborhood_based)
    print("CONTENT BASED")
    pprint(scores_content_based)
    scores = []
    for movie in scores_content_based:
        # Finden Sie das entsprechende movie in den anderen Listen
        neighborhood_movie = next(
            (
                item
                for item in scores_neighborhood_based
                if item["movieId"] == movie["movieId"]
            ),
            None,
        )
        matrix_movie = next(
            (
                item
                for item in scores_matrix_factorization
                if item["movieId"] == movie["movieId"]
            ),
            None,
        )

        # Überprüfen Sie, ob das movie in den anderen Listen gefunden wurde
        if neighborhood_movie is None or matrix_movie is None:
            continue

        # Berechnen Sie den Durchschnittsscore
        average_score = round(
            (movie["score"] + neighborhood_movie["score"] + matrix_movie["score"]) / 3
        )

        scores.append(
            {
                "externalId": movie["externalId"],
                "title": movie["title"],
                "year": movie["year"],
                "movieId": movie["movieId"],
                "score": average_score,
            }
        )

    print("AVERAGE SCORE")
    pprint(scores)
    return scores


def _get_trained_models():
    """
    This function loads trained models from disk.

    Returns:
    - neighborhood_model: The trained neighborhood model
    - matrix_factorization_model: The trained matrix factorization model
    """
    with open("./models/neighborhood_model.pkl", "rb") as f:
        neighborhood_model = pickle.load(f)
    with open("./models/matrix_factorization_model.pkl", "rb") as f:
        matrix_factoization_model = pickle.load(f)
    return neighborhood_model, matrix_factoization_model


def _normalize_title(title):
    """
    This function normalizes the title of a movie.

    Args:
    - title: The title of the movie

    Returns:
    - normalized_title: The normalized title
    """
    articles = ["the", "a", "an"]
    words = title.strip().split()
    if words[-1].strip(",").lower() in articles:
        return title.strip().lower()
    if words[0].lower() in articles:
        return ", ".join(words[1:]) + ", " + words[0].capitalize()
    return title.lower()


def _alternate_title_format(title):
    """
    This function returns an alternate title format for a movie.

    Args:
    - title: The title of the movie

    Returns:
    - alternate_title: The alternate title format
    """
    articles = ["the", "a", "an"]
    words = title.strip().split()
    if words[0].lower() in articles:
        return ", ".join(words[1:]) + ", " + words[0].capitalize()
    if words[-1].strip(",").lower() in articles:
        return words[-1].capitalize() + " " + " ".join(words[:-1]).replace(",", "")
    return title.lower()


# Funktion zur Zuordnung der IDs
def _map_movie_ids(user_ratings_input, cinema_movies_input):
    """
    This function maps the movie IDs to the user ratings and cinema movies.

    Args:
    - user_ratings_input: The user ratings
    - cinema_movies_input: The cinema movies

    Returns:
    - mapped_user_ratings: The user ratings with movie IDs
    - mapped_cinema_movies: The cinema movies with movie IDs
    """
    movies_df = pd.read_csv("./data/ml-latest-small/movies_processed.csv")
    # Normalisieren der Titelspalte des DataFrames
    movies_df["normalized_title"] = movies_df["title"].apply(_normalize_title)
    movies_df["alternate_title"] = movies_df["title"].apply(_alternate_title_format)

    def get_movie_id(title, year):
        normalized_title = _normalize_title(title)
        alternate_title = _alternate_title_format(title)

        filtered_movies = movies_df[
            (
                (movies_df["normalized_title"] == normalized_title)
                | (movies_df["alternate_title"] == alternate_title)
            )
            & (movies_df["year"] == year)
        ]
        if not filtered_movies.empty:
            return filtered_movies.iloc[0]["movieId"]
        else:
            print(f"{title} ({year}) not found")
            return None

    # IDs zu den Filmen im user_ratings_input Array hinzufügen
    mapped_user_ratings = []
    for rating in user_ratings_input:
        movie_id = get_movie_id(rating["title"], rating["year"])
        mapped_user_ratings.append({**rating, "movieId": movie_id})

    # IDs zu den Filmen im cinema_movies_input Array hinzufügen
    mapped_cinema_movies = []
    for movie in cinema_movies_input:
        movie_id = get_movie_id(movie["title"], movie["year"])
        mapped_cinema_movies.append({**movie, "movieId": movie_id})

    return mapped_user_ratings, mapped_cinema_movies
