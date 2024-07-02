import pickle
from pprint import pprint

import numpy as np


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

    user_ratings = _map_user_ratings(user_ratings_input)
    cinema_movies = _map_cinema_movies(cinema_movies_input)

    scores_content_based = _make_content_based_recommendations(
        user_ratings, cinema_movies
    )
    scores_neighborhood_based = _make_neighborhood_based_recommendations(
        user_ratings, cinema_movies, neighborhood_model
    )
    scores_matrix_factorization = _make_matrix_factorization_recommendations(
        user_ratings, cinema_movies, matrix_factorization_model
    )

    scores = _combine_recommendations(
        scores_content_based,
        scores_neighborhood_based,
        scores_matrix_factorization,
    )

    sorted_scores = sorted(scores, key=lambda x: x["score"], reverse=True)

    return sorted_scores


def _make_content_based_recommendations(user_ratings, cinema_movies):
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


def _make_neighborhood_based_recommendations(user_ratings, cinema_movies, model):
    """
    This function makes recommendations using neighborhood-based collaborative filtering.

    Args:
    - user_ratings: The user ratings
    - cinema_movies: The cinema movies
    - model: The neighborhood-based model

    Returns:
    - scores: The scores for the cinema movies
    """
    return [
        {"externalId": "1", "score": 19, "title": "Test", "year": 2011, "movieId": 1},
    ]


def _make_matrix_factorization_recommendations(user_ratings, cinema_movies, model):
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


def _map_user_ratings(user_ratings_input):
    """
    This function maps the user ratings to the internal movie ids.

    Args:
    - user_ratings_input: The user ratings to be mapped

    Returns:
    - user_ratings: The user ratings with the internal movie ids
    """
    # TODO: Implement logic to map movie id
    user_ratings = []
    for rating in user_ratings_input:
        user_ratings.append({**rating, "movieId": int(rating["externalId"])})
    return user_ratings


def _map_cinema_movies(cinema_movies_input):
    """
    This function maps the cinema movies to the internal movie ids.

    Args:
    - cinema_movies_input: The cinema movies to be mapped

    Returns:
    - cinema_movies: The cinema movies with the internal movie ids
    """
    # TODO: Implement logic append movie id
    cinema_movies = []
    for movie in cinema_movies_input:
        cinema_movies.append({**movie, "movieId": int(movie["externalId"])})
    return cinema_movies


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
