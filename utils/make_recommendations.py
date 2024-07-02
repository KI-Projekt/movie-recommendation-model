from pprint import pprint


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
    print("Sorted scores:")
    print(sorted_scores)
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
        {"externalId": "2", "score": 18, "title": "Test2", "year": 2012, "movieId": 2},
        {"externalId": "3", "score": 17, "title": "Test3", "year": 2013, "movieId": 3},
        {"externalId": "4", "score": 16, "title": "Test4", "year": 2014, "movieId": 4},
        {"externalId": "5", "score": 15, "title": "Test5", "year": 2015, "movieId": 5},
        {"externalId": "6", "score": 14, "title": "Test6", "year": 2016, "movieId": 6},
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
        {"externalId": "2", "score": 18, "title": "Test2", "year": 2012, "movieId": 2},
        {"externalId": "3", "score": 17, "title": "Test3", "year": 2013, "movieId": 3},
        {"externalId": "4", "score": 16, "title": "Test4", "year": 2014, "movieId": 4},
        {"externalId": "5", "score": 15, "title": "Test5", "year": 2015, "movieId": 5},
        {"externalId": "6", "score": 14, "title": "Test6", "year": 2016, "movieId": 6},
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
    return [
        {"externalId": "1", "score": 19, "title": "Test", "year": 2011, "movieId": 1},
        {"externalId": "2", "score": 18, "title": "Test2", "year": 2012, "movieId": 2},
        {"externalId": "3", "score": 17, "title": "Test3", "year": 2013, "movieId": 3},
        {"externalId": "4", "score": 16, "title": "Test4", "year": 2014, "movieId": 4},
        {"externalId": "5", "score": 15, "title": "Test5", "year": 2015, "movieId": 5},
        {"externalId": "6", "score": 14, "title": "Test6", "year": 2016, "movieId": 6},
    ]


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
    return user_ratings_input


def _map_cinema_movies(cinema_movies_input):
    """
    This function maps the cinema movies to the internal movie ids.

    Args:
    - cinema_movies_input: The cinema movies to be mapped

    Returns:
    - cinema_movies: The cinema movies with the internal movie ids
    """
    # TODO: Implement logic append movie id
    return cinema_movies_input


def _get_trained_models():
    """
    This function loads trained models from disk.

    Returns:
    - neighborhood_model: The trained neighborhood model
    - matrix_factorization_model: The trained matrix factorization model
    """
    # TODO: Implement logic to load trained models
    return None, None
