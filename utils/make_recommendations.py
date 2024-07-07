import pickle
from pprint import pprint
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd


def make_recommendations(user_ratings_input, cinema_movies_input, is_evaluation):
    """
    This function makes movie recommendations for a user.

    Args:
    - user_ratings_input: The user ratings
    - cinema_movies_input: The cinema movies
    - is_evaluation: A flag indicating whether the function is being called for evaluation

    Returns:
    - sorted_scores: The sorted scores for the cinema movies
    """
    neighborhood_model, matrix_factorization_model = _get_trained_models()

    if is_evaluation:
        user_ratings = user_ratings_input
        cinema_movies = cinema_movies_input
    else:
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


def make_content_based_recommendations(user_ratings_input, cinema_movies_input):
    """
    This function makes recommendations using content-based filtering.

    Args:
    - user_ratings: The user ratings
    - cinema_movies: The cinema movies

    Returns:
    - scores: The scores for the cinema movies
    """

    def _load_data():
        movies = pd.read_csv("./data/ml-latest-small/movies.csv")
        movies_with_genres = pd.read_csv("./data/movies_with_genres.csv")
        movies_with_year = pd.read_csv("./data/movies_with_year.csv", sep=";")

        # Extrahiere das Erscheinungsjahr aus dem Titel und füge es als neue Spalte hinzu
        movies["year"] = movies["title"].str.extract(r"\((\d{4})\)")

        # Entferne das Erscheinungsjahr aus dem Titel
        movies["title"] = movies["title"].str.replace(r"\s*\(\d{4}\)", "", regex=True)
        return movies, movies_with_genres, movies_with_year

    def _handle_missing_years(movies, movies_with_year):
        # Füge die fehlenden Jahre aus der neuen CSV-Datei ein, basierend auf den Titeln
        movies["year"] = movies.apply(
            lambda row: (
                movies_with_year[movies_with_year["title"] == row["title"]][
                    "year"
                ].values[0]
                if pd.isnull(row["year"])
                and row["title"] in movies_with_year["title"].values
                else row["year"]
            ),
            axis=1,
        )

        # Entferne Filme ohne Erscheinungsjahr
        movies = movies.dropna(subset=["year"])
        movies.isnull().sum()
        return movies

    def _prepare_genres(movies, movies_with_genres):
        # Zusammenführen der DataFrames basierend auf den Titeln und Jahren
        movies_updated = pd.merge(
            movies,
            movies_with_genres[["title", "year", "genres"]],
            on=["title", "year"],
            how="left",
            suffixes=("", "_new"),
        )

        # Aktualisieren der Genres im movies DataFrame nur für die übereinstimmenden Einträge
        movies_updated["genres"] = movies_updated["genres_new"].combine_first(
            movies_updated["genres"]
        )

        # Entferne die temporäre Spalte
        movies_updated = movies_updated.drop(columns=["genres_new"])
        movies = movies_updated

        # Entferne "(no genres listed)" aus der Genre-Liste
        movies["genres"] = movies["genres"].replace("(no genres listed)", "")

        # Trenne die Genres in separate Listen
        genre_list = movies["genres"].str.split("|")

        # Finde alle einzigartigen Genres
        all_genres = set(genre for sublist in genre_list for genre in sublist if genre)

        # Erstelle für jedes Genre eine Spalte und fülle sie mit binären Werten
        for genre in all_genres:
            movies[genre] = movies["genres"].apply(lambda x: int(genre in x.split("|")))

        # Entferne die ursprüngliche 'genres' Spalte
        movies = movies.drop(columns=["genres"])
        return movies

    def _get_movie_features(movies, movie_id):
        filtered_movie = movies[movies["movieId"] == movie_id]
        if not filtered_movie.empty:
            # Wähle nur die numerischen Spalten aus, die für die Berechnung der Ähnlichkeiten verwendet werden sollen, außer 'movieId'
            numeric_features = filtered_movie.drop(columns=["movieId"]).select_dtypes(
                include=[np.number]
            )
            return numeric_features.iloc[0]
        else:
            print(f"Movie with ID {movie_id} not found")
            return None

    movies, movies_with_genres, movies_with_year = _load_data()
    movies = _handle_missing_years(movies, movies_with_year)
    movies = _prepare_genres(movies, movies_with_genres)

    # Schritt 1: Extrahieren der Merkmale der vom Nutzer bewerteten Filme
    user_movie_features = []
    has_positive_ratings = any(
        "rating" in movie and movie["rating"] >= 2.5 for movie in user_ratings_input
    )

    for movie in user_ratings_input:
        if has_positive_ratings and movie.get("rating", 0) >= 2.5:
            features = _get_movie_features(movies, movie["movieId"])
            if features is not None:
                user_movie_features.append(features.values)
        elif not has_positive_ratings and movie.get("rating", 0) < 2.5:
            features = _get_movie_features(movies, movie["movieId"])
            if features is not None:
                user_movie_features.append(features.values)

    # Schritt 2: Feature-Vektorisierung
    if not user_movie_features:
        raise ValueError("Keine positiv bewerteten Filme vorhanden.")

    user_profile = np.mean(user_movie_features, axis=0)

    # Schritt 3: Ähnlichkeitsberechnung
    kino_movie_features = []
    for kino_movie in cinema_movies_input:
        features = _get_movie_features(movies, kino_movie["movieId"])
        if features is not None:
            kino_movie_features.append(features.values)

    if not kino_movie_features:
        raise ValueError("Keine Kino-Filme mit passenden Features gefunden.")

    similarities = cosine_similarity([user_profile], kino_movie_features)[0]
    similarities = similarities * 100

    # Schritt 4: Sortierung und Ausgabe
    # Falls der Nutzer positive Bewertungen abgegeben hat, sortiere nach absteigender Ähnlichkeit
    # Andernfalls sortiere nach aufsteigender Ähnlichkeit
    cinema_movies_with_similarity = []
    for i, kino_movie in enumerate(cinema_movies_input):
        kino_movie_with_similarity = {
            "externalId": kino_movie["externalId"],
            "title": kino_movie["title"],
            "year": kino_movie["year"],
            "score": similarities[i],
            "movieId": kino_movie["movieId"],
        }
        cinema_movies_with_similarity.append(kino_movie_with_similarity)

    if has_positive_ratings:
        sorted_cinema_movies = sorted(
            cinema_movies_with_similarity, key=lambda x: x["score"], reverse=True
        )
    else:
        sorted_cinema_movies = sorted(
            cinema_movies_with_similarity, key=lambda x: x["score"]
        )

    return sorted_cinema_movies


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

    def _find_nearest_neighbors(user_rated_movies, n_similar=1):
        """
        Find a similar user based on the given user's ratings using a pre-trained KNNBasic model.

        Parameters:
        user_rated_movies (list of dicts): List of ratings by the user in the form [{'movieId': int, 'rating': float}].
        model (KNNBasic): The pre-trained Surprise KNNBasic model.
        n_similar (int): The number of similar users to find. Default is 1.

        Returns:
        list: List of similar user IDs.
        """
        # Step 1: Load the Data
        data = pd.read_csv("./data/ml-latest-small/ratings.csv")

        # Step 2: Create a User-Item Matrix
        ratings_matrix = data.pivot_table(
            index="userId", columns="movieId", values="rating", fill_value=0
        )

        # Prepare the new user's ratings
        new_user_ratings = pd.Series(index=ratings_matrix.columns)

        for movie in user_rated_movies:
            movie_id = movie["movieId"]  # Use movieId to match the column
            new_user_ratings[movie_id] = movie["rating"]

        # Convert the Series to a DataFrame to append it
        new_user_df = pd.DataFrame([new_user_ratings.fillna(0)])

        # Append the new user's ratings to the ratings_matrix using pd.concat
        ratings_matrix = pd.concat([ratings_matrix, new_user_df], ignore_index=True)

        # Convert the updated DataFrame to a numpy array for similarity computation
        ratings_matrix_np = ratings_matrix.to_numpy()

        # Compute cosine similarities with the updated matrix
        user_similarities = cosine_similarity(ratings_matrix_np)

        # The new user is the last row in the matrix
        input_user_index = len(ratings_matrix_np) - 1
        input_user_similarity = user_similarities[input_user_index]

        # Ignore the similarity of the user to themselves by setting it to -1
        input_user_similarity[input_user_index] = -1

        # Find the nearest user
        nearest_user_index = np.argmax(input_user_similarity)
        return nearest_user_index

    nearest_user_id = _find_nearest_neighbors(user_ratings, n_similar=1)
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
