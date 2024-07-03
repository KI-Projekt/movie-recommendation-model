from pprint import pprint
import pandas as pd

movie_path = "data/ml-latest-small/movies.csv"
movies_with_genres_path = "data/movies_with_genres.csv"
movie_with_year_path = "data/movies_with_year.csv"


def check_movies_without_genre():
    # Lese die CSV-Dateien
    movies_df = pd.read_csv(movie_path)
    movies_with_genres_df = pd.read_csv(movies_with_genres_path)

    # Entferne das Erscheinungsjahr aus dem Titel
    movies_df["title"] = movies_df["title"].str.replace(r"\s*\(\d{4}\)", "", regex=True)

    # Filtere die Filme ohne Genre
    movies_without_genre = movies_df[movies_df["genres"] == "(no genres listed)"]
    print(
        "the number of movies without genre is: ",
        movies_without_genre.__len__(),
    )

    # Überprüfe, ob die Filme in der zweiten Datei vorhanden sind
    is_in_movies_with_genres = movies_without_genre["title"].isin(
        movies_with_genres_df["title"]
    )

    # print(is_in_movies_with_genres)
    # Filtere die Filme, die nicht in der zweiten Datei vorhanden sind
    movies_not_in_second_file = movies_without_genre[~is_in_movies_with_genres]

    # Gib die Filme zurück, die nicht in der zweiten Datei vorhanden sind
    return movies_not_in_second_file


def check_movies_without_year():
    # Lese die CSV-Dateien
    movies_df = pd.read_csv(movie_path)
    print("hallo")
    movie_with_year_df = pd.read_csv(movie_with_year_path, sep=";")

    # Extrahiere das Erscheinungsjahr aus dem Titel und füge es als neue Spalte hinzu
    movies_df["year"] = movies_df["title"].str.extract(r"\((\d{4})\)")

    # Entferne das Erscheinungsjahr aus dem Titel
    movies_df["title"] = movies_df["title"].str.replace(r"\s*\(\d{4}\)", "", regex=True)

    # Filme ohne Erscheinungsjahr ausgeben
    movies_without_year = movies_df[movies_df["year"].isna()]
    print("the number of movies without year is: ", movies_df["year"].isna().sum())

    # Überprüfe, ob die Filme in der zweiten Datei vorhanden sind
    is_in_movies_with_genres = movies_without_year["title"].isin(
        movie_with_year_df["title"]
    )

    # Filtere die Filme, die nicht in der zweiten Datei vorhanden sind
    movies_not_in_second_file = movies_without_year[~is_in_movies_with_genres]

    # Gib die Filme zurück, die nicht in der zweiten Datei vorhanden sind
    return movies_not_in_second_file


# Rufe die Funktion auf
print(check_movies_without_year())

# Rufe die Funktion auf
print(check_movies_without_genre())
