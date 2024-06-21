import csv
import requests
import time
import pandas as pd


api_key = 'd48fd14f'
# api_key = '7e0bda13'


def get_movie_year(title):
    url = f'http://www.omdbapi.com/?t={title}&apikey={api_key}'
    response = requests.get(url)
    data = response.json()
    print(data)
    return data['Year'] if 'Year' in data else None
    


# Lesen der CSV-Datei mit den Filmtiteln
movies_df = pd.read_csv('C:/Users/I554814/Documents/Programmierprojekte/movie-recommendation-model/notebooks/lisaTemporaryStuff/movies_without_year.csv')
# movies_df = movies_df.head(3)
print(movies_df)

# Neue DataFrame für die Ergebnisse
results = []

# Filme verarbeiten und Jahr abfragen
for title in movies_df['title']:
    print(title)
    year = get_movie_year(title)
    if year is not None:
        print(year)
        results.append({'title': title, 'year': year})
    time.sleep(1)  # kleine Pause, um die API nicht zu überlasten

#Ergebnisse in eine neue CSV-Datei schreiben
with open ("movies_with_year2.csv", "w+", encoding="utf-8", newline='') as file:
    writer = csv.DictWriter(file, fieldnames=["title", "year"], delimiter=';')
    writer.writeheader()
    writer.writerows(results)

print("Die neuen Daten wurden erfolgreich in 'movies_with_year2.csv' gespeichert.")