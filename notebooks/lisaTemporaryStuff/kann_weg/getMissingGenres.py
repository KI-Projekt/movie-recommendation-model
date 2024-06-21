import csv
import requests
import time
import pandas as pd


# api_key = 'd48fd14f'  #lisaress96 2
api_key = '7e0bda13'    #lisaresss96 2
# api_key = '4cab4c0a'  #t-online genutzt
# api_key = 'd3513930'  #müllmail 
# api_key = 'cdb4d160 '  #müllmail 
# api_key = 'c6863b28 '  #müllmail 2
# api_key = 'f1aad92e  '  #müllmail 2



def get_genre(title, year):
    url = f'http://www.omdbapi.com/?t={title}&y={year}&apikey={api_key}'
    try:
          response = requests.get(url)
          data = response.json()
          print(data)
          return data['Genre'] if 'Genre' in data else None
    except requests.exceptions.RequestException as e:
        print(f"Request exception for {title} ({year}): {e}")
        return None
    except ValueError as e:
        print(f"JSON decode error for {title} ({year}): {e}")
        return None
    


# Lesen der CSV-Datei mit den Filmtiteln
df = pd.read_csv('C:\\Users\\I554814\\Documents\\Programmierprojekte\\movie-recommendation-model\\split_4.csv')

# Neue DataFrame für die Ergebnisse
results = []

# Für jeden Film die Genres abrufen und in der neuen Liste speichern
for index, row in df.iterrows():
    genre = get_genre(row['title'], row['year'])
    results.append({'title': row['title'], 'year': row['year'], 'genres': genre})
    time.sleep(1)  # Wartezeit einfügen, um die API nicht zu überlasten

# Neue DataFrame aus den gesammelten Daten erstellen
new_df = pd.DataFrame(results)

# Neue CSV-Datei speichern
new_df.to_csv('filme_mit_genres4.csv', index=False)

print("Genres wurden erfolgreich abgerufen und in einer neuen Datei gespeichert.")