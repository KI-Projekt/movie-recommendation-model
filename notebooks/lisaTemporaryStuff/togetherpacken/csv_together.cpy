import pandas as pd
import glob
import os

# Verzeichnis der aktuellen Python-Datei
current_directory = os.path.dirname(os.path.abspath(__file__))

# Liste der CSV-Dateien im aktuellen Verzeichnis
csv_files = glob.glob(os.path.join(current_directory, '*.csv'))

# Liste, um DataFrames zu speichern
dfs = []

# Jede CSV-Datei einlesen und zur Liste hinzufügen, außer der Python-Datei und der kombinierten Datei
for file in csv_files:
    if file.endswith('.csv') and not file.endswith('movies_with_genres.csv'):  # Sicherstellen, dass wir die resultierende Datei nicht wieder einlesen
        df = pd.read_csv(file)
        dfs.append(df)

# Alle DataFrames zu einem einzigen DataFrame zusammenführen
combined_df = pd.concat(dfs, ignore_index=True)

# # Zeilen mit leeren Genres entfernen
# filtered_df = combined_df.dropna(subset=['genres'])

# # Genres durch "|" trennen
# filtered_df['genres'] = filtered_df['genres'].str.replace(', ', '|')

# Gefilterte Daten in eine neue CSV-Datei speichern
combined_df.to_csv(os.path.join(current_directory, 'movies_with_genres.csv'), index=False)

print("CSV-Dateien wurden erfolgreich zusammengeführt und gefiltert, und die Daten wurden gespeichert.")
