import pandas as pd
import re
import os

def split_csv(file_path, rows_per_file):
    # Lade die CSV-Datei
    df = pd.read_csv(file_path)
    
    # Bestimme die Anzahl der zu erstellenden Dateien
    num_files = len(df) // rows_per_file + (1 if len(df) % rows_per_file != 0 else 0)
    
    for i in range(num_files):
        # Bestimme den Start- und Endindex für den aktuellen Split
        start_idx = i * rows_per_file
        end_idx = start_idx + rows_per_file
        
        # Erzeuge einen neuen DataFrame mit den entsprechenden Einträgen
        df_split = df[start_idx:end_idx]
        
        # Speichere den neuen DataFrame in eine neue CSV-Datei
        output_file = f'split_{i+1}.csv'
        df_split.to_csv(output_file, index=False)
        print(f'Datei {output_file} wurde erstellt.')

# Pfad zur Original-CSV-Datei und Anzahl der Zeilen pro Datei
file_path = 'C:\\Users\\I554814\\Documents\\Programmierprojekte\\movie-recommendation-model\\notebooks\\lisaTemporaryStuff\\cleaned_filme.csv'
rows_per_file = 500

# Aufrufen der Funktion
split_csv(file_path, rows_per_file)



# # Verzeichnis der aktuellen Python-Datei
# current_directory = os.path.dirname(os.path.abspath(__file__))

# # CSV-Datei einlesen
# input_file = os.path.join(current_directory, 'movies_no_genres_listed.csv')  # Ersetzen Sie 'input_filme.csv' durch den tatsächlichen Namen Ihrer CSV-Datei
# df = pd.read_csv(input_file)

# # Funktion, um den Originaltitel in Klammern zu entfernen
# def remove_original_title(title):
#     return re.sub(r'\s*\(.*?\)', '', title)

# # Den Originaltitel in der 'title'-Spalte entfernen
# df['title'] = df['title'].apply(remove_original_title)

# # Bereinigte Daten in eine neue CSV-Datei speichern
# output_file = os.path.join(current_directory, 'cleaned_filme.csv')
# df.to_csv(output_file, index=False)

# print("Der Originaltitel in Klammern wurde entfernt und die bereinigten Daten wurden gespeichert.")