# import pandas as pd
# import tag_mapping
# import warnings
# from collections import Counter

# warnings.filterwarnings("ignore")

# genome_tags = pd.read_csv('C:/Users/I554814/Documents/Programmierprojekte/movie-recommendation-model/data/ml-25m/genome-tags.csv')
# tags = genome_tags['tag']

# # Tag-Mapping alphabetisch sortieren
# sorted_tag_mapping = dict(sorted(tag_mapping.tag_mapping.items()))

# fehlende_tags = [tag for tag in tags if tag not in tag_mapping.tag_mapping]

# # for tag in fehlende_tags:
# #     print(tag)

# # print('Fertig!')

# # Prüfen auf Duplikate in den Schlüsseln des tag_mapping
# key_counts = Counter(tag_mapping.tag_mapping.keys())
# duplicates = {key: count for key, count in key_counts.items() if count > 1}

# if duplicates:
#     print("\nDuplikate in den Schlüsseln des tag_mapping gefunden:")
#     for key, count in duplicates.items():
#         print(f"{key}: {count} mal")
# else:
#     print("\nKeine Duplikate in den Schlüsseln des tag_mapping gefunden.")

import re

# Pfad zur tag_mapping.py Datei anpassen
file_path = 'c:/Users/I554814/Documents/Programmierprojekte/movie-recommendation-model/notebooks/tag_mapping.py'

# Datei tag_mapping.py einlesen
with open(file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

# Tag-Mapping-Dictionary initialisieren
tag_mapping = {}

# Regulärer Ausdruck zum Extrahieren von Schlüssel-Wert-Paaren
pattern = re.compile(r"'((?:\\'|[^'])*)':\s*'((?:\\'|[^'])*)',?")

# Zeilen verarbeiten und Dictionary befüllen
for line in lines:
    match = pattern.search(line)
    if match:
        key, value = match.groups()
        tag_mapping[key] = value

# Dictionary nach Schlüsseln sortieren
sorted_tag_mapping = dict(sorted(tag_mapping.items()))

# Pfad für die sortierte Datei anpassen
sorted_file_path = 'c:/Users/I554814/Documents/Programmierprojekte/movie-recommendation-model/notebooks/tag_mapping_sorted.py'

# Sortiertes Dictionary in die Datei schreiben
with open(sorted_file_path, 'w', encoding='utf-8') as file:
    file.write("tag_mapping = {\n")
    for key, value in sorted_tag_mapping.items():
        file.write(f"    '{key}': '{value}',\n")
    file.write("}\n")

print("Die Datei wurde erfolgreich sortiert und als 'tag_mapping_sorted.py' gespeichert.")