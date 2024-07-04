# CRISP-DM

## Allgemein
1. **Business Understanding (Geschäftsverständnis)**:
   - Ziel: Verstehen der Geschäftsziele und Anforderungen sowie Definition der Projektziele aus geschäftlicher Sicht.
   - Aktivitäten: 
     - Formulieren des Geschäftsziels
     - Bewertung der Situation
     - Festlegen der Projektziele
     - Erstellung eines Projektplans

2. **Data Understanding (Datenverständnis)**:
   - Ziel: Sammeln, beschreiben und untersuchen der Daten, um ein grundlegendes Verständnis der Daten zu erlangen.
   - Aktivitäten:
     - Datensammlung
     - Erste Datenerhebung und -analyse
     - Beschreiben der Daten
     - Überprüfen der Datenqualität

3. **Data Preparation (Datenvorbereitung)**:
   - Ziel: Vorbereiten der Daten für das Modellieren durch Auswahl, Bereinigung und Transformation der Daten.
   - Aktivitäten:
     - Auswahl der Daten
     - Bereinigung der Daten (z.B. Umgang mit fehlenden Werten)
     - Konstruktion von neuen Daten (Feature Engineering)
     - Formatierung der Daten

4. **Modeling (Modellierung)**:
   - Ziel: Anwenden verschiedener Modellierungstechniken und Kalibrierung der Parameter, um die besten Modelle zu entwickeln.
   - Aktivitäten:
     - Auswahl der Modellierungstechniken
     - Generierung von Testdesigns
     - Erstellen und Bewerten der Modelle

5. **Evaluation (Evaluierung)**:
   - Ziel: Überprüfen der Modelle hinsichtlich der Projektziele und Sicherstellen, dass sie die Geschäftsanforderungen erfüllen.
   - Aktivitäten:
     - Auswerten der Modellleistung
     - Überprüfen, ob das Modell die Geschäftsziele erfüllt
     - Entscheiden, ob das Modell in die nächste Phase überführt werden kann

6. **Deployment (Einsatz)**:
   - Ziel: Implementieren des Modells in der Praxis und Bereitstellung der Ergebnisse.
   - Aktivitäten:
     - Planung des Einsatzes
     - Überwachen und Warten des Modells
     - Dokumentation des Projekts
     - Übergabe der Endergebnisse an den Geschäftsbetrieb

## Movie Recommendations

1. **Business Understanding**:
   - Geschäftsproblem: 
   - Projektziel: 

2. **Data Understanding**:
   - Datenquellen: DataLens Movie DB und für genres und jahre OMDB
   - Erste Analyse: Ratings Verteilung (Score von 0 bis 5), evtl movies

3. **Data Preparation**:
   - Auswahl: Movies und ratings
   - Bereinigung: Title und jahr bereinigen, genre bereinigen


4. **Modeling**:
   - Techniken: Content-based und Collaborative (KNN and Matrix Factorization)
   - Testdesign: Erstellung von Trainings- und Testdatensätzen.
   - Modellerstellung: Training und Testen verschiedener Modelle und Auswahl des besten Modells basierend auf der Genauigkeit. (Matrix: SVD, NMF, ALS und bei content-based: 2,5 und hyperparamate tuning und overfitting) Kombination der Modelle und Gewichtung; 
  anpassung an eigenen use-case. bewerten der einzelnden modelle

5. **Evaluation**:
   - Auswertung: Überprüfung der Modellleistung anhand von Metriken wie RMSE und MAE. einzeln und gesamt
   - Geschäftsziele: 

6. **Deployment**:
   - Implementierung: 
   - Monitoring: 
   - Kommunikation: 

Durch die Einhaltung der CRISP-DM-Methode wird sichergestellt, dass das KI-Projekt systematisch und zielgerichtet durchgeführt wird und letztendlich einen echten geschäftlichen Mehrwert bietet.