# NFL Week 15 Prediction Project (WER OMC)

## Projektübersicht
Dieses Projekt analysiert Daten der NFL-Saison 2021, um die Gewinnwahrscheinlichkeiten aller Teams für die Spiele in **Woche 15** vorherzusagen. Die Prognose basiert ausschliesslich auf der Punkteverteilung der vorangegangenen 14 Wochen.

Ziel ist es, ein statistisch fundiertes Modell zu entwickeln, das nicht nur die Gesamtpunkte betrachtet, sondern die Art und Weise, *wie* Punkte erzielt werden (Touchdowns, Field Goals, Extrapunkte etc.), differenziert betrachtet. Anschliessend werden diese Wahrscheinlichkeitsverteilungen in einer Monte-Carlo-Simulation genutzt, um virtuelle Spielergebnisse zu generieren.

## Methodik

### 1. Datenaufbereitung
Der Datensatz (`nfl_data_all.csv`) enthält detaillierte Informationen auf Spielzugebene ("Play-by-Play"). Für die Analyse wurden diese Daten aggregiert, um für jedes Team pro Spiel die Anzahl der folgenden Ereignisse zu ermitteln:
*   **Touchdowns (TD)**
*   **Field Goals (FG)**
*   **Extra Points (PAT)**
*   **Two-Point Conversions (2PC)**
*   **Other Scores** (z.B. Safetys)

### 2. Statistische Modellierung
Anstatt einfach den Durchschnitt der Punkte zu nehmen, wurden für jede Punkteart Wahrscheinlichkeitsverteilungen an die historischen Daten angepasst:

*   **Touchdowns & Field Goals:** Diese diskreten Zähldaten wurden mit einer **Poisson-Verteilung** modelliert.
*   **Other Scores:** Da diese Ereignisse sehr selten sind (viele Nullen), wurde ein **Zero-Inflated Poisson (ZIP)** Modell verwendet, das die hohe Anzahl an Nullen besser abbildet.
*   **PATs & 2-Point Conversions:** Diese sind nicht unabhängig, sondern können nur nach einem Touchdown auftreten. Daher wurden sie **konditional** modelliert: Basierend auf der simulierten Anzahl an Touchdowns wird mittels fester Wahrscheinlichkeiten (basierend auf NFL-Durchschnitten) entschieden, ob ein PAT oder eine 2PC versucht wird und ob dieser Versuch erfolgreich ist.

### 3. Monte-Carlo-Simulation
Für jedes Spiel der Woche 15 wurden **10'000 Simulationen** durchgeführt. In jedem Durchlauf wurde für beide Teams eine Punktzahl basierend auf ihren individuellen Verteilungen gezogen. Der Gewinner wurde durch den Vergleich der durchschnittlichen Punktzahlen über alle Simulationen ermittelt.

## Projektstruktur

*   `script.ipynb`: Das Haupt-Notebook. Es enthält den gesamten Code für:
    *   Datenbereinigung und Feature Engineering.
    *   Explorative Datenanalyse (EDA).
    *   Fitting der statistischen Verteilungen.
    *   Durchführung der Monte-Carlo-Simulation.
    *   Auswertung und Visualisierung der Ergebnisse (Vergleich Vorhersage vs. Realität).
*   `data/`: Ordner für die Datensätze.
    *   `nfl_data_all.csv`: Der Rohdatensatz (Play-by-Play Daten).
*   `remove_unused_columns.py`: Ein Hilfsskript, um die grosse CSV-Datei auf die tatsächlich benötigten Spalten zu reduzieren und Speicherplatz zu sparen.
*   `requirements.txt`: Liste der benötigten Python-Bibliotheken.

## Installation & Ausführung

### Voraussetzungen
*   Python 3.x
*   Eine virtuelle Umgebung (empfohlen)

### Setup

1. **Virtuelle Umgebung erstellen und aktivieren:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # Mac/Linux
   # .venv\Scripts\activate   # Windows
   ```

2. **Abhängigkeiten installieren:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Daten vorbereiten (Optional):**
   Falls die Rohdaten-Datei sehr gross ist und Sie nur die relevanten Spalten behalten möchten:
   ```bash
   python3 remove_unused_columns.py
   ```

### Analyse starten
Die gesamte Analyse befindet sich im Jupyter Notebook:
1. Öffnen Sie `script.ipynb` in VS Code oder Jupyter Lab.
2. Führen Sie alle Zellen nacheinander aus ("Run All").

### Quarto Export (Optional)
Falls Quarto installiert ist, kann das Notebook in ein Dokument konvertiert werden:
```bash
quarto convert script.ipynb
```
