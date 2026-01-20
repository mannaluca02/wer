# Fragestellung

> Mit welcher Wahrscheinlichkeit werden die Teams in der 15 Woche gegen ihren Gegner Gewinnen anhand den Informationen von erzielten Punkte, Record (Gewonnene und verlorene Spiele), Anzahl passing yards, Anzahl running yards, Anzahl turnover von den vorherigen 14 Wochen (13 Spielen)?


# WER OMC Football prediction
Starte venv auf Mac
```
source .venv/bin/activate
```

Installiere alle Packages aus dem Requirements
```
pip install -r ./requirements.txt 
```

Jupyter Notebook zu Quarto File konvertieren
```
quarto convert script.ipynb
```
Quarto File zu Jupyter Notebook konvertieren
```
quarto convert script.qmd
```

# WER Hustle Informationbeschaffung
Aggregieren der Daten auf Teamebene
```python
import pandas as pd

# Daten einlesen
df = pd.read_csv("deine_datei.csv")

# Turnover berechnen
df['turnovers'] = df['interception'] + df['fumble_lost']

# yards, turnovers zusammenrechnen
agg = df.groupby(['game_id', 'posteam', 'away_team', 'home_team', 'week']).agg({
    'yards_gained': 'sum',
    'turnovers': 'sum',
    'away_score': 'first',
    'home_score': 'first'
}).reset_index()

# Punkte berechnen (für posteam)
def get_points_scored(row):
    return row['away_score'] if row['posteam'] == row['away_team'] else row['home_score']

def get_points_allowed(row):
    return row['home_score'] if row['posteam'] == row['away_team'] else row['away_score']

agg['points_scored'] = agg.apply(get_points_scored, axis=1)
agg['points_allowed'] = agg.apply(get_points_allowed, axis=1)

# Jetzt brauchst du für jeden Eintrag:
# team, week, yards_gained, turnovers, points_scored, points_allowed
team_game_stats = agg[['posteam', 'week', 'yards_gained', 'turnovers', 'points_scored', 'points_allowed']].rename(columns={'posteam': 'team'})
```

Teamstatistiken für Wochen 1–14
```python
# Nur Spiele bis Woche 14
team_stats = team_game_stats[team_game_stats['week'] <= 14]

# Mittelwerte pro Team
team_summary = team_stats.groupby('team').agg({
    'yards_gained': 'mean',
    'turnovers': 'mean',
    'points_scored': 'mean',
    'points_allowed': 'mean'
}).reset_index().rename(columns={
    'yards_gained': 'avg_yards',
    'turnovers': 'avg_turnovers',
    'points_scored': 'avg_scored',
    'points_allowed': 'avg_allowed'
})
```
Matchups Woche 15 extrahieren
```python
games_wk15 = df[df['week'] == 15][['away_team', 'home_team']].drop_duplicates()
```
Monte-Carlo Simulation:
- Abhängige Analyse
```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def simulate_matchup(team_a_stats, team_b_stats, sims=10000):
    score_diff = (team_a_stats['avg_scored'] - team_b_stats['avg_allowed']) \
               - (team_b_stats['avg_scored'] - team_a_stats['avg_allowed'])
    turnover_diff = team_b_stats['avg_turnovers'] - team_a_stats['avg_turnovers']
    yard_diff = team_a_stats['avg_yards'] - team_b_stats['avg_yards']
    
    combined_score = 0.4 * score_diff + 0.3 * yard_diff + 0.3 * turnover_diff
    p_win = sigmoid(combined_score)
    
    wins = np.sum(np.random.rand(sims) < p_win)
    return wins / sims
```
- Unabhängig wäre zum beispiel ?
```python
p_win_team_a = sigmoid(0.5 * avg_points + 0.3 * avg_yards - 0.2 * avg_turnovers)
```
Für alle Woche-15-Spiele anwenden
```python
results = []

for _, game in games_wk15.iterrows():
    away = game['away_team']
    home = game['home_team']
    
    if away not in team_summary['team'].values or home not in team_summary['team'].values:
        continue
    
    stats_away = team_summary[team_summary['team'] == away].iloc[0]
    stats_home = team_summary[team_summary['team'] == home].iloc[0]
    
    p_away_win = simulate_matchup(stats_away, stats_home)
    p_home_win = 1 - p_away_win
    
    results.append({
        'away_team': away,
        'home_team': home,
        'p_away_win': round(p_away_win, 4),
        'p_home_win': round(p_home_win, 4)
    })

df_results = pd.DataFrame(results)
print(df_results)
```
## Gewichtungen von score diff, yard diff und turnover diff
### Regressionsmodell zum bestimmen von Gewichtungen
[Breaking down the play: Exploring Factors Impacting NFL Team Win/Loss Percentages](https://www.stat.cmu.edu/capstoneresearch/315files_f24/team6.html)
## Modell zur Berechnung unabhängiger Wahrscheinlichkeiten
### Klassifikation: Gewinnwahrscheinlichkeit (Logistic Regression auf `won`)

| Eigenschaft | Beschreibung |
|------------|--------------|
| Ziel     | Ob ein Team das Spiel **gewinnt (=1)** oder **verliert (=0)** |
| Input    | Feature-Mittelwerte pro Team (z. B. Yards, Turnovers, Scores) |
| Output   | Wahrscheinlichkeit für **Sieg** |
| Vorteil  | Leicht interpretierbar: z. B. „Team A hat 72 % Siegwahrscheinlichkeit“ |
| Nachteil | Unterscheidet nicht, **wie knapp oder deutlich** der Sieg war |
| Tipp     | Guter Ansatz, wenn du einfach wissen willst: **"Wird Team X voraussichtlich gewinnen?"** |

Eine Monte-Carlo-Simulation ist in unserem Fall besonders sinnvoll, da wir bei der unabhängigen Betrachtung mit Mittelwerten arbeiten. Auf Basis der mittels multipler logistischer Regression modellierten Siegwahrscheinlichkeiten können wir tausende mögliche Spielausgänge simulieren und dadurch die Unsicherheit der Durchschnittswerte realistisch abbilden.


## Abhängig vs Unabhängige betrachtung

## Satz von Bayes
Erweiterungsidee: Integration des Satzes von Bayes

Eine Idee zur Erweiterung unseres Modells ist es, den Satz von Bayes als zusätzliches Element zu integrieren, um die Gewinnwahrscheinlichkeit eines Teams – insbesondere ab Woche 16 – noch präziser berechnen zu können.

Der Satz von Bayes ermöglicht es, bedingte Wahrscheinlichkeiten zu berücksichtigen. Das heißt, wir könnten die Wahrscheinlichkeit, dass ein Team gewinnt, dynamisch anpassen, basierend auf bereits bekannten Informationen wie z. B. Turnover-Differenz, Yard-Gewinne oder bisherigen Spielausgang seit Woche 15.

So ließe sich zum Beispiel abschätzen:

„Wie wahrscheinlich ist ein Sieg in Woche 16, wenn das Team bisher alle Spiele mit positiver Yards-Differenz und ohne Turnover verloren hat?“

Durch den Einsatz des Bayes’schen Satzes würde unser Modell um eine theoretisch fundierte, probabilistische Komponente ergänzt. Besonders hilfreich ist das, wenn sich im Saisonverlauf Muster abzeichnen, die in klassische Regressionsmodelle nur schwer integrierbar sind.
