"""
Step 1: Data Preparation & Reshaping
=====================================
Goal: Transform the raw match-level dataset into a team-match-level dataset
      and label which teams are World Cup champions.

Input:  data/fifa_world_cup_enhanced_1974_2022.csv.csv (43 matches, one row per match)
Output: data/team_match_data.csv (86 rows, one row per team per match)
"""

import pandas as pd
import os

# -------------------------------------------------------
# 1. Load the raw data
# -------------------------------------------------------
raw_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'fifa_world_cup_enhanced_1974_2022.csv')
df = pd.read_csv(raw_path)

print("=" * 50)
print("STEP 1: DATA PREPARATION")
print("=" * 50)
print(f"\nLoaded raw data: {df.shape[0]} matches, {df.shape[1]} columns")
print(f"Years covered: {df['year'].min()} - {df['year'].max()}")

# -------------------------------------------------------
# 2. Reshape: one row per TEAM per match
# -------------------------------------------------------
# Problem: each row has home_goals AND away_goals mixed together
# Solution: split each match into TWO rows, one from each team's view
#
# Example: Brazil 2 - 0 Germany becomes:
#   Row 1: Brazil  | goals=2 | goals_conceded=0 | possession=52 ...
#   Row 2: Germany | goals=0 | goals_conceded=2 | possession=48 ...

rows = []

for _, match in df.iterrows():

    # --- Home team's row ---
    rows.append({
        'match_id':             match['match_id'],
        'year':                 match['year'],
        'date':                 match['date'],
        'stage':                match['stage'],
        'team':                 match['home_team'],
        'opponent':             match['away_team'],
        'is_home':              1,
        'goals':                match['home_goals'],
        'goals_conceded':       match['away_goals'],
        'xg':                   match['home_xg'],
        'xg_against':           match['away_xg'],
        'possession':           match['possession_home'],
        'shots':                match['shots_home'],
        'shots_on_target':      match['shots_ontarget_home'],
        'corners':              match['corners_home'],
        'fouls':                match['fouls_home'],
        'yellow_cards':         match['yellow_cards_home'],
        'red_cards':            match['red_cards_home'],
        'opp_shots':            match['shots_away'],
        'opp_shots_on_target':  match['shots_ontarget_away'],
        'won_match':            1 if match['winner'] == match['home_team'] else 0,
        'attendance':           match['attendance'],
    })

    # --- Away team's row (everything flipped) ---
    rows.append({
        'match_id':             match['match_id'],
        'year':                 match['year'],
        'date':                 match['date'],
        'stage':                match['stage'],
        'team':                 match['away_team'],
        'opponent':             match['home_team'],
        'is_home':              0,
        'goals':                match['away_goals'],
        'goals_conceded':       match['home_goals'],
        'xg':                   match['away_xg'],
        'xg_against':           match['home_xg'],
        'possession':           match['possession_away'],
        'shots':                match['shots_away'],
        'shots_on_target':      match['shots_ontarget_away'],
        'corners':              match['corners_away'],
        'fouls':                match['fouls_away'],
        'yellow_cards':         match['yellow_cards_away'],
        'red_cards':            match['red_cards_away'],
        'opp_shots':            match['shots_home'],
        'opp_shots_on_target':  match['shots_ontarget_home'],
        'won_match':            1 if match['winner'] == match['away_team'] else 0,
        'attendance':           match['attendance'],
    })

team_df = pd.DataFrame(rows)

print(f"\nReshaped: {df.shape[0]} matches → {team_df.shape[0]} team-match rows")

# -------------------------------------------------------
# 3. Label the champions
# -------------------------------------------------------
# Find who won the final each year = that year's World Cup champion

finals = df[df['stage'] == 'Final']
tournament_winners = {}

for _, final in finals.iterrows():
    tournament_winners[final['year']] = final['winner']

print(f"\n--- Tournament Winners ---")
for year, winner in sorted(tournament_winners.items()):
    print(f"  {year}: {winner}")

# Add champion label: 1 if team won the World Cup that year, 0 otherwise
team_df['is_champion'] = team_df.apply(
    lambda row: 1 if tournament_winners.get(row['year']) == row['team'] else 0,
    axis=1
)

# -------------------------------------------------------
# 4. Validation
# -------------------------------------------------------
print(f"\n--- Class Balance ---")
champ_count = team_df['is_champion'].sum()
non_champ_count = (team_df['is_champion'] == 0).sum()
print(f"  Champion rows:     {champ_count} ({champ_count/len(team_df)*100:.1f}%)")
print(f"  Non-champion rows: {non_champ_count} ({non_champ_count/len(team_df)*100:.1f}%)")

print(f"\n--- Champion Matches in Dataset ---")
champ_rows = team_df[team_df['is_champion'] == 1]
for year in sorted(champ_rows['year'].unique()):
    subset = champ_rows[champ_rows['year'] == year]
    team = subset['team'].iloc[0]
    matches = len(subset)
    opponents = ', '.join(subset['opponent'].tolist())
    print(f"  {year} {team} ({matches} matches): vs {opponents}")

print(f"\n--- Sample: Argentina 2022 ---")
sample = team_df[(team_df['team'] == 'Argentina') & (team_df['year'] == 2022)]
print(sample[['team', 'opponent', 'stage', 'goals', 'goals_conceded',
              'possession', 'xg', 'won_match', 'is_champion']].to_string(index=False))

# -------------------------------------------------------
# 5. Save
# -------------------------------------------------------
output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'team_match_data.csv')
team_df.to_csv(output_path, index=False)
print(f"\n✅ Saved to data/team_match_data.csv ({team_df.shape[0]} rows, {team_df.shape[1]} columns)")