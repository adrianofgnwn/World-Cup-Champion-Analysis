"""
Step 2: Feature Engineering
============================
Goal: Create meaningful derived features from the raw stats
      that better capture what makes a team perform like a champion.

Input:  data/team_match_data.csv (86 rows from Step 1)
Output: data/team_match_featured.csv (86 rows with new features)
"""

import pandas as pd
import os

# -------------------------------------------------------
# 1. Load the reshaped data from Step 1
# -------------------------------------------------------
data_path = os.path.join(os.path.dirname(__file__), '../..', 'data', 'team_match_data.csv')
df = pd.read_csv(data_path)

print("=" * 50)
print("STEP 2: FEATURE ENGINEERING")
print("=" * 50)
print(f"\nLoaded: {df.shape[0]} rows, {df.shape[1]} columns")

# -------------------------------------------------------
# 2. Attacking Features
# -------------------------------------------------------
# Shot Conversion Rate: what % of shots become goals?
# Champions are clinical — they don't just shoot, they score.
df['shot_conversion'] = (df['goals'] / df['shots'] * 100).round(2)

# xG Overperformance: goals scored minus expected goals
# Positive = scoring more than expected (clutch/clinical)
# Negative = wasteful in front of goal
df['xg_overperformance'] = (df['goals'] - df['xg']).round(2)

# Shots on Target Rate: what % of shots hit the target?
# Shows quality of chances, not just volume
df['shot_accuracy'] = (df['shots_on_target'] / df['shots'] * 100).round(2)

print("\n--- Attacking Features Created ---")
print(f"  shot_conversion:    goals / shots * 100")
print(f"  xg_overperformance: goals - xG")
print(f"  shot_accuracy:      shots_on_target / shots * 100")

# -------------------------------------------------------
# 3. Defensive Features
# -------------------------------------------------------
# Defensive Save Rate: what % of opponent shots did NOT go in?
# Higher = harder to score against
df['defensive_save_rate'] = ((1 - df['goals_conceded'] / df['opp_shots']) * 100).round(2)

# xG Against Overperformance: expected goals against minus actual goals conceded
# Positive = conceding LESS than expected (strong defense/keeper)
df['defensive_xg_overperf'] = (df['xg_against'] - df['goals_conceded']).round(2)

# Shots Allowed per Foul: how many opponent shots per foul committed
# Lower could mean smart tactical fouling
df['shots_allowed_per_foul'] = (df['opp_shots'] / df['fouls']).round(2)

print("\n--- Defensive Features Created ---")
print(f"  defensive_save_rate:    (1 - goals_conceded / opp_shots) * 100")
print(f"  defensive_xg_overperf:  xG_against - goals_conceded")
print(f"  shots_allowed_per_foul: opp_shots / fouls")

# -------------------------------------------------------
# 4. Dominance Features
# -------------------------------------------------------
# Goal Difference: simple but powerful
df['goal_difference'] = df['goals'] - df['goals_conceded']

# Shot Dominance: ratio of your shots vs opponent shots
# >1 means you're outshooting them
df['shot_dominance'] = (df['shots'] / df['opp_shots']).round(2)

# Corner Ratio: corners won vs opponent
# Shows how much pressure you're applying
df['corner_dominance'] = (df['corners'] / (df['corners'] + df['opp_shots_on_target'])).round(2)

# Possession Effectiveness: goals scored per % of possession
# Are you doing something useful with the ball?
df['possession_effectiveness'] = (df['goals'] / df['possession'] * 100).round(2)

print("\n--- Dominance Features Created ---")
print(f"  goal_difference:          goals - goals_conceded")
print(f"  shot_dominance:           shots / opp_shots")
print(f"  corner_dominance:         corners / (corners + opp_shots_on_target)")
print(f"  possession_effectiveness: goals / possession * 100")

# -------------------------------------------------------
# 5. Discipline Features
# -------------------------------------------------------
# Discipline Score: weighted card penalty
# Yellow = 1 point, Red = 3 points (harsher punishment)
df['discipline_score'] = df['yellow_cards'] + (df['red_cards'] * 3)

print("\n--- Discipline Features Created ---")
print(f"  discipline_score: yellow_cards + (red_cards * 3)")

# -------------------------------------------------------
# 6. Stage Encoding
# -------------------------------------------------------
# Encode knockout stages as a numeric "pressure level"
# Higher number = higher stakes match
stage_map = {
    'Group 1': 1, 'Group 2': 1, 'Group 3': 1,
    'Group A': 1, 'Group B': 1, 'Group C': 1,
    'Group D': 1, 'Group E': 1, 'Group F': 1,
    'Group H': 1,
    'Round of 16': 2,
    'Quarter-final': 3,
    'Semi-final': 4,
    'Final': 5,
}
df['stage_level'] = df['stage'].map(stage_map)

print(f"\n--- Stage Encoding ---")
print(f"  Group = 1, R16 = 2, QF = 3, SF = 4, Final = 5")

# -------------------------------------------------------
# 7. Validation — Compare champions vs non-champions
# -------------------------------------------------------
print(f"\n{'='*50}")
print("FEATURE COMPARISON: Champions vs Non-Champions")
print(f"{'='*50}")

new_features = [
    'shot_conversion', 'xg_overperformance', 'shot_accuracy',
    'defensive_save_rate', 'defensive_xg_overperf', 'goal_difference',
    'shot_dominance', 'possession_effectiveness', 'discipline_score'
]

champ = df[df['is_champion'] == 1]
non_champ = df[df['is_champion'] == 0]

print(f"\n{'Feature':<30} {'Champion':>10} {'Other':>10} {'Diff':>10}")
print("-" * 62)

for feat in new_features:
    c_avg = champ[feat].mean()
    nc_avg = non_champ[feat].mean()
    diff = c_avg - nc_avg
    sign = "+" if diff > 0 else ""
    print(f"{feat:<30} {c_avg:>10.2f} {nc_avg:>10.2f} {sign}{diff:>9.2f}")

# -------------------------------------------------------
# 8. Save
# -------------------------------------------------------
output_path = os.path.join(os.path.dirname(__file__), '../..', 'data', 'team_match_featured.csv')
df.to_csv(output_path, index=False)

print(f"\n✅ Saved to data/team_match_featured.csv ({df.shape[0]} rows, {df.shape[1]} columns)")
print(f"   New features added: {df.shape[1] - 23}")