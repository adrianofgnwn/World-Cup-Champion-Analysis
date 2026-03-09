"""
Step 5: Insights & Final Summary
==================================
Goal: Pull together all findings into a final report with:
      - The champion DNA profile
      - Key statistical differences
      - Model results summary
      - 2026 World Cup contender scoring

Input:  data/team_match_featured.csv
Output: outputs/final_report.txt, outputs/09_champion_dna.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import warnings
warnings.filterwarnings('ignore')

matplotlib.use('Agg')

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

# -------------------------------------------------------
# 1. Load data
# -------------------------------------------------------
data_path = os.path.join(os.path.dirname(__file__), '../..', 'data', 'team_match_featured.csv')
output_dir = os.path.join(os.path.dirname(__file__), '../..', 'outputs')
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(data_path)
champ = df[df['is_champion'] == 1]
non_champ = df[df['is_champion'] == 0]

print("=" * 60)
print("STEP 5: INSIGHTS & FINAL SUMMARY")
print("=" * 60)

# -------------------------------------------------------
# 2. Build the Champion DNA Profile
# -------------------------------------------------------
# These are the benchmarks a team needs to hit to "look like a champion"

profile_features = {
    'Goals per Match':          ('goals',                champ['goals'].mean()),
    'Goals Conceded':           ('goals_conceded',       champ['goals_conceded'].mean()),
    'Possession %':             ('possession',           champ['possession'].mean()),
    'Shot Conversion %':        ('shot_conversion',      champ['shot_conversion'].mean()),
    'Shot Accuracy %':          ('shot_accuracy',        champ['shot_accuracy'].mean()),
    'xG per Match':             ('xg',                   champ['xg'].mean()),
    'xG Overperformance':       ('xg_overperformance',   champ['xg_overperformance'].mean()),
    'Defensive Save Rate %':    ('defensive_save_rate',  champ['defensive_save_rate'].mean()),
    'Def. xG Overperformance':  ('defensive_xg_overperf',champ['defensive_xg_overperf'].mean()),
    'Goal Difference':          ('goal_difference',      champ['goal_difference'].mean()),
    'Shot Dominance':           ('shot_dominance',       champ['shot_dominance'].mean()),
    'Discipline Score':         ('discipline_score',     champ['discipline_score'].mean()),
}

print(f"\n--- THE CHAMPION DNA PROFILE ---")
print(f"(Average per match across all World Cup winners 1974-2022)\n")
print(f"{'Metric':<30} {'Champion':>10} {'Others':>10} {'Edge':>10}")
print("-" * 62)

for label, (col, c_val) in profile_features.items():
    nc_val = non_champ[col].mean()
    diff = c_val - nc_val
    sign = '+' if diff > 0 else ''
    print(f"{label:<30} {c_val:>10.2f} {nc_val:>10.2f} {sign}{diff:>9.2f}")

# -------------------------------------------------------
# 3. Key Insights (narrative)
# -------------------------------------------------------
report_lines = []
report_lines.append("=" * 60)
report_lines.append("WHAT MAKES A WORLD CUP CHAMPION?")
report_lines.append("A Machine Learning Analysis of FIFA World Cup 1974-2022")
report_lines.append("=" * 60)

report_lines.append("\n1. THE QUESTION")
report_lines.append("-" * 40)
report_lines.append("What statistical profile separates World Cup champions")
report_lines.append("from every other team in the tournament?")
report_lines.append(f"We analyzed {len(df)} team-match performances across 13 World Cups.")

report_lines.append("\n2. THE CHAMPION DNA")
report_lines.append("-" * 40)
report_lines.append("Champions consistently show 4 key traits:\n")

report_lines.append("  TRAIT 1: CLINICAL FINISHING")
report_lines.append(f"  Champions convert {champ['shot_conversion'].mean():.1f}% of shots vs {non_champ['shot_conversion'].mean():.1f}%")
report_lines.append(f"  Shot accuracy: {champ['shot_accuracy'].mean():.1f}% vs {non_champ['shot_accuracy'].mean():.1f}%")
report_lines.append("  → They don't just shoot more, they score more efficiently.\n")

report_lines.append("  TRAIT 2: DEFENSIVE SOLIDITY")
report_lines.append(f"  Save rate: {champ['defensive_save_rate'].mean():.1f}% vs {non_champ['defensive_save_rate'].mean():.1f}%")
report_lines.append(f"  Goals conceded: {champ['goals_conceded'].mean():.2f} vs {non_champ['goals_conceded'].mean():.2f} per match")
report_lines.append(f"  Concede {champ['defensive_xg_overperf'].mean():.2f} fewer goals than expected (xG)")
report_lines.append("  → Champions are extremely hard to score against.\n")

report_lines.append("  TRAIT 3: THE CLUTCH FACTOR")
report_lines.append(f"  xG Overperformance: +{champ['xg_overperformance'].mean():.2f} vs {non_champ['xg_overperformance'].mean():.2f}")
report_lines.append("  → Champions score MORE than the stats say they should.")
report_lines.append("    Whether it's composure, star quality, or luck — they deliver.\n")

report_lines.append("  TRAIT 4: MATCH DOMINANCE")
report_lines.append(f"  Avg goal difference: +{champ['goal_difference'].mean():.1f} vs {non_champ['goal_difference'].mean():.1f}")
report_lines.append(f"  Shot dominance ratio: {champ['shot_dominance'].mean():.2f} vs {non_champ['shot_dominance'].mean():.2f}")
report_lines.append(f"  Possession: {champ['possession'].mean():.1f}% vs {non_champ['possession'].mean():.1f}%")
report_lines.append("  → Champions control matches, not just survive them.")

report_lines.append("\n3. WHAT DOESN'T MATTER")
report_lines.append("-" * 40)
report_lines.append(f"  Discipline: Champions avg {champ['discipline_score'].mean():.2f} vs {non_champ['discipline_score'].mean():.2f}")
report_lines.append("  → Virtually identical. Being 'cleaner' doesn't make you a champion.")

report_lines.append("\n4. ML MODEL RESULTS")
report_lines.append("-" * 40)
report_lines.append("  Best model: Logistic Regression")
report_lines.append("  Leave-One-Out CV Accuracy: 77.9%")
report_lines.append("  ROC-AUC: 84.7%")
report_lines.append("  Champion detection: 18/24 matches correctly identified (75%)")
report_lines.append("")
report_lines.append("  Top predictive features (Random Forest importance):")
report_lines.append("    1. Shot Accuracy")
report_lines.append("    2. Goal Difference")
report_lines.append("    3. Expected Goals (xG)")
report_lines.append("    4. Stage Level (knockout rounds)")
report_lines.append("    5. Defensive xG Overperformance")

report_lines.append("\n5. LIMITATIONS")
report_lines.append("-" * 40)
report_lines.append("  - Small dataset: 43 curated matches, not every World Cup game")
report_lines.append("  - Some champions have only 1 match in the data")
report_lines.append("  - xG data for older tournaments (1974-1990) may be estimated")
report_lines.append("  - Results reflect correlation, not necessarily causation")

report_lines.append("\n6. FOR THE 2026 WORLD CUP")
report_lines.append("-" * 40)
report_lines.append("  Teams to watch should match this champion profile:")
report_lines.append(f"    → Convert 15%+ of shots into goals")
report_lines.append(f"    → Keep 93%+ defensive save rate")
report_lines.append(f"    → Overperform their xG (the clutch factor)")
report_lines.append(f"    → Maintain 53%+ possession")
report_lines.append(f"    → Outshoot opponents by 1.7x ratio")
report_lines.append("")
report_lines.append("=" * 60)

# Save report
report_text = '\n'.join(report_lines)
report_path = os.path.join(output_dir, 'final_report.txt')
with open(report_path, 'w') as f:
    f.write(report_text)

print(report_text)

# -------------------------------------------------------
# 4. Final Visualization: The Champion DNA Card
# -------------------------------------------------------
print("\nGenerating champion DNA card...")

GOLD = '#D4AF37'
GOLD_LIGHT = '#F5E6A3'
GRAY = '#7A7A8A'
BG_COLOR = '#0F1117'
TEXT_COLOR = '#E8E8F0'
RED = '#E74C3C'
GREEN = '#2ECC71'

plt.rcParams.update({
    'figure.facecolor': BG_COLOR, 'axes.facecolor': BG_COLOR,
    'text.color': TEXT_COLOR, 'axes.labelcolor': TEXT_COLOR,
    'xtick.color': TEXT_COLOR, 'ytick.color': TEXT_COLOR,
    'font.family': 'sans-serif', 'font.size': 11,
})

fig = plt.figure(figsize=(16, 12))

# --- Top: Radar ---
ax_radar = fig.add_subplot(221, polar=True)

radar_metrics = [
    ('Shot\nConversion', champ['shot_conversion'].mean(), non_champ['shot_conversion'].mean(), 30),
    ('Shot\nAccuracy', champ['shot_accuracy'].mean(), non_champ['shot_accuracy'].mean(), 70),
    ('Defensive\nSave Rate', champ['defensive_save_rate'].mean(), non_champ['defensive_save_rate'].mean(), 100),
    ('Possession', champ['possession'].mean(), non_champ['possession'].mean(), 70),
    ('Shot\nDominance', champ['shot_dominance'].mean(), non_champ['shot_dominance'].mean(), 3),
    ('Goal\nDifference', champ['goal_difference'].mean() + 2, non_champ['goal_difference'].mean() + 2, 6),
]

angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
angles += angles[:1]

c_vals = [m[1]/m[3] for m in radar_metrics] + [radar_metrics[0][1]/radar_metrics[0][3]]
nc_vals = [m[2]/m[3] for m in radar_metrics] + [radar_metrics[0][2]/radar_metrics[0][3]]

ax_radar.plot(angles, c_vals, 'o-', color=GOLD, linewidth=2.5, label='Champions', markersize=6)
ax_radar.fill(angles, c_vals, color=GOLD, alpha=0.15)
ax_radar.plot(angles, nc_vals, 'o-', color=GRAY, linewidth=2, label='Others', markersize=5)
ax_radar.fill(angles, nc_vals, color=GRAY, alpha=0.08)

ax_radar.set_xticks(angles[:-1])
ax_radar.set_xticklabels([m[0] for m in radar_metrics], size=9)
ax_radar.set_ylim(0, 1)
ax_radar.set_yticklabels([])
ax_radar.grid(color='white', alpha=0.1)
ax_radar.set_title('Champion Profile', fontsize=13, fontweight='bold', pad=20)
ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), framealpha=0.3, fontsize=9)

# --- Top right: Key stats ---
ax_stats = fig.add_subplot(222)
ax_stats.axis('off')

stats_text = [
    ("THE 4 TRAITS OF A CHAMPION", GOLD, 16),
    ("", TEXT_COLOR, 8),
    ("1. CLINICAL FINISHING", GOLD_LIGHT, 13),
    (f"   15.1% shot conversion (vs 9.7%)", TEXT_COLOR, 11),
    ("", TEXT_COLOR, 6),
    ("2. DEFENSIVE WALL", GOLD_LIGHT, 13),
    (f"   93.5% save rate (vs 87.0%)", TEXT_COLOR, 11),
    ("", TEXT_COLOR, 6),
    ("3. THE CLUTCH FACTOR", GOLD_LIGHT, 13),
    (f"   +0.19 xG overperformance (vs -0.13)", TEXT_COLOR, 11),
    ("", TEXT_COLOR, 6),
    ("4. TOTAL DOMINANCE", GOLD_LIGHT, 13),
    (f"   +1.5 goal diff, 1.75x shot ratio", TEXT_COLOR, 11),
    ("", TEXT_COLOR, 10),
    ("MODEL: 77.9% accuracy | 84.7% AUC", GRAY, 10),
    ("18/24 champion matches detected", GRAY, 10),
]

y_pos = 0.95
for text, color, size in stats_text:
    ax_stats.text(0.05, y_pos, text, fontsize=size, color=color,
                  fontweight='bold' if size >= 13 else 'normal',
                  transform=ax_stats.transAxes, verticalalignment='top')
    y_pos -= 0.065

# --- Bottom left: Feature importance (top 8) ---
ax_imp = fig.add_subplot(223)

FEATURES = [
    'possession', 'shots', 'shots_on_target', 'corners',
    'xg', 'xg_against', 'opp_shots', 'opp_shots_on_target',
    'shot_conversion', 'xg_overperformance', 'shot_accuracy',
    'defensive_save_rate', 'defensive_xg_overperf',
    'goal_difference', 'shot_dominance', 'possession_effectiveness',
    'discipline_score', 'stage_level',
]

X_scaled = pd.DataFrame(StandardScaler().fit_transform(df[FEATURES]), columns=FEATURES)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', max_depth=4, random_state=42)
rf.fit(X_scaled, df['is_champion'])

imp = pd.DataFrame({'feature': FEATURES, 'importance': rf.feature_importances_})
imp = imp.sort_values('importance', ascending=True).tail(8)

colors = [GOLD if v > imp['importance'].median() else GRAY for v in imp['importance']]
ax_imp.barh(imp['feature'], imp['importance'], color=colors, alpha=0.85)
ax_imp.set_title('Top Predictive Features', fontsize=13, fontweight='bold', pad=10)
ax_imp.grid(axis='x', alpha=0.1)

# --- Bottom right: Champion timeline ---
ax_time = fig.add_subplot(224)

timeline = champ.groupby('year').agg({
    'goal_difference': 'mean',
    'shot_conversion': 'mean',
    'team': 'first',
}).reset_index()

ax_time.plot(timeline['year'], timeline['goal_difference'], 'o-', color=GOLD,
             linewidth=2, markersize=7, label='Goal Diff')
ax_time.plot(timeline['year'], timeline['shot_conversion']/10, 's--', color='#3498DB',
             linewidth=2, markersize=6, label='Shot Conv ÷10', alpha=0.8)

for _, row in timeline.iterrows():
    ax_time.annotate(row['team'], (row['year'], row['goal_difference']),
                     fontsize=7, color=TEXT_COLOR, alpha=0.6,
                     xytext=(0, 10), textcoords='offset points', ha='center', rotation=45)

ax_time.set_title('Champion Evolution', fontsize=13, fontweight='bold', pad=10)
ax_time.legend(framealpha=0.3, fontsize=9)
ax_time.grid(alpha=0.1)
ax_time.set_xlabel('Year')

fig.suptitle('What Makes a World Cup Champion? — Final Analysis',
             fontsize=18, fontweight='bold', y=1.02, color=GOLD)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '09_champion_dna.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"\n✅ Final report saved to outputs/final_report.txt")
print(f"✅ Champion DNA card saved to outputs/09_champion_dna.png")
print(f"\n🏆 Project complete!")