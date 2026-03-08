"""
Step 3: Exploratory Data Analysis & Visualizations
====================================================
Goal: Visually compare champions vs non-champions across all features
      to understand the "DNA" of a World Cup winner.

Input:  data/team_match_featured.csv (86 rows from Step 2)
Output: outputs/ (charts as PNG files)
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

matplotlib.use('Agg')  # For saving plots without display

# -------------------------------------------------------
# 1. Load featured data
# -------------------------------------------------------
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'team_match_featured.csv')
output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(data_path)

champ = df[df['is_champion'] == 1]
non_champ = df[df['is_champion'] == 0]

print("=" * 50)
print("STEP 3: EXPLORATORY DATA ANALYSIS")
print("=" * 50)
print(f"\nLoaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Champions: {len(champ)} rows | Non-champions: {len(non_champ)} rows")

# -------------------------------------------------------
# Style settings
# -------------------------------------------------------
GOLD = '#D4AF37'
GRAY = '#7A7A8A'
BG_COLOR = '#0F1117'
TEXT_COLOR = '#E8E8F0'

plt.rcParams.update({
    'figure.facecolor': BG_COLOR,
    'axes.facecolor': BG_COLOR,
    'text.color': TEXT_COLOR,
    'axes.labelcolor': TEXT_COLOR,
    'xtick.color': TEXT_COLOR,
    'ytick.color': TEXT_COLOR,
    'font.family': 'sans-serif',
    'font.size': 11,
})

# -------------------------------------------------------
# 2. Bar Chart: Champion vs Non-Champion averages
# -------------------------------------------------------
print("\n[1/5] Creating bar comparison chart...")

features = [
    ('shot_conversion', 'Shot Conversion %'),
    ('shot_accuracy', 'Shot Accuracy %'),
    ('xg_overperformance', 'xG Overperformance'),
    ('defensive_save_rate', 'Defensive Save Rate %'),
    ('defensive_xg_overperf', 'Defensive xG Overperf.'),
    ('goal_difference', 'Goal Difference'),
    ('shot_dominance', 'Shot Dominance Ratio'),
    ('possession_effectiveness', 'Possession Effectiveness'),
    ('discipline_score', 'Discipline Score'),
]

fig, ax = plt.subplots(figsize=(12, 7))

y_pos = np.arange(len(features))
champ_vals = [champ[f[0]].mean() for f in features]
non_champ_vals = [non_champ[f[0]].mean() for f in features]

bars1 = ax.barh(y_pos + 0.2, champ_vals, 0.35, color=GOLD, label='Champions', alpha=0.9)
bars2 = ax.barh(y_pos - 0.2, non_champ_vals, 0.35, color=GRAY, label='Non-Champions', alpha=0.7)

ax.set_yticks(y_pos)
ax.set_yticklabels([f[1] for f in features])
ax.set_title('Champion DNA: Average Stats Comparison', fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='lower right', framealpha=0.3)
ax.grid(axis='x', alpha=0.15)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '01_bar_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()

# -------------------------------------------------------
# 3. Radar Chart: Champion profile
# -------------------------------------------------------
print("[2/5] Creating radar chart...")

radar_features = [
    ('shot_conversion', 'Clinical\nFinishing', 30),
    ('shot_accuracy', 'Shot\nAccuracy', 70),
    ('defensive_save_rate', 'Defensive\nSolidity', 100),
    ('possession', 'Possession', 70),
    ('shot_dominance', 'Shot\nDominance', 3),
    ('xg_overperformance', 'Clutch\nFactor', 2),
]

# Normalize to 0-1 scale for radar
champ_radar = []
non_champ_radar = []
for feat, label, max_val in radar_features:
    c_val = champ[feat].mean() / max_val
    nc_val = non_champ[feat].mean() / max_val
    # Handle negative values (like xg_overperformance)
    c_val = max(0, min(1, (c_val + 0.5) if max_val <= 2 else c_val))
    nc_val = max(0, min(1, (nc_val + 0.5) if max_val <= 2 else nc_val))
    champ_radar.append(c_val)
    non_champ_radar.append(nc_val)

angles = np.linspace(0, 2 * np.pi, len(radar_features), endpoint=False).tolist()
angles += angles[:1]
champ_radar += champ_radar[:1]
non_champ_radar += non_champ_radar[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax.set_facecolor(BG_COLOR)
fig.set_facecolor(BG_COLOR)

ax.plot(angles, champ_radar, 'o-', color=GOLD, linewidth=2.5, label='Champions')
ax.fill(angles, champ_radar, color=GOLD, alpha=0.15)
ax.plot(angles, non_champ_radar, 'o-', color=GRAY, linewidth=2, label='Non-Champions')
ax.fill(angles, non_champ_radar, color=GRAY, alpha=0.1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels([f[1] for f in radar_features], size=10, color=TEXT_COLOR)
ax.set_ylim(0, 1)
ax.set_yticklabels([])
ax.grid(color='white', alpha=0.1)
ax.set_title('Champion Profile: Radar Comparison', fontsize=16, fontweight='bold', pad=30, color=TEXT_COLOR)
ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), framealpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '02_radar_chart.png'), dpi=150, bbox_inches='tight')
plt.close()

# -------------------------------------------------------
# 4. Scatter: xG vs Actual Goals
# -------------------------------------------------------
print("[3/5] Creating xG scatter plot...")

fig, ax = plt.subplots(figsize=(10, 8))

# Non-champions
ax.scatter(non_champ['xg'], non_champ['goals'], c=GRAY, alpha=0.5, s=60,
           label='Non-Champions', edgecolors='white', linewidth=0.5)
# Champions
ax.scatter(champ['xg'], champ['goals'], c=GOLD, alpha=0.9, s=100,
           label='Champions', edgecolors='white', linewidth=0.5, zorder=5)

# Perfect line (goals = xG)
max_val = max(df['xg'].max(), df['goals'].max()) + 0.5
ax.plot([0, max_val], [0, max_val], '--', color='white', alpha=0.3, label='Goals = xG')

# Annotate some champion data points
for _, row in champ.iterrows():
    if abs(row['goals'] - row['xg']) > 1.5 or row['goals'] >= 4:
        ax.annotate(f"{row['team']} {row['year']}", (row['xg'], row['goals']),
                    fontsize=8, color=GOLD, alpha=0.8,
                    xytext=(8, 5), textcoords='offset points')

ax.set_xlabel('Expected Goals (xG)', fontsize=13)
ax.set_ylabel('Actual Goals', fontsize=13)
ax.set_title('xG vs Actual Goals: Do Champions Overperform?', fontsize=16, fontweight='bold', pad=20)
ax.legend(framealpha=0.3)
ax.grid(alpha=0.1)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '03_xg_scatter.png'), dpi=150, bbox_inches='tight')
plt.close()

# -------------------------------------------------------
# 5. Box plots: Key feature distributions
# -------------------------------------------------------
print("[4/5] Creating distribution box plots...")

box_features = [
    ('shot_conversion', 'Shot Conversion %'),
    ('goal_difference', 'Goal Difference'),
    ('defensive_save_rate', 'Defensive Save Rate %'),
    ('xg_overperformance', 'xG Overperformance'),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, (feat, label) in enumerate(box_features):
    ax = axes[i]
    data = [non_champ[feat].values, champ[feat].values]
    bp = ax.boxplot(data, tick_labels=['Non-Champion', 'Champion'], patch_artist=True,
                    widths=0.5, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='white', markersize=6))

    bp['boxes'][0].set_facecolor(GRAY)
    bp['boxes'][0].set_alpha(0.6)
    bp['boxes'][1].set_facecolor(GOLD)
    bp['boxes'][1].set_alpha(0.7)

    for element in ['whiskers', 'caps', 'medians']:
        for line in bp[element]:
            line.set_color(TEXT_COLOR)
            line.set_alpha(0.7)

    ax.set_title(label, fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.1)

fig.suptitle('Feature Distributions: Champions vs Non-Champions', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '04_box_plots.png'), dpi=150, bbox_inches='tight')
plt.close()

# -------------------------------------------------------
# 6. Timeline: How champion profiles evolved
# -------------------------------------------------------
print("[5/5] Creating champion timeline...")

timeline = champ.groupby('year').agg({
    'goals': 'mean',
    'possession': 'mean',
    'shot_conversion': 'mean',
    'xg_overperformance': 'mean',
    'team': 'first',
}).reset_index()

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
axes = axes.flatten()

metrics = [
    ('goals', 'Avg Goals per Match'),
    ('possession', 'Avg Possession %'),
    ('shot_conversion', 'Shot Conversion %'),
    ('xg_overperformance', 'xG Overperformance'),
]

for i, (col, title) in enumerate(metrics):
    ax = axes[i]
    ax.plot(timeline['year'], timeline[col], 'o-', color=GOLD, linewidth=2, markersize=7)

    # Label each point with the team name
    for _, row in timeline.iterrows():
        ax.annotate(row['team'], (row['year'], row[col]),
                    fontsize=7, color=TEXT_COLOR, alpha=0.7,
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', rotation=45)

    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(alpha=0.1)
    ax.set_xlabel('Year')

fig.suptitle('Evolution of World Cup Champions (1974-2022)', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '05_champion_timeline.png'), dpi=150, bbox_inches='tight')
plt.close()

# -------------------------------------------------------
# Summary
# -------------------------------------------------------
print(f"\n{'='*50}")
print("KEY FINDINGS")
print(f"{'='*50}")
print(f"\n🏆 Champions score {champ['shot_conversion'].mean():.1f}% of their shots vs {non_champ['shot_conversion'].mean():.1f}%")
print(f"🛡️  Champions save {champ['defensive_save_rate'].mean():.1f}% of opponent shots vs {non_champ['defensive_save_rate'].mean():.1f}%")
print(f"🎯 Champions overperform xG by +{champ['xg_overperformance'].mean():.2f}, others underperform by {non_champ['xg_overperformance'].mean():.2f}")
print(f"⚽ Champions avg goal diff: +{champ['goal_difference'].mean():.1f} vs {non_champ['goal_difference'].mean():.1f}")
print(f"📊 All 5 charts saved to outputs/")