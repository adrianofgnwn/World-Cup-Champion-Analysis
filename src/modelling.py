"""
Step 4: Machine Learning Modeling
==================================
Goal: Build classifiers to predict whether a team's match stats
      look like a World Cup champion's, and identify which features
      matter most.

Input:  data/team_match_featured.csv (86 rows from Step 2)
Output: outputs/ (model results, feature importance chart)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import warnings
warnings.filterwarnings('ignore')

matplotlib.use('Agg')

from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, f1_score, precision_score, recall_score
)

# -------------------------------------------------------
# 1. Load data and select features
# -------------------------------------------------------
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'team_match_featured.csv')
output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(data_path)

print("=" * 55)
print("STEP 4: MACHINE LEARNING MODELING")
print("=" * 55)

# Features we engineered in Step 2 + key raw stats
FEATURES = [
    # Raw stats
    'possession', 'shots', 'shots_on_target', 'corners',
    'xg', 'xg_against', 'opp_shots', 'opp_shots_on_target',
    # Engineered features
    'shot_conversion', 'xg_overperformance', 'shot_accuracy',
    'defensive_save_rate', 'defensive_xg_overperf',
    'goal_difference', 'shot_dominance', 'possession_effectiveness',
    'discipline_score', 'stage_level',
]

X = df[FEATURES]
y = df['is_champion']

print(f"\nFeatures: {len(FEATURES)}")
print(f"Samples:  {len(X)} ({y.sum()} champion, {(y==0).sum()} non-champion)")

# -------------------------------------------------------
# 2. Scale features
# -------------------------------------------------------
# Important: models like Logistic Regression need scaled features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=FEATURES)

print(f"\nFeatures scaled with StandardScaler (mean=0, std=1)")

# -------------------------------------------------------
# 3. Define models
# -------------------------------------------------------
models = {
    'Logistic Regression': LogisticRegression(
        class_weight='balanced',  # Handle imbalanced classes
        max_iter=1000,
        random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        max_depth=4,            # Keep it shallow — small dataset
        random_state=42
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    ),
}

# -------------------------------------------------------
# 4. Evaluate with cross-validation
# -------------------------------------------------------
# With only 86 samples, we use two strategies:
# - Stratified 5-Fold: standard, reliable
# - Leave-One-Out (LOO): uses maximum training data per fold

print(f"\n{'='*55}")
print("MODEL EVALUATION")
print(f"{'='*55}")

results = {}

# --- Stratified 5-Fold ---
print(f"\n--- Stratified 5-Fold Cross-Validation ---")
print(f"{'Model':<25} {'Accuracy':>10} {'F1':>10} {'Precision':>10} {'Recall':>10} {'ROC-AUC':>10}")
print("-" * 77)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    y_pred = cross_val_predict(model, X_scaled, y, cv=skf)
    y_prob = cross_val_predict(model, X_scaled, y, cv=skf, method='predict_proba')[:, 1]

    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    auc = roc_auc_score(y, y_prob)

    results[name] = {'accuracy': acc, 'f1': f1, 'precision': prec, 'recall': rec, 'auc': auc}
    print(f"{name:<25} {acc:>10.3f} {f1:>10.3f} {prec:>10.3f} {rec:>10.3f} {auc:>10.3f}")

# --- Leave-One-Out ---
print(f"\n--- Leave-One-Out Cross-Validation ---")
print(f"{'Model':<25} {'Accuracy':>10} {'F1':>10} {'Precision':>10} {'Recall':>10} {'ROC-AUC':>10}")
print("-" * 77)

loo = LeaveOneOut()
loo_results = {}

for name, model in models.items():
    y_pred = cross_val_predict(model, X_scaled, y, cv=loo)
    y_prob = cross_val_predict(model, X_scaled, y, cv=loo, method='predict_proba')[:, 1]

    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    auc = roc_auc_score(y, y_prob)

    loo_results[name] = {'accuracy': acc, 'f1': f1, 'precision': prec, 'recall': rec, 'auc': auc}
    print(f"{name:<25} {acc:>10.3f} {f1:>10.3f} {prec:>10.3f} {rec:>10.3f} {auc:>10.3f}")

# -------------------------------------------------------
# 5. Feature Importance (from Random Forest)
# -------------------------------------------------------
print(f"\n{'='*55}")
print("FEATURE IMPORTANCE (Random Forest)")
print(f"{'='*55}\n")

# Train on full data for feature importance
rf = RandomForestClassifier(
    n_estimators=100, class_weight='balanced',
    max_depth=4, random_state=42
)
rf.fit(X_scaled, y)

importances = pd.DataFrame({
    'feature': FEATURES,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

for _, row in importances.iterrows():
    bar = '█' * int(row['importance'] * 100)
    print(f"  {row['feature']:<28} {row['importance']:.4f}  {bar}")

# -------------------------------------------------------
# 6. Logistic Regression Coefficients
# -------------------------------------------------------
print(f"\n{'='*55}")
print("LOGISTIC REGRESSION COEFFICIENTS")
print(f"{'='*55}")
print("(Positive = more likely champion, Negative = less likely)\n")

lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
lr.fit(X_scaled, y)

coefs = pd.DataFrame({
    'feature': FEATURES,
    'coefficient': lr.coef_[0]
}).sort_values('coefficient', ascending=False)

for _, row in coefs.iterrows():
    sign = '+' if row['coefficient'] > 0 else ''
    print(f"  {row['feature']:<28} {sign}{row['coefficient']:.4f}")

# -------------------------------------------------------
# 7. Confusion Matrix (best model, LOO)
# -------------------------------------------------------
# Pick the model with the best F1 from LOO
best_model_name = max(loo_results, key=lambda k: loo_results[k]['f1'])
best_model = models[best_model_name]
y_pred_best = cross_val_predict(best_model, X_scaled, y, cv=loo)

print(f"\n{'='*55}")
print(f"CONFUSION MATRIX — {best_model_name} (LOO)")
print(f"{'='*55}\n")

cm = confusion_matrix(y, y_pred_best)
print(f"                  Predicted")
print(f"                  Non-Champ  Champion")
print(f"  Actual Non-Champ    {cm[0][0]:<10} {cm[0][1]}")
print(f"  Actual Champion     {cm[1][0]:<10} {cm[1][1]}")

print(f"\n  Classification Report:")
print(classification_report(y, y_pred_best, target_names=['Non-Champion', 'Champion']))

# -------------------------------------------------------
# 8. Visualization: Feature Importance
# -------------------------------------------------------
print("Generating feature importance chart...")

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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Random Forest importance
imp_sorted = importances.sort_values('importance', ascending=True)
colors = [GOLD if v > imp_sorted['importance'].median() else GRAY for v in imp_sorted['importance']]
ax1.barh(imp_sorted['feature'], imp_sorted['importance'], color=colors, alpha=0.85)
ax1.set_title('Random Forest: Feature Importance', fontsize=14, fontweight='bold', pad=15)
ax1.set_xlabel('Importance')
ax1.grid(axis='x', alpha=0.1)

# Logistic Regression coefficients
coef_sorted = coefs.sort_values('coefficient', ascending=True)
colors2 = [GOLD if v > 0 else '#E74C3C' for v in coef_sorted['coefficient']]
ax2.barh(coef_sorted['feature'], coef_sorted['coefficient'], color=colors2, alpha=0.85)
ax2.axvline(x=0, color='white', alpha=0.3, linestyle='--')
ax2.set_title('Logistic Regression: Coefficients', fontsize=14, fontweight='bold', pad=15)
ax2.set_xlabel('Coefficient (+ = Champion, - = Non-Champion)')
ax2.grid(axis='x', alpha=0.1)

fig.suptitle('What Makes a World Cup Champion? — Model Insights', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '06_feature_importance.png'), dpi=150, bbox_inches='tight')
plt.close()

# -------------------------------------------------------
# 9. Visualization: Model Comparison
# -------------------------------------------------------
print("Generating model comparison chart...")

fig, ax = plt.subplots(figsize=(12, 6))

model_names = list(loo_results.keys())
metrics = ['accuracy', 'f1', 'precision', 'recall', 'auc']
x = np.arange(len(model_names))
width = 0.15
colors_met = [GOLD, '#3498DB', '#2ECC71', '#E74C3C', '#9B59B6']

for i, metric in enumerate(metrics):
    vals = [loo_results[m][metric] for m in model_names]
    ax.bar(x + i * width, vals, width, label=metric.upper(), color=colors_met[i], alpha=0.85)

ax.set_xticks(x + width * 2)
ax.set_xticklabels(model_names, fontsize=12)
ax.set_ylabel('Score')
ax.set_ylim(0, 1.1)
ax.set_title('Model Performance Comparison (Leave-One-Out CV)', fontsize=14, fontweight='bold', pad=15)
ax.legend(loc='upper right', framealpha=0.3)
ax.grid(axis='y', alpha=0.1)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '07_model_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()

# -------------------------------------------------------
# 10. Summary
# -------------------------------------------------------
print(f"\n{'='*55}")
print("SUMMARY")
print(f"{'='*55}")
print(f"\nBest model (by F1): {best_model_name}")
print(f"  LOO Accuracy: {loo_results[best_model_name]['accuracy']:.1%}")
print(f"  LOO F1 Score: {loo_results[best_model_name]['f1']:.1%}")
print(f"  LOO ROC-AUC:  {loo_results[best_model_name]['auc']:.1%}")

print(f"\nTop 5 most important features (Random Forest):")
for i, (_, row) in enumerate(importances.head(5).iterrows()):
    print(f"  {i+1}. {row['feature']} ({row['importance']:.4f})")

print(f"\n📊 Charts saved to outputs/")