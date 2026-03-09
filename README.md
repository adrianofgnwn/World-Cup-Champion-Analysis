# What Makes a World Cup Champion?

![Python](https://img.shields.io/badge/Python-3.14-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview

This project analyzes **FIFA World Cup matches from 1974–2022** to identify the statistical DNA of tournament champions. Using classification models, we examine how factors such as shot accuracy, xG performance, defensive solidity, possession, and match dominance contribute to whether a team performs like a **World Cup winner**.

We then apply this champion profile to **score 2026 World Cup contenders** using recent international match data from 16 sources, including WC qualifiers, friendlies, Nations League, Euro 2024, Copa America 2024, and AFCON.

## Research Questions

1. *What statistical profile separates World Cup champions from every other team in the tournament?*
2. *Which match-level features are most predictive of champion performance?*
3. *Which 2026 World Cup contenders most closely match the champion DNA?*

## Datasets

**Champion Analysis:**
* **Name:** FIFA World Cup Enhanced (1974–2022)
* **Source:** [Kaggle](https://www.kaggle.com/datasets/samyakrajbayar/fifa-world-cup)
* **Size:** 43 matches, 38 features → reshaped to 86 team-match rows
* **Target Variable:** `is_champion` (1 = World Cup winner that year, 0 = other)

**Contender Scoring (16 sources):**
* WC Qualifiers: Europe, South America, CONCACAF, Asia, Africa, Oceania
* International Friendlies: 2023, 2024, 2025
* UEFA Nations League: 2022–23, 2024–25
* Euro 2024 (match-level + tournament stats)
* Copa America 2024 (match-level + official possession stats)
* AFCON: 2023, 2025

## Project Pipeline

1. Data Loading & Preparation
2. Feature Engineering
3. Exploratory Data Analysis (EDA)
4. Model Training & Evaluation
5. Insights & Conclusions
6. 2026 Contender Scoring

## Results

### Champion Analysis

| Model | Accuracy | F1 Score | ROC-AUC |
| --- | --- | --- | --- |
| **Logistic Regression** | **0.779** | **0.655** | **0.847** |
| Random Forest | 0.779 | 0.596 | 0.806 |
| Gradient Boosting | 0.767 | 0.524 | 0.750 |

### 2026 Contender Scoring

| Rank | Team | Champion DNA Score | Matches Analyzed |
| --- | --- | --- | --- |
| 1 | Argentina | 97 | 34 |
| 2 | Japan | 90 | 32 |
| 3 | Senegal | 87 | 34 |
| 4 | Colombia | 78 | 38 |
| 5 | Spain | 78 | 35 |

*Note: Opposition quality varies by confederation — teams in stronger confederations (UEFA, CONMEBOL) face tougher opponents, which may deflate their scores relative to teams in AFC or CAF.*

## Key Findings

### The 4 Traits of a World Cup Champion

* **Clinical finishing** — Champions convert 15.1% of shots vs 9.7%
* **Defensive solidity** — 93.5% save rate vs 87.0%, conceding only 0.79 goals per match
* **The clutch factor** — Champions overperform their xG by +0.19, others underperform by -0.13
* **Match dominance** — +1.5 avg goal difference, 1.75x shot dominance ratio

### What Doesn't Matter
* **Discipline** — Virtually identical card averages between champions and non-champions

### 2026 Contender Insights
* **Argentina** profiles closest to a World Cup champion across 34 matches — clinical finishing, elite defense, strong xG overperformance
* Logistic Regression outperformed all other models on F1-score and ROC-AUC
* Shot Accuracy and Goal Difference were the top predictive features

## Project Structure

```
World-Cup-Champion-Analysis/
│
├── data/
│   ├── fifa_world_cup_enhanced_1974_2022.csv
│   ├── Euro_2024_Matches.csv
│   ├── Copa_2024_Matches.csv
│   ├── UEFA_Euro_2024_Tournament_Stats.csv
│   └── ... (qualifier, friendly, Nations League, AFCON files)
├── notebooks/
│   ├── world_cup_champion_analysis.ipynb
│   └── 2026_contender_scoring.ipynb
├── requirements.txt
└── README.md
```

## Requirements

Install dependencies using:

```
pip install -r requirements.txt
```

## Dependencies

* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn

## Author

Made as a portfolio project for a BSc Computer Science degree.

## License

This project is licensed under the MIT License.