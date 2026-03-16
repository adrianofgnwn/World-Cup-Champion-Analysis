# FIFA World Cup 2026 — Official Tournament Structure
# Source: https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/standings
# Bracket: https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/articles/knockout-stage-match-schedule-bracket

# =============================================================================
# GROUPS (official from FIFA.com)
# Playoff spots filled with most likely winners
# =============================================================================

GROUPS = {
    "A": ["Mexico", "South Africa", "South Korea", "Denmark"],          # Denmark = UEFA PO-D winner (most likely)
    "B": ["Canada", "Qatar", "Switzerland", "Italy"],                    # Italy = UEFA PO-A winner (most likely)
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["USA", "Paraguay", "Australia", "Turkey"],                     # Turkey = UEFA PO-C winner (most likely)
    "E": ["Germany", "Curacao", "Cote d'Ivoire", "Ecuador"],
    "F": ["Netherlands", "Japan", "Tunisia", "Ukraine"],                 # Ukraine = UEFA PO-B winner (most likely)
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cabo Verde", "Saudi Arabia", "Uruguay"],
    "I": ["Iraq", "France", "Senegal", "Norway"],                       # Iraq = IC PO-2 winner (most likely)
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["DR Congo", "Portugal", "Uzbekistan", "Colombia"],            # DR Congo = IC PO-1 winner (most likely)
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

# Original playoff slots (for reference)
PLAYOFF_SLOTS = {
    "UEFA PO-A": {"group": "B", "candidates": ["Italy", "Northern Ireland", "Wales", "Bosnia and Herzegovina"], "most_likely": "Italy"},
    "UEFA PO-B": {"group": "F", "candidates": ["Ukraine", "Sweden", "Poland", "Albania"], "most_likely": "Ukraine"},
    "UEFA PO-C": {"group": "D", "candidates": ["Turkey", "Romania", "Slovakia", "Kosovo"], "most_likely": "Turkey"},
    "UEFA PO-D": {"group": "A", "candidates": ["Denmark", "North Macedonia", "Czechia", "Republic of Ireland"], "most_likely": "Denmark"},
    "IC PO-1":   {"group": "K", "candidates": ["New Caledonia", "Jamaica", "DR Congo"], "most_likely": "DR Congo"},
    "IC PO-2":   {"group": "I", "candidates": ["Bolivia", "Suriname", "Iraq"], "most_likely": "Iraq"},
}

# =============================================================================
# KNOCKOUT BRACKET (Round of 32)
# Format: 48 teams → top 2 per group + 8 best 3rd-place teams = 32 teams
# Source: Official FIFA bracket image
# =============================================================================

# LEFT HALF of bracket
LEFT_BRACKET = [
    # Round of 32 matchups (winner advances to R16)
    ("1E", "3rd_ABCDF"),      # Match 49
    ("1I", "3rd_CDFGH"),      # Match 50
    ("2A", "2B"),             # Match 51
    ("1F", "2C"),             # Match 52
    ("2K", "2L"),             # Match 53
    ("1H", "2J"),             # Match 54
    ("1D", "3rd_BEFIJ"),      # Match 55
    ("1G", "3rd_AEHIJ"),      # Match 56
]

# RIGHT HALF of bracket
RIGHT_BRACKET = [
    # Round of 32 matchups (winner advances to R16)
    ("1C", "2F"),             # Match 57
    ("2E", "2I"),             # Match 58
    ("1A", "3rd_CEFHI"),      # Match 59
    ("1L", "3rd_EHIJK"),      # Match 60
    ("1J", "2H"),             # Match 61
    ("2D", "2G"),             # Match 62
    ("1B", "3rd_EFGIJ"),      # Match 63
    ("1K", "3rd_DEIJL"),      # Match 64
]

# R16 matchups (winners of R32 pairs)
LEFT_R16 = [
    # Match 49 winner vs Match 50 winner
    # Match 51 winner vs Match 52 winner
    # Match 53 winner vs Match 54 winner
    # Match 55 winner vs Match 56 winner
]

RIGHT_R16 = [
    # Match 57 winner vs Match 58 winner
    # Match 59 winner vs Match 60 winner
    # Match 61 winner vs Match 62 winner
    # Match 63 winner vs Match 64 winner
]

# QF → SF → Final follows standard bracket progression
# Left QF winners → Left SF → Left finalist
# Right QF winners → Right SF → Right finalist
# Final: Left finalist vs Right finalist

# =============================================================================
# TOURNAMENT FORMAT
# =============================================================================

TOURNAMENT_FORMAT = {
    "total_teams": 48,
    "groups": 12,
    "teams_per_group": 4,
    "group_matches_per_team": 3,
    "advance_from_groups": {
        "top_2_per_group": 24,
        "best_3rd_place": 8,
        "total": 32,
    },
    "knockout_rounds": ["R32", "R16", "QF", "SF", "3rd Place", "Final"],
    "total_matches": 104,
    "host_countries": ["USA", "Canada", "Mexico"],
    "dates": "June 11 - July 19, 2026",
}

# =============================================================================
# FIFA RANKING FORMULA (from official FIFA PDF)
# P = Pbefore + I * (W - We)
# We = 1 / (10^(-dr/600) + 1)
# dr = rating difference (team A - team B)
# =============================================================================

MATCH_IMPORTANCE = {
    "friendly_outside_window": 5,
    "friendly_in_window": 10,
    "nations_league_group": 15,
    "nations_league_playoff_final": 25,
    "wc_qualifier": 25,
    "confederation_cup_group": 35,
    "confederation_cup_knockout": 40,
    "world_cup_group": 50,
    "world_cup_knockout": 60,
}

# PSO rules: winner gets W=0.75, loser gets W=0.5
# Knockout losses don't deduct points: if (W - We) < 0 then P = Pbefore
