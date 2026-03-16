def elo_expected_score(elo_a, elo_b):
    """FIFA Elo expected score: We = 1 / (10^(-dr/600) + 1)"""
    dr = elo_a - elo_b
    return 1 / (10 ** (-dr / 600) + 1)


def predict_match(elo_a, elo_b, knockout=False):
    """
    Predict match outcome probabilities.
    Returns: (p_win_a, p_draw, p_win_b)
    For knockout: (p_advance_a, 0, p_advance_b)
    """
    exp_a = elo_expected_score(elo_a, elo_b)

    if knockout:
        compression = 0.85
        p_advance_a = 0.5 + (exp_a - 0.5) * compression
        p_advance_b = 1 - p_advance_a
        return round(p_advance_a, 4), 0, round(p_advance_b, 4)
    else:
        base_draw = 0.24
        closeness = 1 - abs(exp_a - 0.5) * 2
        p_draw = base_draw * (0.5 + 0.5 * closeness)
        remaining = 1 - p_draw
        p_win_a = remaining * exp_a
        p_win_b = remaining * (1 - exp_a)
        return round(p_win_a, 4), round(p_draw, 4), round(p_win_b, 4)


def predict_goals(elo_a, elo_b):
    """Estimate expected goals based on Elo difference."""
    exp_a = elo_expected_score(elo_a, elo_b)
    total_goals = 2.6
    return round(total_goals * exp_a, 2), round(total_goals * (1 - exp_a), 2)