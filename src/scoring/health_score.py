from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HealthScoreWeights:
    usage: float = 0.45
    engagement: float = 0.2
    support: float = 0.25
    recency: float = 0.1


def _minmax(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    lo = float(np.nanmin(s))
    hi = float(np.nanmax(s))
    if hi <= lo:
        return pd.Series(np.full(len(s), 0.5), index=s.index, dtype=float)
    return (s - lo) / (hi - lo)


def compute_health_score(df: pd.DataFrame, weights: HealthScoreWeights | None = None) -> pd.DataFrame:
    """
    Returns a copy of df with:
    - health_score (0..100)
    - risk_bucket (At Risk / Monitor / Healthy)
    - expansion_signal (bool-ish int)
    """
    w = weights or HealthScoreWeights()
    required = [
        "active_days_30",
        "events_30",
        "features_30",
        "seats_avg_30",
        "days_since_last_active",
        "tickets_60",
        "unresolved_60",
        "sentiment_mean_60",
        "urgency_mean_60",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for health score: {missing}")

    out = df.copy()

    usage = 0.6 * _minmax(out["active_days_30"]) + 0.4 * _minmax(out["events_30"])
    engagement = 0.55 * _minmax(out["features_30"]) + 0.45 * _minmax(out["seats_avg_30"])

    # Support: penalize volume + unresolved + urgency; reward positive sentiment
    support_pain = (
        0.35 * _minmax(out["tickets_60"])
        + 0.45 * _minmax(out["unresolved_60"])
        + 0.2 * _minmax(out["urgency_mean_60"])
    )
    support = 1.0 - np.clip(support_pain, 0.0, 1.0)
    support = 0.75 * support + 0.25 * _minmax(out["sentiment_mean_60"])  # sentiment helps

    recency = 1.0 - _minmax(out["days_since_last_active"])

    score01 = (
        w.usage * usage
        + w.engagement * engagement
        + w.support * support
        + w.recency * recency
    )
    score = np.clip(score01 * 100.0, 0.0, 100.0)

    out["health_score"] = score.round(1)
    out["risk_bucket"] = pd.cut(
        out["health_score"],
        bins=[-0.01, 45, 70, 100.01],
        labels=["At Risk", "Monitor", "Healthy"],
    ).astype("string")

    # A simple expansion heuristic: very healthy + low support pain + decent seat activation
    out["expansion_signal"] = (
        (out["health_score"] >= 80)
        & (out["unresolved_60"].astype(float) <= 1)
        & (out["seats_avg_30"].astype(float) >= out["seats_avg_30"].median())
    ).astype(int)
    return out

