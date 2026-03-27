from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import duckdb
import pandas as pd


@dataclass(frozen=True)
class FeatureSpec:
    label_col: str = "churned_30d"


NUMERIC_FEATURES: tuple[str, ...] = (
    "tenure_days",
    "active_days_30",
    "events_30",
    "features_30",
    "seats_avg_30",
    "days_since_last_active",
    "tickets_60",
    "unresolved_60",
    "sentiment_mean_60",
    "urgency_mean_60",
)

CATEGORICAL_FEATURES: tuple[str, ...] = ("tier",)


def load_customer_agg_labels(data_dir: str) -> pd.DataFrame:
    path = f"{data_dir}/customer_agg_labels.parquet"
    con = duckdb.connect(database=":memory:")
    df = con.execute(f"SELECT * FROM read_parquet('{path}')").df()
    con.close()
    return df


def build_model_matrix(df: pd.DataFrame, spec: FeatureSpec | None = None) -> tuple[pd.DataFrame, pd.Series]:
    spec = spec or FeatureSpec()
    missing = [c for c in (spec.label_col, *NUMERIC_FEATURES, *CATEGORICAL_FEATURES) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X_num = df.loc[:, list(NUMERIC_FEATURES)].copy()
    X_cat = pd.get_dummies(df.loc[:, list(CATEGORICAL_FEATURES)].astype("string"), drop_first=False)
    X = pd.concat([X_num, X_cat], axis=1)
    y = df[spec.label_col].astype(int)
    return X, y


def feature_columns() -> Iterable[str]:
    # Note: tier one-hot columns are data-dependent; caller can infer from build_model_matrix output.
    return list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)

