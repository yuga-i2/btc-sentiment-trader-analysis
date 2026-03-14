"""
src/data_loader.py
──────────────────
Modular data loading and cleaning utilities.
"""

import pandas as pd
import numpy as np


SENTIMENT_SIMPLIFY = {
    "extreme fear": "Fear",
    "fear":         "Fear",
    "neutral":      "Neutral",
    "greed":        "Greed",
    "extreme greed":"Greed",
}


def load_trader_data(path: str) -> pd.DataFrame:
    """Load and do initial cleaning of Hyperliquid trade CSV."""
    df = pd.read_csv(path)
    df.columns = (df.columns.str.strip().str.lower()
                             .str.replace(r'\s+', '_', regex=True))

    rename = {
        "execution_price": "exec_price",
        "closedpnl":       "closed_pnl",
        "startposition":   "start_position",
    }
    df.rename(columns={k: v for k, v in rename.items() if k in df.columns}, inplace=True)

    # Parse timestamps
    if df["time"].dtype in [np.int64, np.float64]:
        df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    else:
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")

    df["trade_date"] = df["time"].dt.normalize()

    # Coerce numeric cols
    for col in ["closed_pnl", "size", "exec_price", "leverage"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df.dropna(subset=["account", "time", "side"], inplace=True)
    df["side"] = df["side"].str.upper().str.strip()
    return df


def load_sentiment_data(path: str) -> pd.DataFrame:
    """Load and clean Fear & Greed Index CSV."""
    df = pd.read_csv(path)
    df.columns = (df.columns.str.strip().str.lower()
                             .str.replace(r'\s+', '_', regex=True))

    rename = {"classification": "sentiment", "value": "fg_score"}
    df.rename(columns={k: v for k, v in rename.items() if k in df.columns}, inplace=True)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.dropna(subset=["date"], inplace=True)
    df.sort_values("date", inplace=True)

    df["sentiment_simple"] = (
        df["sentiment"].str.strip().str.lower()
        .map(SENTIMENT_SIMPLIFY)
        .fillna("Neutral")
    )
    return df


def merge_datasets(trades: pd.DataFrame, sentiment: pd.DataFrame) -> pd.DataFrame:
    """Merge on trade date (UTC date) ↔ sentiment date."""
    sent_sub = sentiment[["date", "sentiment_simple", "fg_score"]].copy()
    sent_sub["date"] = sent_sub["date"].dt.normalize()

    trades["trade_date_naive"] = trades["trade_date"].dt.tz_localize(None)

    merged = trades.merge(
        sent_sub,
        left_on  = "trade_date_naive",
        right_on = "date",
        how      = "left"
    )
    merged["sentiment_simple"].fillna("Neutral", inplace=True)
    return merged
