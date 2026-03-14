"""
src/features.py
───────────────
Feature engineering for trader behaviour analysis.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


CLUSTER_FEATURES = [
    "avg_leverage", "win_rate", "avg_pnl_per_trade",
    "buy_sell_ratio", "risk_score", "sharpe_proxy"
]


def add_trade_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-trade derived columns."""
    df = df.copy()
    df["is_buy"]         = (df["side"] == "BUY").astype(int)
    df["is_win"]         = (df["closed_pnl"] > 0).astype(int)
    df["is_loss"]        = (df["closed_pnl"] < 0).astype(int)
    df["is_closing"]     = (df["closed_pnl"] != 0).astype(int)
    df["abs_pnl"]        = df["closed_pnl"].abs()
    df["pnl_per_size"]   = np.where(df["size"] > 0,
                                     df["closed_pnl"] / df["size"], 0)
    df["risk_per_trade"] = df["size"] * df["leverage"].clip(lower=1)
    return df


def build_daily_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to daily sentiment-level statistics."""
    daily = (df.groupby(["trade_date_naive", "sentiment_simple"])
               .agg(
                   daily_pnl        = ("closed_pnl",   "sum"),
                   trade_count      = ("closed_pnl",   "count"),
                   avg_leverage     = ("leverage",      "mean"),
                   total_volume     = ("size",          "sum"),
                   win_count        = ("is_win",        "sum"),
                   loss_count       = ("is_loss",       "sum"),
               )
               .reset_index()
    )
    daily["win_rate"] = (daily["win_count"] /
                         (daily["win_count"] + daily["loss_count"]).clip(lower=1))
    daily["date"]     = pd.to_datetime(daily["trade_date_naive"])
    return daily


def build_trader_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Build per-trader feature matrix including sentiment-conditional stats."""
    trader = (df.groupby("account")
                .agg(
                    total_pnl         = ("closed_pnl",   "sum"),
                    trade_count       = ("closed_pnl",   "count"),
                    avg_pnl_per_trade = ("closed_pnl",   "mean"),
                    pnl_std           = ("closed_pnl",   "std"),
                    avg_leverage      = ("leverage",      "mean"),
                    max_leverage      = ("leverage",      "max"),
                    total_volume      = ("size",          "sum"),
                    buy_trades        = ("is_buy",        "sum"),
                    win_trades        = ("is_win",        "sum"),
                    loss_trades       = ("is_loss",       "sum"),
                    closing_trades    = ("is_closing",    "sum"),
                    avg_risk          = ("risk_per_trade","mean"),
                )
                .reset_index()
    )

    trader["buy_sell_ratio"] = (trader["buy_trades"] /
                                 trader["trade_count"].clip(lower=1))
    trader["win_rate"]       = (trader["win_trades"] /
                                 trader["closing_trades"].clip(lower=1))
    trader["pnl_std"]        = trader["pnl_std"].fillna(0)
    trader["sharpe_proxy"]   = np.where(
        trader["pnl_std"] > 0,
        trader["avg_pnl_per_trade"] / trader["pnl_std"], 0
    )
    trader["risk_score"]     = (
        trader["avg_leverage"].rank(pct=True) * 0.40
      + trader["pnl_std"].rank(pct=True)      * 0.35
      + (1 - trader["win_rate"]).rank(pct=True) * 0.25
    )

    # Sentiment-conditional pivot
    for regime in ["Fear", "Neutral", "Greed"]:
        sub = (df[df["sentiment_simple"] == regime]
                 .groupby("account")
                 .agg(
                     **{f"pnl_{regime}":    ("closed_pnl", "sum"),
                        f"trades_{regime}": ("closed_pnl", "count"),
                        f"lev_{regime}":    ("leverage",   "mean")}
                 )
                 .reset_index()
        )
        trader = trader.merge(sub, on="account", how="left")

    regime_cols = [c for c in trader.columns if c.startswith(("pnl_","trades_","lev_"))]
    trader[regime_cols] = trader[regime_cols].fillna(0)
    return trader


def cluster_traders(trader: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    """KMeans clustering with descriptive label assignment."""
    trader = trader.copy()
    X = trader[CLUSTER_FEATURES].fillna(0)
    X_scaled = StandardScaler().fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    trader["cluster"] = km.fit_predict(X_scaled)

    summary = (trader.groupby("cluster")
                     [["avg_leverage","win_rate","avg_pnl_per_trade","risk_score"]]
                     .mean())

    labels = {}
    labels[summary["avg_leverage"].idxmax()]       = "High-Leverage Gamblers"
    labels[summary["win_rate"].idxmax()]            = "Consistent Winners"
    labels[summary["avg_pnl_per_trade"].idxmin()]   = "Loss-Taking Learners"
    for c in range(n_clusters):
        if c not in labels:
            labels[c] = "Moderate Traders"
            break

    trader["cluster_label"] = trader["cluster"].map(lambda c: labels.get(c, f"Cluster {c}"))
    return trader
