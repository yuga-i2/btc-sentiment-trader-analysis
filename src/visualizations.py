"""
src/visualizations.py
──────────────────────
All plotting functions. Each function saves a PNG and returns the figure.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

PALETTE         = {"Fear": "#E74C3C", "Greed": "#2ECC71", "Neutral": "#95A5A6"}
SENTIMENT_ORDER = ["Fear", "Neutral", "Greed"]
CLUSTER_COLORS  = {
    "High-Leverage Gamblers": "#E74C3C",
    "Consistent Winners":     "#2ECC71",
    "Loss-Taking Learners":   "#3498DB",
    "Moderate Traders":       "#F39C12",
}

sns.set_theme(style="darkgrid", palette="muted", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 140,
                     "axes.spines.top": False,
                     "axes.spines.right": False})


def _save(fig, path):
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved {path}")


def plot_pnl_distribution(df, out="fig1_pnl_distribution.png"):
    closing = df[(df["is_closing"] == 1) & (df["closed_pnl"].between(-5000, 5000))]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Profit Distribution by Sentiment Regime", fontsize=14, fontweight="bold")
    for ax, regime in zip(axes, SENTIMENT_ORDER):
        sub = closing[closing["sentiment_simple"] == regime]["closed_pnl"]
        ax.hist(sub, bins=60, color=PALETTE[regime], alpha=0.85, edgecolor="none")
        ax.axvline(sub.median(), color="white", linestyle="--", linewidth=1.5,
                   label=f"Median: {sub.median():.1f}")
        ax.set_title(regime, fontweight="bold", color=PALETTE[regime])
        ax.set_xlabel("Closed PnL (USD)"); ax.set_ylabel("Count")
        ax.legend(fontsize=9)
    plt.tight_layout()
    _save(fig, out)


def plot_leverage_sentiment(df, out="fig2_leverage_sentiment.png"):
    lev = df.groupby("sentiment_simple")["leverage"].mean().reindex(SENTIMENT_ORDER)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(lev.index, lev.values,
                  color=[PALETTE[s] for s in lev.index], width=0.5, alpha=0.9)
    for b, v in zip(bars, lev.values):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.05,
                f"{v:.2f}x", ha="center", va="bottom", fontweight="bold")
    ax.set_title("Average Leverage by Sentiment Regime", fontsize=13, fontweight="bold")
    ax.set_ylabel("Mean Leverage (x)")
    plt.tight_layout()
    _save(fig, out)


def plot_trade_count_pie(df, out="fig3_trade_count_sentiment.png"):
    tc = df["sentiment_simple"].value_counts().reindex(SENTIMENT_ORDER)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(tc.values, labels=tc.index, autopct="%1.1f%%",
           colors=[PALETTE[s] for s in tc.index],
           startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 2})
    ax.set_title("Trade Volume Share by Sentiment", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, out)


def plot_pnl_trend(daily, out="fig4_pnl_trend.png"):
    ds = daily.sort_values("date").copy()
    ds["cumulative_pnl"] = ds["daily_pnl"].cumsum()
    fig, ax = plt.subplots(figsize=(14, 5))
    for regime in SENTIMENT_ORDER:
        sub = ds[ds["sentiment_simple"] == regime]
        ax.scatter(sub["date"], sub["cumulative_pnl"],
                   color=PALETTE[regime], alpha=0.55, s=12, label=regime)
    ax.plot(ds["date"], ds["cumulative_pnl"], color="white", linewidth=0.8, alpha=0.4)
    ax.set_title("Cumulative PnL Over Time (coloured by Sentiment)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date"); ax.set_ylabel("Cumulative PnL (USD)")
    ax.legend(title="Sentiment")
    plt.tight_layout()
    _save(fig, out)


def plot_win_rate(df, out="fig5_win_rate.png"):
    wr = (df[df["is_closing"] == 1]
            .groupby("sentiment_simple")["is_win"]
            .mean().mul(100).reindex(SENTIMENT_ORDER))
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(wr.index, wr.values,
                  color=[PALETTE[s] for s in wr.index], width=0.5, alpha=0.9)
    ax.axhline(50, color="grey", linestyle="--", linewidth=1, label="50% baseline")
    for b, v in zip(bars, wr.values):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.3,
                f"{v:.1f}%", ha="center", va="bottom", fontweight="bold")
    ax.set_title("Win Rate by Sentiment Regime", fontsize=13, fontweight="bold")
    ax.set_ylabel("Win Rate (%)"); ax.set_ylim(0, 80)
    ax.legend()
    plt.tight_layout()
    _save(fig, out)


def plot_trader_clusters(trader, out="fig6_trader_clusters.png"):
    features = ["avg_leverage","win_rate","avg_pnl_per_trade",
                "buy_sell_ratio","risk_score","sharpe_proxy"]
    X = StandardScaler().fit_transform(trader[features].fillna(0))
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    trader = trader.copy()
    trader["pc1"] = coords[:, 0]; trader["pc2"] = coords[:, 1]
    fig, ax = plt.subplots(figsize=(10, 7))
    for label, grp in trader.groupby("cluster_label"):
        ax.scatter(grp["pc1"], grp["pc2"], label=label,
                   color=CLUSTER_COLORS.get(label, "grey"),
                   alpha=0.6, s=18, edgecolors="none")
    ax.set_title("Trader Behavioural Clusters (PCA)", fontsize=13, fontweight="bold")
    ax.set_xlabel(f"PC-1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"PC-2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax.legend(title="Cluster", fontsize=9)
    plt.tight_layout()
    _save(fig, out)


def plot_fg_index(sentiment, out="fig7_fg_index.png"):
    if "fg_score" not in sentiment.columns:
        return
    fig, ax = plt.subplots(figsize=(14, 4))
    sc = ax.scatter(sentiment["date"], sentiment["fg_score"],
                    c=sentiment["fg_score"], cmap="RdYlGn",
                    s=8, alpha=0.8, vmin=0, vmax=100)
    plt.colorbar(sc, ax=ax, label="F&G Score")
    ax.axhline(50, color="grey", linestyle="--", linewidth=1)
    ax.set_title("Bitcoin Fear & Greed Index Over Time", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date"); ax.set_ylabel("Score")
    plt.tight_layout()
    _save(fig, out)
