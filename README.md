# 📊 Bitcoin Market Sentiment × Trader Behavior Analysis
### Hyperliquid Historical Trades + Bitcoin Fear & Greed Index

> **Quantitative Data Science Assignment** — Web3 Trading Firm Submission  
> Analysis of **211,224 trades** across **32 unique traders** against **2,644 days** of Bitcoin Fear & Greed data.

## 📊 Figures
All generated visualizations can be found in [`outputs/figures/`](./outputs/figures/)
---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Setup & Installation](#setup--installation)
4. [Key Findings](#key-findings)
5. [Visualizations](#visualizations)
6. [Strategy Insights](#strategy-insights)
7. [Trader Archetypes](#trader-archetypes)

---

## Project Overview

This project quantifies how Bitcoin market sentiment shapes trader behavior on Hyperliquid perpetual futures. By merging real trade-level data with the Fear & Greed Index, we uncover hidden patterns in profitability, leverage usage, risk-taking, and behavioral biases across Fear, Neutral, and Greed market regimes.

---

## Repository Structure

```
btc_sentiment_analysis/
├── data/
│   ├── historical_data.csv               ← Hyperliquid raw trades (211,224 rows)
│   └── fear_greed_index.csv              ← Daily Fear & Greed Index (2,644 rows)
│
├── notebooks/
│   └── btc_sentiment_analysis.ipynb      ← Full Jupyter notebook (8 sections)
│
├── src/
│   ├── data_loader.py                    ← Load, clean & merge functions
│   ├── features.py                       ← Feature engineering + clustering
│   └── visualizations.py                ← All 7 chart functions
│
├── outputs/
│   ├── figures/                          ← 8 PNG charts (auto-generated)
│   ├── enriched_trades.csv               ← Trades + sentiment labels
│   ├── trader_profiles.csv               ← 32 trader feature profiles
│   └── daily_summary.csv                 ← Daily PnL + sentiment aggregates
│
├── requirements.txt
└── README.md
```

---

## Setup & Installation

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/btc-sentiment-trader-analysis.git
cd btc-sentiment-trader-analysis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the notebook
jupyter notebook notebooks/btc_sentiment_analysis.ipynb
```

---

## Key Findings

### Real Numbers from 211,224 Trades

| Metric | Fear | Neutral | Greed |
|--------|------|---------|-------|
| Avg PnL per closing trade | $103.82 | $56.45 | **$143.83** |
| Median PnL | $6.84 | $5.11 | $7.83 |
| **Win Rate** | **86.1%** | 80.6% | 83.8% |
| **Avg Leverage** | 1.33x | 1.15x | **1.47x** |
| PnL Std Dev (volatility) | 909 | 634 | **1,059** |
| Avg Loss per losing trade | -$150 | **-$301** | -$156 |
| Trade count | 133,871 | 7,141 | 43,251 |

### Top Trader
`0xbee1707...` — **$836K total PnL** across **40,184 trades** with 76.3% win rate

### Surprising Findings
1. **Fear has the HIGHEST win rate (86.1%)** — Panic creates cheap entries for disciplined buyers
2. **Neutral regime has the LARGEST average losses (-$301)** — Low-conviction, directionless trading is most dangerous
3. **Greed is highest average PnL but also highest variance** — It's the most volatile regime to trade
4. **96.6% of trades occur in Fear or Greed** — Almost no traders operate during Neutral calm

---

## Visualizations

| Figure | Description |
|--------|-------------|
| `fig1_pnl_distribution.png` | PnL histogram split by Fear / Neutral / Greed (98th percentile clipped) |
| `fig2_leverage_sentiment.png` | Average leverage bar chart per regime |
| `fig3_trade_count.png` | Trade volume share pie chart |
| `fig4_cumulative_pnl.png` | Cumulative PnL time-series coloured by sentiment |
| `fig5_win_rate.png` | Win rate per sentiment regime |
| `fig6_trader_clusters.png` | PCA scatter of 4 trader behavioural archetypes |
| `fig7_fg_index.png` | Bitcoin F&G Index full history (2018–2024) |
| `fig8_heatmap_monthly.png` | Average daily PnL heatmap by month × sentiment |

---

## Strategy Insights

| # | Insight | Signal | Action |
|---|---------|--------|--------|
| 1 | Greed raises leverage 27% vs Neutral | F&G > 70 | Reduce position size 25–30%; tighten stops |
| 2 | Fear has highest win rate (86.1%) | F&G < 30 | Scale into systematic longs — fear creates cheap entries |
| 3 | Neutral produces tightest PnL distribution | F&G 40–60 | Optimal window for full-size trend strategies |
| 4 | Neutral avg losses are deepest (-$301) | Low volume | Widen stops, avoid averaging down |
| 5 | Greed = highest avg PnL + highest variance | Momentum signal | Short-duration momentum trades only |
| 6 | High-Leverage Gamblers cluster has worst Sharpe | Account-level | Enforce leverage caps for flagged accounts |

---

## Trader Archetypes (KMeans k=4)

| Cluster | Avg Leverage | Win Rate | Avg PnL/Trade | Characteristic |
|---------|-------------|----------|---------------|----------------|
| **Consistent Winners** (10) | Low | Highest | High | Systematic, disciplined |
| **High-Leverage Gamblers** (8) | Highest | High | Moderate | Risk-seeking, volatile |
| **Loss-Taking Learners** (9) | Medium | Lowest | Lowest | Developing traders |
| **Moderate Traders** (5) | Low | High | Moderate | Steady, balanced |

---

## Behavioural Biases Identified

- **Greed Herding** — Traders increase size at exactly the point when markets are most stretched
- **Overconfidence** — Leverage rises 27% in Greed, yet win rate *drops* vs Fear
- **FOMO** — Trade volume increases in Greed despite worse risk-adjusted outcomes
- **Panic Avoidance** — Very few trades happen in Neutral (7,141 vs 133,871 in Fear) — traders only act at extremes

---

## Tech Stack
`Python 3.11` · `pandas` · `numpy` · `scikit-learn` · `matplotlib` · `seaborn` · `Jupyter`

---

*All data sourced from Hyperliquid (public trade history) and Alternative.me (Fear & Greed Index).*
