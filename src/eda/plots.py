# =============================================================================
# src/eda/plots.py — OP'26 EDA Visualisation Module
#
# All plots are saved to outputs/eda/ by default.
# Post-run plots (predicted vs actual, reward convergence, theta evolution)
# require orchestrator outputs to exist first.
#
# Run:
#   python -m src.eda.plots
#   python -m src.eda.plots --base data/processed/unified_analytical_base.csv \
#                           --acn  data/raw/acndata_sessions.json.xlsx
# =============================================================================

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.gridspec import GridSpec

from src.config import (
    P_BASE, P_SURGE_CAP, P_DISCOUNT_FLOOR,
    SURGE_THRESHOLD, DISCOUNT_THRESHOLD,
    PROCESSED_BASE_PATH, RAW_ACN_PATH, EDA_OUTPUTS_DIR,
    OUTPUTS_DIR, engineer_features, FEATURE_COLS,
    RANDOM_STATE,
)

warnings.filterwarnings("ignore")
logger = logging.getLogger("ev_agentic.eda")

# ── Dark industrial theme ────────────────────────────────────────────────────
BG      = "#0d0f14"
PANEL   = "#151821"
ACCENT  = "#00d4ff"
AMBER   = "#ffb830"
GREEN   = "#00ff9f"
RED     = "#ff4757"
PURPLE  = "#9b59b6"
GRID    = "#1e2535"
TEXT    = "#d0d6e8"
SUBTEXT = "#6b7a99"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT,
    "xtick.color": SUBTEXT, "ytick.color": SUBTEXT,
    "text.color": TEXT, "grid.color": GRID, "grid.linewidth": 0.6,
    "font.family": "monospace", "axes.titlecolor": TEXT,
    "axes.titlesize": 11, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.facecolor": PANEL, "legend.edgecolor": GRID,
    "legend.labelcolor": TEXT, "figure.dpi": 120,
})


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _save(name: str, output_dir: str) -> None:
    path = Path(output_dir) / name
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor=BG)
    plt.close()
    logger.info("Saved: %s", path)


def _load_unified(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["hourly_timestamp"])
    df.sort_values("hourly_timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    logger.info("Unified base: %d rows", len(df))
    return df


def _load_acn(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.dropna(subset=["connectionTime", "kWhDelivered"], inplace=True)
    df = df[df["kWhDelivered"] > 0].copy()
    df["connect_dt"] = pd.to_datetime(df["connectionTime"], errors="coerce")
    df.dropna(subset=["connect_dt"], inplace=True)
    df["disconnect_dt"] = pd.to_datetime(df.get("disconnectTime", pd.NaT), errors="coerce")
    df["session_hours"] = (
        (df["disconnect_dt"] - df["connect_dt"]).dt.total_seconds() / 3600.0
    ).clip(lower=0)
    df["hour_of_day"] = df["connect_dt"].dt.hour
    df["day_of_week"] = df["connect_dt"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["baseline_revenue"] = df["kWhDelivered"] * P_BASE
    logger.info("ACN sessions: %d", len(df))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1 — Long-run Demand Trend + 7-day Rolling Mean
# ─────────────────────────────────────────────────────────────────────────────

def plot_demand_trend(df: pd.DataFrame, output_dir: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.suptitle("OP'26 — Long-Run Demand & Queue Trends", fontsize=13,
                 color=TEXT, fontweight="bold", y=0.98)
    roll = 24 * 7
    ts = df["hourly_timestamp"]
    for ax, col, color, label in [
        (axes[0], "urban_mean_utilization", ACCENT, "Mean Charger Utilisation"),
        (axes[1], "urban_peak_queue", AMBER, "Peak Queue Length Proxy"),
    ]:
        ax.fill_between(ts, df[col], alpha=0.15, color=color)
        ax.plot(ts, df[col], lw=0.4, alpha=0.5, color=color)
        ax.plot(ts, df[col].rolling(roll, min_periods=1).mean(),
                lw=2.0, color=color, label="7-day rolling mean")
        ax.set_ylabel(label)
        ax.grid(True, axis="y")
        ax.legend(fontsize=8, loc="upper right")
        if col == "urban_mean_utilization":
            ax.axhline(SURGE_THRESHOLD, color=RED, lw=1.0, ls="--", alpha=0.7,
                       label="Surge threshold (80%)")
            ax.axhline(DISCOUNT_THRESHOLD, color=GREEN, lw=1.0, ls="--", alpha=0.7,
                       label="Discount threshold (30%)")
            ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    _save("01_demand_trend.png", output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2 — Intraday Demand Cycle
# ─────────────────────────────────────────────────────────────────────────────

def plot_intraday_cycle(df: pd.DataFrame, output_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("OP'26 — Intraday Demand Cycles", fontsize=13,
                 color=TEXT, fontweight="bold")
    hourly = df.groupby("hour_of_day")["urban_mean_utilization"].agg(
        ["mean", "std", "median"]
    ).reset_index()
    ax = axes[0]
    ax.fill_between(hourly["hour_of_day"],
                    hourly["mean"] - hourly["std"],
                    hourly["mean"] + hourly["std"],
                    alpha=0.2, color=ACCENT, label="±1 std dev")
    ax.plot(hourly["hour_of_day"], hourly["mean"], color=ACCENT, lw=2.5, label="Mean")
    ax.plot(hourly["hour_of_day"], hourly["median"], color=AMBER, lw=1.5, ls="--",
            label="Median")
    ax.axhline(SURGE_THRESHOLD, color=RED, lw=1.2, ls="--", alpha=0.8,
               label="Surge threshold")
    ax.axhline(DISCOUNT_THRESHOLD, color=GREEN, lw=1.2, ls="--", alpha=0.8,
               label="Discount threshold")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Mean Charger Utilisation")
    ax.set_title("Utilisation by Hour")
    ax.set_xticks(range(0, 24, 2))
    ax.legend(fontsize=7)
    ax.grid(True)
    vol_hourly = df.groupby("hour_of_day")["urban_total_volume"].mean()
    ax2 = axes[1]
    bars = ax2.bar(vol_hourly.index, vol_hourly.values, color=ACCENT, alpha=0.7, width=0.7)
    for bar, h in zip(bars, vol_hourly.index):
        u = hourly.loc[hourly["hour_of_day"] == h, "mean"].values
        if len(u):
            bar.set_color(RED if u[0] >= SURGE_THRESHOLD
                          else (GREEN if u[0] <= DISCOUNT_THRESHOLD else ACCENT))
    ax2.set_xlabel("Hour of Day")
    ax2.set_ylabel("Mean Total Charging Volume")
    ax2.set_title("Charging Volume by Hour")
    ax2.set_xticks(range(0, 24, 2))
    ax2.grid(True, axis="y")
    fig.tight_layout()
    _save("02_intraday_cycle.png", output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3 — Weekday vs Weekend + DoW×Hour Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_weekday_weekend(df: pd.DataFrame, output_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("OP'26 — Weekday vs Weekend Demand Profiles", fontsize=13,
                 color=TEXT, fontweight="bold")
    wd = df[df["is_weekend"] == 0].groupby("hour_of_day")["urban_mean_utilization"].agg(
        ["mean", "std"])
    we = df[df["is_weekend"] == 1].groupby("hour_of_day")["urban_mean_utilization"].agg(
        ["mean", "std"])
    ax = axes[0]
    for data, color, label in [(wd, ACCENT, "Weekday"), (we, AMBER, "Weekend")]:
        ax.fill_between(data.index, data["mean"] - data["std"],
                        data["mean"] + data["std"], alpha=0.15, color=color)
        ax.plot(data.index, data["mean"], lw=2.5, color=color, label=label)
    ax.axhline(SURGE_THRESHOLD, color=RED, lw=1.0, ls="--", alpha=0.8)
    ax.axhline(DISCOUNT_THRESHOLD, color=GREEN, lw=1.0, ls="--", alpha=0.8)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Mean Utilisation")
    ax.set_title("Utilisation Profile")
    ax.set_xticks(range(0, 24, 2))
    ax.legend()
    ax.grid(True)
    ax2 = axes[1]
    dow_hour = df.groupby(["day_of_week", "hour_of_day"])[
        "urban_mean_utilization"].mean().unstack()
    im = ax2.imshow(dow_hour.values, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1,
                    extent=[-0.5, 23.5, -0.5, 6.5])
    ax2.set_yticks(range(7))
    ax2.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][::-1])
    ax2.set_xticks(range(0, 24, 2))
    ax2.set_xlabel("Hour of Day")
    ax2.set_title("Utilisation Heatmap (DoW × Hour)")
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("Mean Utilisation", color=TEXT)
    fig.tight_layout()
    _save("03_weekday_weekend.png", output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 4 — ACN Session Distributions
# ─────────────────────────────────────────────────────────────────────────────

def plot_acn_distributions(acn: pd.DataFrame, output_dir: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("OP'26 — ACN Session Distributions (Caltech/JPL)", fontsize=13,
                 color=TEXT, fontweight="bold")
    ax = axes[0, 0]
    kwh = acn["kWhDelivered"].clip(0, 50)
    ax.hist(kwh, bins=60, color=ACCENT, alpha=0.8, edgecolor="none")
    ax.axvline(kwh.mean(), color=AMBER, lw=2, label=f"Mean: {kwh.mean():.1f} kWh")
    ax.axvline(kwh.median(), color=GREEN, lw=2, ls="--",
               label=f"Median: {kwh.median():.1f} kWh")
    ax.set_xlabel("kWh Delivered")
    ax.set_ylabel("Sessions")
    ax.set_title("Energy per Session")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y")

    ax = axes[0, 1]
    dur = acn["session_hours"].clip(0, 12)
    ax.hist(dur, bins=60, color=AMBER, alpha=0.8, edgecolor="none")
    ax.axvline(dur.mean(), color=ACCENT, lw=2, label=f"Mean: {dur.mean():.1f} h")
    ax.axvline(dur.median(), color=GREEN, lw=2, ls="--",
               label=f"Median: {dur.median():.1f} h")
    ax.set_xlabel("Session Duration (hours)")
    ax.set_ylabel("Sessions")
    ax.set_title("Session Duration")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y")

    ax = axes[1, 0]
    hourly_s = acn.groupby("hour_of_day").size()
    ax.bar(hourly_s.index, hourly_s.values, color=GREEN, alpha=0.8, width=0.8)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Session Count")
    ax.set_title("Sessions by Hour of Day")
    ax.set_xticks(range(0, 24, 2))
    ax.grid(True, axis="y")

    ax = axes[1, 1]
    top_st = acn["stationID"].value_counts().head(15)
    ax.barh(range(len(top_st)), top_st.values, color=PURPLE, alpha=0.8)
    ax.set_yticks(range(len(top_st)))
    ax.set_yticklabels(top_st.index, fontsize=7)
    ax.set_xlabel("Session Count")
    ax.set_title("Top 15 Stations by Usage")
    ax.grid(True, axis="x")

    fig.tight_layout()
    _save("04_acn_distributions.png", output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 5 — Peak / Shoulder / Off-Peak Volatility
# ─────────────────────────────────────────────────────────────────────────────

def plot_peak_volatility(df: pd.DataFrame, output_dir: str) -> None:
    def _regime(u: float) -> str:
        if u >= SURGE_THRESHOLD:
            return "Peak (Surge)"
        if u <= DISCOUNT_THRESHOLD:
            return "Off-Peak (Discount)"
        return "Shoulder (Neutral)"

    df2 = df.copy()
    df2["regime"] = df2["urban_mean_utilization"].apply(_regime)
    order = ["Peak (Surge)", "Shoulder (Neutral)", "Off-Peak (Discount)"]
    colors = [RED, ACCENT, GREEN]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("OP'26 — Volatility by Tariff Regime", fontsize=13,
                 color=TEXT, fontweight="bold")
    for ax, col, ylabel in [
        (axes[0], "urban_mean_utilization", "Utilisation"),
        (axes[1], "urban_peak_queue", "Queue Length"),
        (axes[2], "urban_total_volume", "Charging Volume"),
    ]:
        data = [df2[df2["regime"] == r][col].values for r in order]
        bp = ax.boxplot(data, positions=[1, 2, 3], patch_artist=True,
                        medianprops=dict(color="white", lw=2),
                        whiskerprops=dict(color=SUBTEXT),
                        capprops=dict(color=SUBTEXT),
                        flierprops=dict(marker=".", color=SUBTEXT, alpha=0.3, ms=2))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(["Peak", "Shoulder", "Off-Peak"], fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} Distribution")
        ax.grid(True, axis="y")
        for pos, d in zip([1, 2, 3], data):
            ax.text(pos, ax.get_ylim()[1] * 0.95, f"σ={np.std(d):.3f}",
                    ha="center", va="top", fontsize=7, color=TEXT)
    fig.tight_layout()
    _save("05_peak_volatility.png", output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 6 — Feature Correlation Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame, output_dir: str) -> None:
    d = engineer_features(df.copy())
    corr_cols = [
        "urban_mean_utilization", "urban_peak_queue", "urban_total_volume",
        "acn_sessions_count", "acn_total_kwh",
        "hour_sin", "hour_cos", "is_weekend",
        "util_lag1", "util_lag24", "util_roll6_mean", "util_roll6_std",
        "queue_lag1", "queue_roll6_mean",
    ]
    corr_cols = [c for c in corr_cols if c in d.columns]
    corr = d[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(13, 10))
    fig.suptitle("OP'26 — Feature Correlation Matrix", fontsize=13,
                 color=TEXT, fontweight="bold")
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, ax=ax, mask=mask,
                cmap=sns.diverging_palette(220, 20, as_cmap=True),
                vmin=-1, vmax=1, center=0,
                annot=True, fmt=".2f", annot_kws={"size": 7},
                linewidths=0.3, linecolor=BG,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Lower triangle — Pearson correlation", fontsize=9, color=SUBTEXT)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)
    fig.tight_layout()
    _save("06_correlation_heatmap.png", output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 7 — Revenue Analysis
# ─────────────────────────────────────────────────────────────────────────────

def plot_revenue_analysis(df: pd.DataFrame, acn: pd.DataFrame, output_dir: str) -> None:
    def _sim_tariff(u: float) -> float:
        if u >= SURGE_THRESHOLD:
            ss = min(1.0, (u - SURGE_THRESHOLD) / (1.0 - SURGE_THRESHOLD))
            return float(np.clip(P_BASE + ss * (P_SURGE_CAP - P_BASE), P_DISCOUNT_FLOOR, P_SURGE_CAP))
        if u <= DISCOUNT_THRESHOLD:
            ds = min(1.0, (DISCOUNT_THRESHOLD - u) / DISCOUNT_THRESHOLD)
            return float(np.clip(P_BASE - ds * (P_BASE - P_DISCOUNT_FLOOR), P_DISCOUNT_FLOOR, P_SURGE_CAP))
        return P_BASE

    df2 = df.copy()
    df2["sim_tariff"] = df2["urban_mean_utilization"].apply(_sim_tariff)
    df2["revenue_dynamic"] = df2["acn_total_kwh"] * df2["sim_tariff"]
    df2["revenue_baseline"] = df2["acn_total_kwh"] * P_BASE
    df2["revenue_gain"] = df2["revenue_dynamic"] - df2["revenue_baseline"]

    total_base = df2["revenue_baseline"].sum()
    total_dyn = df2["revenue_dynamic"].sum()
    # Revenue Gain % = ((Dynamic − Baseline) / Baseline) × 100
    gain_pct = (total_dyn - total_base) / max(total_base, 1e-6) * 100

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("OP'26 — Revenue & Tariff Implication Analysis", fontsize=13,
                 color=TEXT, fontweight="bold")

    ax = axes[0, 0]
    ax.hist(df2["sim_tariff"], bins=40, color=AMBER, alpha=0.8, edgecolor="none")
    ax.axvline(P_BASE, color=TEXT, lw=1.5, ls="--", label=f"Baseline ₹{P_BASE}")
    ax.axvline(df2["sim_tariff"].mean(), color=ACCENT, lw=1.5,
               label=f"Mean ₹{df2['sim_tariff'].mean():.2f}")
    ax.set_xlabel("Simulated Tariff (₹/kWh)")
    ax.set_ylabel("Hourly Count")
    ax.set_title("Dynamic Tariff Distribution")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y")

    ax = axes[0, 1]
    ax.bar(["Baseline\n₹15/kWh", "Dynamic\nTariff"], [total_base, total_dyn],
           color=[SUBTEXT, ACCENT], alpha=0.8, width=0.5)
    ax.set_ylabel("Total Revenue (₹)")
    ax.set_title(f"Total Revenue Comparison\nRevenue Gain: {gain_pct:+.1f}%")
    ax.text(1, total_dyn * 1.02, f"{gain_pct:+.1f}%", ha="center",
            color=GREEN, fontweight="bold", fontsize=12)
    ax.grid(True, axis="y")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))

    ax = axes[1, 0]
    regime_counts = df2["sim_tariff"].apply(
        lambda t: "Surge" if t > P_BASE else ("Discount" if t < P_BASE else "Neutral")
    ).value_counts()
    colors_pie = [RED if "Surge" in r else (GREEN if "Discount" in r else ACCENT)
                  for r in regime_counts.index]
    _, _, autotexts = ax.pie(regime_counts.values, labels=regime_counts.index,
                              colors=colors_pie, autopct="%1.1f%%", startangle=90,
                              wedgeprops=dict(edgecolor=BG, linewidth=1.5))
    for at in autotexts:
        at.set_fontsize(9)
        at.set_color("white")
    ax.set_title("Hourly Regime Distribution")

    ax = axes[1, 1]
    rev_by_hour = df2.groupby("hour_of_day")["revenue_gain"].mean()
    bar_colors = [RED if v > 0 else GREEN for v in rev_by_hour.values]
    ax.bar(rev_by_hour.index, rev_by_hour.values, color=bar_colors, alpha=0.8, width=0.8)
    ax.axhline(0, color=TEXT, lw=1.0)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Mean Revenue Gain per Hour (₹)")
    ax.set_title("Revenue Gain/Loss by Hour")
    ax.set_xticks(range(0, 24, 2))
    ax.grid(True, axis="y")

    fig.tight_layout()
    _save("07_revenue_analysis.png", output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 8 — ACN Session Efficiency Scatter
# ─────────────────────────────────────────────────────────────────────────────

def plot_acn_session_efficiency(acn: pd.DataFrame, output_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("OP'26 — ACN Charging Session Efficiency", fontsize=13,
                 color=TEXT, fontweight="bold")
    sample = acn.sample(min(3000, len(acn)), random_state=RANDOM_STATE)
    sample = sample.copy()
    sample["kwh_per_hour"] = sample["kWhDelivered"] / sample["session_hours"].clip(lower=0.1)

    ax = axes[0]
    sc = ax.scatter(sample["session_hours"].clip(0, 12),
                    sample["kWhDelivered"].clip(0, 50),
                    c=sample["hour_of_day"], cmap="plasma",
                    alpha=0.4, s=10, edgecolors="none")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Hour of Day", color=TEXT)
    ax.set_xlabel("Session Duration (hours)")
    ax.set_ylabel("kWh Delivered")
    ax.set_title("Duration vs Energy — coloured by hour")
    ax.grid(True)

    ax = axes[1]
    rate = sample["kwh_per_hour"].clip(0, 20)
    ax.hist(rate, bins=50, color=GREEN, alpha=0.8, edgecolor="none")
    ax.axvline(rate.mean(), color=AMBER, lw=2,
               label=f"Mean: {rate.mean():.1f} kWh/h")
    ax.set_xlabel("Charging Rate (kWh/hour)")
    ax.set_ylabel("Sessions")
    ax.set_title("Effective Charging Rate Distribution")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y")

    fig.tight_layout()
    _save("08_session_efficiency.png", output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 9 — Tariff Narrative with Annotated Pricing Zones
# ─────────────────────────────────────────────────────────────────────────────

def plot_tariff_narrative(df: pd.DataFrame, output_dir: str) -> None:
    fig = plt.figure(figsize=(14, 6))
    gs = GridSpec(1, 3, figure=fig, wspace=0.05)
    fig.suptitle("OP'26 — Pricing Implications from EDA", fontsize=13,
                 color=TEXT, fontweight="bold")
    hourly = df.groupby("hour_of_day")["urban_mean_utilization"].mean()

    ax = fig.add_subplot(gs[0, :2])
    ax.fill_between(hourly.index, hourly.values, alpha=0.15, color=ACCENT)
    ax.plot(hourly.index, hourly.values, color=ACCENT, lw=2.5)
    ax.axhline(SURGE_THRESHOLD, color=RED, lw=1.5, ls="--", alpha=0.9)
    ax.axhline(DISCOUNT_THRESHOLD, color=GREEN, lw=1.5, ls="--", alpha=0.9)
    ax.fill_between(hourly.index, SURGE_THRESHOLD, 1.0, alpha=0.07, color=RED)
    ax.fill_between(hourly.index, 0.0, DISCOUNT_THRESHOLD, alpha=0.07, color=GREEN)
    ax.annotate("▲ SURGE ZONE\n₹15→₹22/kWh", xy=(8, 0.85), fontsize=8, color=RED,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=PANEL, edgecolor=RED, alpha=0.8))
    ax.annotate("▼ DISCOUNT ZONE\n₹10→₹15/kWh", xy=(8, 0.15), fontsize=8, color=GREEN,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=PANEL, edgecolor=GREEN, alpha=0.8))
    ax.set_xlim(0, 23)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Mean Utilisation")
    ax.set_title("Hourly Utilisation → Dynamic Tariff Zones")
    ax.set_xticks(range(0, 24, 2))
    ax.grid(True)

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis("off")
    surge_pct = (df["urban_mean_utilization"] >= SURGE_THRESHOLD).mean() * 100
    disc_pct = (df["urban_mean_utilization"] <= DISCOUNT_THRESHOLD).mean() * 100
    stats = [
        (f"Surge hours (≥{SURGE_THRESHOLD*100:.0f}%)", f"{surge_pct:.1f}%", RED),
        ("Neutral hours", f"{100-surge_pct-disc_pct:.1f}%", ACCENT),
        (f"Discount hours (≤{DISCOUNT_THRESHOLD*100:.0f}%)", f"{disc_pct:.1f}%", GREEN),
        ("", "", TEXT),
        ("Mean utilisation", f"{df['urban_mean_utilization'].mean():.3f}", TEXT),
        ("Std dev utilisation", f"{df['urban_mean_utilization'].std():.3f}", SUBTEXT),
        ("Peak hour (avg)", f"{hourly.idxmax():02d}:00", AMBER),
        ("Trough hour (avg)", f"{hourly.idxmin():02d}:00", GREEN),
        ("", "", TEXT),
        ("ACN kWh mean", f"{df['acn_total_kwh'].mean():.2f}", TEXT),
        ("ACN baseline revenue", f"₹{(df['acn_total_kwh'] * P_BASE).sum():,.0f}", AMBER),
    ]
    y = 0.97
    for label, value, color in stats:
        if not label:
            y -= 0.035
            continue
        ax2.text(0.05, y, label, transform=ax2.transAxes, fontsize=8, color=SUBTEXT, va="top")
        ax2.text(0.95, y, value, transform=ax2.transAxes, fontsize=8, color=color,
                 va="top", ha="right", fontweight="bold")
        y -= 0.08
    fig.tight_layout()
    _save("09_tariff_narrative.png", output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 10 — XGBoost Feature Importance (graceful fallback if unavailable)
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(df: pd.DataFrame, output_dir: str) -> None:
    try:
        from xgboost import XGBRegressor
        from sklearn.multioutput import MultiOutputRegressor
        from src.config import XGB_PARAMS, RANDOM_STATE
    except ImportError:
        logger.warning("xgboost not available — skipping feature importance plot")
        return

    try:
        d = engineer_features(df.copy())
        feat_cols = [c for c in FEATURE_COLS if c in d.columns]
        split = int(len(d) * 0.80)
        X_tr = d.iloc[:split][feat_cols].values
        y_tr = d.iloc[:split][["urban_mean_utilization", "urban_peak_queue"]].values

        params = {k: v for k, v in XGB_PARAMS.items()}
        params["n_estimators"] = 200  # lighter for EDA
        base = XGBRegressor(**params, random_state=RANDOM_STATE)
        model = MultiOutputRegressor(base, n_jobs=-1)
        model.fit(X_tr, y_tr)

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle("OP'26 — XGBoost Feature Importance (Demand Prediction Agent)",
                     fontsize=13, color=TEXT, fontweight="bold")
        for ax, estimator, title in zip(
            axes, model.estimators_,
            ["urban_mean_utilization", "urban_peak_queue"]
        ):
            fi = pd.Series(estimator.feature_importances_,
                           index=feat_cols).sort_values(ascending=True)
            bar_colors = [
                AMBER if v >= fi.max() * 0.8 else (ACCENT if v >= fi.max() * 0.5 else SUBTEXT)
                for v in fi.values
            ]
            ax.barh(range(len(fi)), fi.values, color=bar_colors, alpha=0.85)
            ax.set_yticks(range(len(fi)))
            ax.set_yticklabels(fi.index, fontsize=7)
            ax.set_xlabel("Feature Importance (gain)")
            ax.set_title(f"Target: {title}")
            ax.grid(True, axis="x")
        fig.tight_layout()
        _save("10_feature_importance.png", output_dir)
    except Exception as exc:
        logger.warning("Feature importance plot failed: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# POST-RUN PLOTS (require orchestrator outputs)
# ─────────────────────────────────────────────────────────────────────────────

def plot_predicted_vs_actual(predictions_path: str, output_dir: str) -> None:
    """Reads outputs/predictions.csv; marks train/test split boundary."""
    if not Path(predictions_path).exists():
        logger.warning("predictions.csv not found at %s — skipping plot", predictions_path)
        return
    try:
        pred_df = pd.read_csv(predictions_path)
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        fig.suptitle("OP'26 — Predicted vs Actual Demand", fontsize=13,
                     color=TEXT, fontweight="bold")
        targets = [
            ("actual_urban_mean_utilization", "pred_urban_mean_utilization", "Utilisation"),
            ("actual_urban_peak_queue", "pred_urban_peak_queue", "Peak Queue"),
        ]
        for ax, (actual_col, pred_col, title) in zip(axes, targets):
            if actual_col not in pred_df.columns or pred_col not in pred_df.columns:
                continue
            idx = range(len(pred_df))
            ax.plot(idx, pred_df[actual_col].values, color=ACCENT, lw=1.5,
                    alpha=0.8, label="Actual")
            ax.plot(idx, pred_df[pred_col].values, color=AMBER, lw=1.5,
                    alpha=0.8, label="Predicted", ls="--")
            split_idx = int(len(pred_df) * 0.80)
            ax.axvline(split_idx, color=RED, lw=1.5, ls=":", alpha=0.9,
                       label="Train/Test Split")
            ax.set_title(title)
            ax.legend(fontsize=8)
            ax.grid(True)
        fig.tight_layout()
        _save("11_predicted_vs_actual.png", output_dir)
    except Exception as exc:
        logger.warning("plot_predicted_vs_actual failed: %s", exc)


def plot_reward_convergence(outcomes_path: str, output_dir: str) -> None:
    """Reads outputs/agentic_outcomes.csv; plots per-step reward + 50-step rolling mean."""
    if not Path(outcomes_path).exists():
        logger.warning("agentic_outcomes.csv not found at %s — skipping plot", outcomes_path)
        return
    try:
        df = pd.read_csv(outcomes_path)
        if "reward" not in df.columns:
            logger.warning("'reward' column missing in outcomes CSV — skipping plot")
            return
        fig, ax = plt.subplots(figsize=(14, 5))
        fig.suptitle("OP'26 — Reward Convergence", fontsize=13,
                     color=TEXT, fontweight="bold")
        ax.plot(df.index, df["reward"].values, color=ACCENT, lw=1.0,
                alpha=0.5, label="Per-step reward")
        rolling = df["reward"].rolling(50, min_periods=1).mean()
        ax.plot(df.index, rolling.values, color=AMBER, lw=2.5,
                label="50-step rolling mean")
        ax.set_xlabel("Episode Step")
        ax.set_ylabel("Reward")
        ax.legend(fontsize=8)
        ax.grid(True)
        fig.tight_layout()
        _save("12_reward_convergence.png", output_dir)
    except Exception as exc:
        logger.warning("plot_reward_convergence failed: %s", exc)


def plot_theta_evolution(outcomes_path: str, output_dir: str) -> None:
    """Reads outputs/agentic_outcomes.csv; plots ε, α, β on a single figure."""
    if not Path(outcomes_path).exists():
        logger.warning("agentic_outcomes.csv not found at %s — skipping plot", outcomes_path)
        return
    try:
        df = pd.read_csv(outcomes_path)
        theta_cols = {
            "epsilon_after": ("ε (price elasticity)", ACCENT),
            "alpha_after": ("α (surge sensitivity)", AMBER),
            "beta_after": ("β (discount sensitivity)", GREEN),
        }
        available = {k: v for k, v in theta_cols.items() if k in df.columns}
        if not available:
            logger.warning("No theta columns found in outcomes CSV — skipping plot")
            return
        fig, ax = plt.subplots(figsize=(14, 5))
        fig.suptitle("OP'26 — θ Parameter Evolution", fontsize=13,
                     color=TEXT, fontweight="bold")
        for col, (label, color) in available.items():
            ax.plot(df.index, df[col].values, color=color, lw=2.0, label=label)
        ax.set_xlabel("Episode Step")
        ax.set_ylabel("Parameter Value")
        ax.legend(fontsize=8)
        ax.grid(True)
        fig.tight_layout()
        _save("13_theta_evolution.png", output_dir)
    except Exception as exc:
        logger.warning("plot_theta_evolution failed: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_eda(
    base_path: str = PROCESSED_BASE_PATH,
    acn_path: str = RAW_ACN_PATH,
    output_dir: str = EDA_OUTPUTS_DIR,
) -> None:
    """
    Entry point: loads data, calls all plot functions, saves to output_dir.
    Post-run plots are skipped if orchestrator outputs are not yet available.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Loading unified base from %s", base_path)
    df = _load_unified(base_path)

    logger.info("Loading ACN sessions from %s", acn_path)
    acn = _load_acn(acn_path)

    # ── Core demand plots ────────────────────────────────────────────────────
    logger.info("Generating demand trend plot…")
    plot_demand_trend(df, output_dir)

    logger.info("Generating intraday cycle plot…")
    plot_intraday_cycle(df, output_dir)

    logger.info("Generating weekday/weekend plot…")
    plot_weekday_weekend(df, output_dir)

    # ── ACN distribution and volatility plots ────────────────────────────────
    logger.info("Generating ACN distributions plot…")
    plot_acn_distributions(acn, output_dir)

    logger.info("Generating peak volatility plot…")
    plot_peak_volatility(df, output_dir)

    logger.info("Generating correlation heatmap…")
    plot_correlation_heatmap(df, output_dir)

    # ── Revenue and session efficiency plots ─────────────────────────────────
    logger.info("Generating revenue analysis plot…")
    plot_revenue_analysis(df, acn, output_dir)

    logger.info("Generating ACN session efficiency plot…")
    plot_acn_session_efficiency(acn, output_dir)

    logger.info("Generating tariff narrative plot…")
    plot_tariff_narrative(df, output_dir)

    # ── Feature importance (graceful XGBoost fallback) ───────────────────────
    logger.info("Generating feature importance plot…")
    plot_feature_importance(df, output_dir)

    # ── Post-run plots (require orchestrator outputs) ─────────────────────────
    predictions_path = str(Path(OUTPUTS_DIR) / "predictions.csv")
    outcomes_path = str(Path(OUTPUTS_DIR) / "agentic_outcomes.csv")

    plot_predicted_vs_actual(predictions_path, output_dir)
    plot_reward_convergence(outcomes_path, output_dir)
    plot_theta_evolution(outcomes_path, output_dir)

    logger.info("EDA complete — outputs saved to %s", output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from src.utils.logging_utils import configure_logging

    configure_logging("INFO")

    parser = argparse.ArgumentParser(description="OP'26 EDA Visualisation Module")
    parser.add_argument(
        "--base",
        default=PROCESSED_BASE_PATH,
        help="Path to unified_analytical_base.csv",
    )
    parser.add_argument(
        "--acn",
        default=RAW_ACN_PATH,
        help="Path to acndata_sessions.json.xlsx",
    )
    parser.add_argument(
        "--out",
        default=EDA_OUTPUTS_DIR,
        help="Output directory for EDA plots",
    )
    args = parser.parse_args()
    run_eda(base_path=args.base, acn_path=args.acn, output_dir=args.out)
