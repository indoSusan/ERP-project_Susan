"""
pipeline/visualize.py — Step 09: Matplotlib visualization pipeline

Generates publication-ready PNG figures (300 DPI) from the merged
per-session and cross-session data. No model loading — reads only from
CSVs produced by step 08.

All time-series plots use seconds elapsed as the x-axis and EWMA smoothing
(span=5) to show continuous flow rather than discrete per-turn markers.

Cross-session figures use a time-normalized x-axis (0–100% elapsed) to
enable fair shape comparison across sessions of different lengths.

Per-session figures: outputs/figures/per_session/<vid>/
Cross-session figures: outputs/figures/cross_session/

Caching: per-session figures are skipped if all expected PNGs exist and
--force is not set. Cross-session figures are always regenerated.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for script execution

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import config
from pipeline.cache import (
    get_video_output_dir,
    video_id_from_path,
)
from pipeline.logger import get_logger

log = get_logger(__name__)

# ── Style constants ───────────────────────────────────────────────────────────
HUMAN_COLOR  = "#2166ac"   # blue
AI_COLOR     = "#d6604d"   # red-orange
DPI          = 300
FIGSIZE_WIDE = (14, 4)
FIGSIZE_SQ   = (8, 7)
EWMA_SPAN    = 5            # turns for exponential smoothing
ALPHA_FILL   = 0.18

# Sentiment colour mapping
SENT_COLORS = {"positive": "#4dac26", "neutral": "#b8b8b8", "negative": "#d01c8b"}

# Companion colour and line-style mapping (used in all cross-session figures)
COMPANION_COLORS     = {"noah": "#1f77b4", "charlie": "#d62728"}
COMPANION_LINESTYLE  = {"noah": "-",       "charlie": "--"}
COMPANION_MARKER     = {"noah": "o",       "charlie": "s"}

# Per-session palettes: perceptually distinct colors within each companion family.
# Cool tones for Noah (solid lines), warm tones for Charlie (dashed lines).
# Supports up to 6 sessions per companion; cycles if exceeded.
NOAH_COLORS    = ["#1f77b4", "#2ca02c", "#9467bd", "#17becf", "#e377c2", "#bcbd22"]
CHARLIE_COLORS = ["#d62728", "#ff7f0e", "#e6550d", "#8c564b", "#fd8d3c", "#a63603"]


def _apply_style() -> None:
    """Apply a clean, consistent matplotlib style."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "axes.grid":        True,
        "grid.alpha":       0.3,
        "font.family":      "sans-serif",
        "font.size":        10,
        "axes.titlesize":   11,
        "axes.labelsize":   10,
        "legend.fontsize":  9,
    })


def _save(fig: plt.Figure, path: Path) -> None:
    """Save figure and close to free memory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.debug("[09] Saved: %s", path.name)


def _pct_elapsed(df: pd.DataFrame) -> pd.Series:
    """Return 0–100 normalised position for each turn's start time."""
    s_min = df["start"].min()
    s_max = df["start"].max()
    if s_max == s_min:
        return pd.Series(np.zeros(len(df)), index=df.index)
    return (df["start"] - s_min) / (s_max - s_min) * 100


# ── Per-session figure functions ──────────────────────────────────────────────

def fig_sentiment_flow(df: pd.DataFrame, vid: str, out_dir: Path) -> None:
    """Filled EWMA sentiment curves: human vs AI over time."""
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    for role, color, label in [("human", HUMAN_COLOR, "Human"), ("ai", AI_COLOR, "AI")]:
        sub = df[df["role"] == role].sort_values("start")
        if sub.empty:
            continue
        x = sub["start"].values
        y = sub["sentiment_score"].ewm(span=EWMA_SPAN).mean().values
        ax.plot(x, y, color=color, lw=2, label=label)
        ax.fill_between(x, y, 0, alpha=ALPHA_FILL, color=color)

    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.set(
        title=f"Sentiment Flow — {vid}",
        xlabel="Time (seconds)",
        ylabel="Sentiment score (EWMA)",
        ylim=(-1.1, 1.1),
    )
    ax.legend()
    _save(fig, out_dir / "sentiment_flow.png")


def fig_sycophancy_flow(df: pd.DataFrame, vid: str, out_dir: Path) -> None:
    """AI sycophancy score over time, with cascade highlights."""
    ai = df[df["role"] == "ai"].sort_values("start")
    if ai.empty:
        return
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    x = ai["start"].values
    y = ai["sycophancy_score"].fillna(0).ewm(span=EWMA_SPAN).mean().values
    threshold = y.mean() + y.std() if len(y) > 1 else y.mean()

    ax.plot(x, y, color=AI_COLOR, lw=1.5, label="Sycophancy score (EWMA)")
    ax.fill_between(x, y, 0, alpha=ALPHA_FILL, color=AI_COLOR)
    ax.fill_between(x, y, threshold, where=(y > threshold),
                    alpha=0.35, color="#ff7f00", label="Cascade zone (>1σ)")
    ax.axhline(threshold, color="#ff7f00", lw=0.8, ls="--", alpha=0.6)

    ax.set(
        title=f"Sycophancy Flow (AI) — {vid}",
        xlabel="Time (seconds)",
        ylabel="Sycophancy score (EWMA)",
    )
    ax.legend()
    _save(fig, out_dir / "sycophancy_flow.png")


def fig_prosody_flow(df: pd.DataFrame, vid: str, out_dir: Path) -> None:
    """F0 (pitch) EWMA for both speakers over time."""
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    for role, color, label in [("human", HUMAN_COLOR, "Human F0"), ("ai", AI_COLOR, "AI F0")]:
        sub = df[(df["role"] == role) & df["f0_mean"].notna()].sort_values("start")
        if sub.empty:
            continue
        x = sub["start"].values
        y = sub["f0_mean"].ewm(span=EWMA_SPAN).mean().values
        ax.plot(x, y, color=color, lw=2, label=label)
        ax.fill_between(x, y, sub["f0_mean"].min(), alpha=ALPHA_FILL, color=color)

    ax.set(
        title=f"Pitch (F0) Flow — {vid}",
        xlabel="Time (seconds)",
        ylabel="Mean F0 (Hz, EWMA)",
    )
    ax.legend()
    _save(fig, out_dir / "prosody_flow.png")


def fig_turn_length_flow(pairs_df: pd.DataFrame, vid: str, out_dir: Path) -> None:
    """AI/human turn-length ratio over time."""
    if pairs_df.empty:
        return
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    x = pairs_df["turn_start"].values
    y = pairs_df["turn_length_ratio"].fillna(1).ewm(span=EWMA_SPAN).mean().values

    ax.plot(x, y, color=AI_COLOR, lw=1.5, label="Turn length ratio (EWMA)")
    ax.fill_between(x, y, 1, where=(y > 1), alpha=0.2, color=AI_COLOR, label="AI talks more")
    ax.fill_between(x, y, 1, where=(y < 1), alpha=0.2, color=HUMAN_COLOR, label="Human talks more")
    ax.axhline(1.0, color="black", lw=0.8, ls="--", alpha=0.5)

    ax.set(
        title=f"Turn Length Ratio (AI words / Human words) — {vid}",
        xlabel="Time (seconds)",
        ylabel="Word-count ratio (EWMA)",
    )
    ax.legend()
    _save(fig, out_dir / "turn_length_flow.png")


def fig_windowed_dynamics(windowed_df: pd.DataFrame, vid: str, out_dir: Path) -> None:
    """4-panel windowed dynamics subplot."""
    if windowed_df.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharex=True)
    axes = axes.flatten()

    x = windowed_df["window_start"].values

    panels = [
        ("ai_sentiment_mean",    "AI Sentiment (mean)",    AI_COLOR),
        ("ai_sycophancy_rate",   "AI Sycophancy rate",     "#ff7f00"),
        ("ai_backchannel_rate",  "AI Back-channel rate",   "#984ea3"),
        ("local_AMS",            "Local AMS (Pearson r)",  "#006d2c"),
    ]

    for ax, (col, title, color) in zip(axes, panels):
        if col not in windowed_df.columns:
            ax.set_visible(False)
            continue
        y = windowed_df[col].values
        ax.bar(x, y, width=config.WINDOW_STEP_S * 0.85, color=color, alpha=0.7)
        ax.plot(x, pd.Series(y).fillna(0).ewm(span=3).mean().values,
                color=color, lw=1.5, ls="--")
        ax.set_title(title)
        ax.set_ylabel("Value")

    for ax in axes[-2:]:
        ax.set_xlabel("Window start (seconds)")

    fig.suptitle(f"Windowed Dynamics (2-min windows) — {vid}", fontsize=12)
    plt.tight_layout()
    _save(fig, out_dir / "windowed_dynamics.png")


def fig_cascade_map(df: pd.DataFrame, vid: str, out_dir: Path) -> None:
    """Session timeline as a horizontal coloured ribbon of turn blocks."""
    df_sorted = df.sort_values("start").reset_index(drop=True)
    if df_sorted.empty:
        return

    session_end = df_sorted["end"].max() if "end" in df_sorted.columns else df_sorted["start"].max()
    fig, axes   = plt.subplots(2, 1, figsize=(16, 3), sharex=True)

    for ax_idx, role in enumerate(["human", "ai"]):
        ax    = axes[ax_idx]
        sub   = df_sorted[df_sorted["role"] == role]
        label = role.capitalize()

        for _, row in sub.iterrows():
            start = row["start"]
            end   = row.get("end", start + 5)
            sent  = row.get("sentiment_label", "neutral")
            color = SENT_COLORS.get(sent, "#cccccc")
            ax.barh(0, end - start, left=start, height=0.8,
                    color=color, edgecolor="white", linewidth=0.3)

        ax.set_yticks([])
        ax.set_ylabel(label, fontsize=10, rotation=0, labelpad=40, va="center")
        ax.set_xlim(0, session_end)

    # Legend
    legend_patches = [
        mpatches.Patch(color=SENT_COLORS["positive"], label="Positive"),
        mpatches.Patch(color=SENT_COLORS["neutral"],  label="Neutral"),
        mpatches.Patch(color=SENT_COLORS["negative"], label="Negative"),
    ]
    axes[0].legend(handles=legend_patches, loc="upper right", ncol=3, fontsize=8)

    axes[-1].set_xlabel("Time (seconds)")
    fig.suptitle(f"Sentiment Cascade Map — {vid}", fontsize=11)
    plt.tight_layout()
    _save(fig, out_dir / "cascade_map.png")


def fig_entrainment_flow(df: pd.DataFrame, vid: str, out_dir: Path) -> None:
    """AI lexical entrainment over time."""
    ai = df[df["role"] == "ai"].sort_values("start")
    if ai.empty or "lexical_entrainment" not in ai.columns:
        return
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    x = ai["start"].values
    y = ai["lexical_entrainment"].fillna(0).ewm(span=EWMA_SPAN).mean().values
    ax.plot(x, y, color="#6a3d9a", lw=1.5)
    ax.fill_between(x, y, 0, alpha=ALPHA_FILL, color="#6a3d9a")
    ax.set(
        title=f"Lexical Entrainment (AI) — {vid}",
        xlabel="Time (seconds)",
        ylabel="Jaccard similarity (EWMA)",
        ylim=(0, None),
    )
    _save(fig, out_dir / "entrainment_flow.png")


def fig_hedging_vs_sycophancy(df: pd.DataFrame, vid: str, out_dir: Path) -> None:
    """Scatter: AI turns — hedging_ratio × sycophancy_score, coloured by quartile."""
    ai = df[df["role"] == "ai"].sort_values("start").reset_index(drop=True)
    if ai.empty:
        return

    ai["quartile"] = pd.qcut(ai.index, q=4, labels=["Q1", "Q2", "Q3", "Q4"])
    q_colors = {"Q1": "#fee5d9", "Q2": "#fc9272", "Q3": "#fb6a4a", "Q4": "#a50f15"}

    fig, ax = plt.subplots(figsize=FIGSIZE_SQ)
    for q, grp in ai.groupby("quartile", observed=True):
        ax.scatter(
            grp["hedging_ratio"].fillna(0),
            grp["sycophancy_score"].fillna(0),
            c=q_colors[q], label=str(q), alpha=0.75, edgecolors="grey", lw=0.3, s=50,
        )

    ax.set(
        title=f"Hedging vs Sycophancy (AI turns) — {vid}",
        xlabel="Hedging ratio (epistemic uncertainty)",
        ylabel="Sycophancy score (flattery density)",
    )
    ax.legend(title="Session quartile")
    _save(fig, out_dir / "hedging_vs_sycophancy.png")


def fig_metric_summary_panel(df: pd.DataFrame, vid: str, out_dir: Path) -> None:
    """8-panel metric overview for fast visual audit."""
    fig, axes = plt.subplots(2, 4, figsize=(18, 7))
    axes = axes.flatten()

    panels = [
        ("start", "sentiment_score", "Sentiment over time", "role"),
        ("start", "sycophancy_score", "Sycophancy over time", "role"),
        ("start", "hedging_ratio",    "Hedging ratio over time", "role"),
        ("start", "f0_mean",          "F0 (pitch) over time", "role"),
        ("start", "lexical_entrainment", "Lexical entrainment (AI)", None),
        ("start", "semantic_sim_prev",   "Semantic sim. to prev turn", None),
        ("start", "speech_rate",         "Speech rate over time", "role"),
        ("start", "pause_total_duration","Pause duration over time", "role"),
    ]

    for ax, (xcol, ycol, title, grp_col) in zip(axes, panels):
        if ycol not in df.columns:
            ax.text(0.5, 0.5, f"{ycol}\nnot available",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_title(title)
            continue
        if grp_col:
            for role, color in [("human", HUMAN_COLOR), ("ai", AI_COLOR)]:
                sub = df[df[grp_col] == role].sort_values(xcol)
                y   = sub[ycol].ewm(span=EWMA_SPAN).mean()
                ax.plot(sub[xcol].values, y.values, color=color, lw=1.2, label=role)
            ax.legend(fontsize=7)
        else:
            sub = df.sort_values(xcol)
            y   = sub[ycol].fillna(0).ewm(span=EWMA_SPAN).mean()
            ax.plot(sub[xcol].values, y.values, color="#6a3d9a", lw=1.2)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Time (s)", fontsize=8)

    fig.suptitle(f"Metric Overview — {vid}", fontsize=12)
    plt.tight_layout()
    _save(fig, out_dir / "metric_summary_panel.png")


# ── Cross-session figure functions ────────────────────────────────────────────

def fig_srhi_radar(srhi_df: pd.DataFrame, out_dir: Path) -> None:
    """Spider/radar chart: one polygon per session, 7 SRHI axes (normalized 0–1)."""
    metrics = ["AMS", "VNAC", "PSI", "FDI", "VCI", "LE", "SD"]
    metrics = [m for m in metrics if m in srhi_df.columns]
    if not metrics:
        return

    # Min-max normalise per metric (clip NaN to 0)
    norm = srhi_df[metrics].copy().fillna(0)
    for m in metrics:
        mn, mx = norm[m].min(), norm[m].max()
        norm[m] = (norm[m] - mn) / (mx - mn) if mx > mn else 0.0

    N   = len(metrics)
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]  # close the loop

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})

    noah_rows    = [r for _, r in srhi_df.iterrows() if "noah"    in r["session_id"].lower()]
    charlie_rows = [r for _, r in srhi_df.iterrows() if "charlie" in r["session_id"].lower()]
    other_rows   = [r for _, r in srhi_df.iterrows()
                    if "noah" not in r["session_id"].lower()
                    and "charlie" not in r["session_id"].lower()]

    def _plot_session(row, color, ls):
        values = [float(norm.loc[row.name, m]) if m in norm.columns else 0.0 for m in metrics]
        values += values[:1]
        ax.plot(angles, values, lw=1.5, color=color, ls=ls, label=row["session_id"])
        ax.fill(angles, values, alpha=0.10, color=color)

    for i, row in enumerate(noah_rows):
        _plot_session(row, NOAH_COLORS[i % len(NOAH_COLORS)], COMPANION_LINESTYLE["noah"])
    for i, row in enumerate(charlie_rows):
        _plot_session(row, CHARLIE_COLORS[i % len(CHARLIE_COLORS)], COMPANION_LINESTYLE["charlie"])
    for i, row in enumerate(other_rows):
        _plot_session(row, "#2ca02c", "-")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, size=11)
    ax.set_title("SRHI Radar — all sessions (normalised)", size=12, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
    _save(fig, out_dir / "srhi_radar.png")


def _srhi_bars_panel(srhi_df: pd.DataFrame, metrics: list[str], title: str, ax: plt.Axes) -> None:
    """Draw grouped bars for the given metrics onto ax."""
    n_sess    = len(srhi_df)
    bar_width = 0.7 / n_sess
    x         = np.arange(len(metrics))
    noah_i = charlie_i = 0

    for i, (_, row) in enumerate(srhi_df.iterrows()):
        sid = row["session_id"]
        if "noah" in sid.lower():
            color = NOAH_COLORS[noah_i % len(NOAH_COLORS)]
            noah_i += 1
        elif "charlie" in sid.lower():
            color = CHARLIE_COLORS[charlie_i % len(CHARLIE_COLORS)]
            charlie_i += 1
        else:
            color = "#888888"
        offset = (i - n_sess / 2 + 0.5) * bar_width
        vals   = [float(row.get(m, 0)) if pd.notna(row.get(m)) else 0.0 for m in metrics]
        ax.bar(x + offset, vals, width=bar_width, color=color, label=sid, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_title(title)
    ax.set_ylabel("Value")
    ax.axhline(0, color="black", lw=0.6, ls="--")
    ax.legend(fontsize=8, bbox_to_anchor=(1, 1))


def fig_srhi_bars(srhi_df: pd.DataFrame, out_dir: Path) -> None:
    """Two grouped bar charts: VCI alone, then the remaining 6 SRHI metrics."""
    all_metrics = ["AMS", "VNAC", "PSI", "FDI", "VCI", "LE", "SD"]
    available   = [m for m in all_metrics if m in srhi_df.columns]
    if not available:
        return

    other_metrics = [m for m in available if m != "VCI"]
    has_vci       = "VCI" in available

    # ── Panel A: non-VCI metrics ──────────────────────────────────────────────
    if other_metrics:
        fig, ax = plt.subplots(figsize=(max(12, len(other_metrics) * 2), 5))
        _srhi_bars_panel(
            srhi_df, other_metrics,
            "SRHI Metrics (excl. VCI) — all sessions",
            ax,
        )
        plt.tight_layout()
        _save(fig, out_dir / "srhi_bars.png")

    # ── Panel B: VCI alone ────────────────────────────────────────────────────
    if has_vci:
        fig, ax = plt.subplots(figsize=(5, 5))
        _srhi_bars_panel(
            srhi_df, ["VCI"],
            "VCI (Verbal Congruence Index) — all sessions",
            ax,
        )
        plt.tight_layout()
        _save(fig, out_dir / "srhi_bars_vci.png")


def fig_correlation_matrix(all_df: pd.DataFrame, out_dir: Path) -> None:
    """Seaborn heatmap of all numeric session-level metrics."""
    numeric = all_df.select_dtypes(include=[np.number])
    if numeric.empty or len(numeric.columns) < 2:
        return
    corr = numeric.corr()

    fig, ax = plt.subplots(figsize=(max(10, len(corr) * 0.6 + 2),
                                    max(8,  len(corr) * 0.6)))
    sns.heatmap(
        corr,
        annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, linewidths=0.4, ax=ax,
        annot_kws={"size": 7},
    )
    ax.set_title("Turn-level metric correlation matrix (all sessions)", fontsize=11)
    plt.tight_layout()
    _save(fig, out_dir / "correlation_matrix.png")


def fig_ams_vs_fdi(srhi_df: pd.DataFrame, out_dir: Path) -> None:
    """Scatter: AMS × FDI, labeled, sized by VCI."""
    if "AMS" not in srhi_df.columns or "FDI" not in srhi_df.columns:
        return
    df_plot = srhi_df.dropna(subset=["AMS", "FDI"])
    if df_plot.empty:
        return

    sizes = df_plot["VCI"].fillna(5).values * 30 + 40

    fig, ax = plt.subplots(figsize=FIGSIZE_SQ)

    # Alternate label offsets to reduce collisions when points cluster.
    _offsets = [(6, 6), (-6, 6), (6, -10), (-6, -10), (14, 0), (-14, 0)]
    texts = []
    for i, (_, row) in enumerate(df_plot.iterrows()):
        sid       = row["session_id"]
        companion = "noah" if "noah" in sid.lower() else "charlie"
        color     = COMPANION_COLORS[companion]
        marker    = COMPANION_MARKER[companion]
        size      = float(row.get("VCI", 5) or 5) * 30 + 40
        ax.scatter(row["AMS"], row["FDI"], s=size, c=color, marker=marker,
                   alpha=0.85, edgecolors="grey", lw=0.5)
        dx, dy = _offsets[i % len(_offsets)]
        ann = ax.annotate(
            sid, (row["AMS"], row["FDI"]),
            fontsize=7.5, xytext=(dx, dy), textcoords="offset points",
            arrowprops=dict(arrowstyle="-", color="grey", lw=0.5),
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
        )
        texts.append(ann)

    try:
        from adjustText import adjust_text
        adjust_text(texts, ax=ax)
    except ImportError:
        pass

    # Companion legend
    legend_patches = [
        mpatches.Patch(color=COMPANION_COLORS["noah"],    label="Noah  (circle)"),
        mpatches.Patch(color=COMPANION_COLORS["charlie"], label="Charlie (square)"),
    ]
    ax.legend(handles=legend_patches, fontsize=8)

    ax.axvline(0, color="grey", lw=0.8, ls="--")
    ax.axhline(df_plot["FDI"].mean(), color="grey", lw=0.8, ls="--")
    ax.set(
        title="Affective Mirroring vs Flattery Density",
        xlabel="AMS (Pearson r)",
        ylabel="FDI (flattery density)",
    )
    ax.text(0.02, 0.97, "Point size = VCI", transform=ax.transAxes, fontsize=8, va="top")
    _save(fig, out_dir / "ams_vs_fdi.png")


def _time_normalised_overlay(
    all_df: pd.DataFrame,
    col: str,
    title: str,
    filename: str,
    ylabel: str,
    out_dir: Path,
    role_filter: str | None = "ai",
) -> None:
    """Helper for time-normalized cross-session overlay plots.

    Lines are styled by companion: Noah = solid blue shades, Charlie = dashed
    orange shades. Each session gets its own shade within that companion palette.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    sessions = sorted(all_df["session_id"].unique())

    # Separate sessions by companion so we can shade within each group
    noah_sessions    = [v for v in sessions if "noah"    in v.lower()]
    charlie_sessions = [v for v in sessions if "charlie" in v.lower()]
    other_sessions   = [v for v in sessions if v not in noah_sessions and v not in charlie_sessions]

    # Build per-session color: distinct colors within each companion family
    session_style: dict[str, dict] = {}
    for i, v in enumerate(noah_sessions):
        session_style[v] = {"color": NOAH_COLORS[i % len(NOAH_COLORS)],
                            "ls": COMPANION_LINESTYLE["noah"]}
    for i, v in enumerate(charlie_sessions):
        session_style[v] = {"color": CHARLIE_COLORS[i % len(CHARLIE_COLORS)],
                            "ls": COMPANION_LINESTYLE["charlie"]}
    _other_palette = ["#2ca02c", "#9467bd", "#7f7f7f", "#bcbd22"]
    for i, v in enumerate(other_sessions):
        session_style[v] = {"color": _other_palette[i % len(_other_palette)], "ls": "-"}

    for vid in sessions:
        sess = all_df[all_df["session_id"] == vid].copy()
        if role_filter:
            sess = sess[sess["role"] == role_filter]
        sess = sess.sort_values("start")

        if sess.empty or col not in sess.columns:
            continue

        sess["pct"] = _pct_elapsed(sess)
        sess["bin"] = (sess["pct"] // (100 / config.TIMELINE_BINS)).astype(int)
        binned      = sess.groupby("bin")[col].mean()
        x           = binned.index.values * (100 / config.TIMELINE_BINS)
        y           = binned.values
        style       = session_style.get(vid, {"color": "grey", "ls": "-"})

        ax.plot(x, pd.Series(y).ewm(span=max(2, EWMA_SPAN // 2)).mean().values,
                color=style["color"], ls=style["ls"], lw=1.5, label=vid, alpha=0.9)

    # Companion legend patches
    legend_handles = ax.get_lines().copy() if ax.get_lines() else []
    companion_patches = [
        mpatches.Patch(color=COMPANION_COLORS["noah"],    label="Noah (solid)"),
        mpatches.Patch(color=COMPANION_COLORS["charlie"], label="Charlie (dashed)"),
    ]
    ax.legend(handles=legend_handles + companion_patches, fontsize=7,
              bbox_to_anchor=(1, 1), title="Session / Companion")

    ax.set(title=title, xlabel="Session elapsed (%)", ylabel=ylabel)
    plt.tight_layout()
    _save(fig, out_dir / filename)


def fig_subject_comparison(srhi_df: pd.DataFrame, out_dir: Path) -> None:
    """Side-by-side box plots: Noah sessions vs Charlie sessions."""
    if "session_id" not in srhi_df.columns:
        return

    metrics = ["AMS", "VNAC", "PSI", "FDI", "VCI", "LE", "SD"]
    metrics = [m for m in metrics if m in srhi_df.columns]

    srhi_df = srhi_df.copy()
    # Use companion column if present (populated by merge.py); fall back to session_id parsing
    if "companion" in srhi_df.columns:
        srhi_df["subject"] = srhi_df["companion"].str.capitalize()
    else:
        srhi_df["subject"] = srhi_df["session_id"].apply(
            lambda s: "Noah" if "noah" in s.lower() else "Charlie"
        )

    n_metrics = len(metrics)
    if n_metrics == 0:
        return

    companions = sorted(srhi_df["subject"].unique())
    palette    = {c: COMPANION_COLORS.get(c.lower(), "#888888") for c in companions}

    fig, axes = plt.subplots(1, n_metrics, figsize=(n_metrics * 2.5, 5), sharey=False)
    if n_metrics == 1:
        axes = [axes]

    for ax, m in zip(axes, metrics):
        sns.boxplot(data=srhi_df, x="subject", y=m, hue="subject", palette=palette,
                    ax=ax, order=companions)
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        sns.stripplot(data=srhi_df, x="subject", y=m,
                      order=companions,
                      color="black", size=5, alpha=0.5, ax=ax)
        ax.set_title(m, fontsize=10)
        ax.set_xlabel("")

    fig.suptitle("SRHI Metrics by AI Companion", fontsize=12)
    plt.tight_layout()
    _save(fig, out_dir / "subject_comparison.png")


# ── Per-session orchestration ─────────────────────────────────────────────────

def run_per_session(video_path: Path, force: bool = False) -> None:
    """Generate all per-session figures for one video."""
    vid     = video_id_from_path(video_path)
    out_dir = config.FIGURES_DIR / "per_session" / vid
    out_dir.mkdir(parents=True, exist_ok=True)

    # Cache check: skip if all expected PNGs exist and force is off
    if not force:
        missing = [f for f in config.EXPECTED_FIGURES_PER_SESSION
                   if not (out_dir / f).exists()]
        if not missing:
            log.info("[09] Cached — all per-session figures exist for %s", vid)
            return
        log.debug("[09] %d figure(s) missing for %s — regenerating all", len(missing), vid)

    # Load merged.csv
    merged_p = config.PER_VIDEO_DIR / vid / "merged.csv"
    if not merged_p.exists():
        log.warning("[09] merged.csv not found for %s — skipping per-session figures", vid)
        return

    df = pd.read_csv(merged_p)
    log.info("[09] Generating per-session figures for %s (%d turns)", vid, len(df))

    # Load turn_length_pairs.csv (optional)
    pairs_p = config.PER_VIDEO_DIR / vid / "turn_length_pairs.csv"
    pairs_df = pd.read_csv(pairs_p) if pairs_p.exists() else pd.DataFrame()

    # Load windowed_metrics.csv filtered for this session
    win_path = config.MERGED_DIR / "windowed_metrics.csv"
    if win_path.exists():
        windowed_df = pd.read_csv(win_path)
        windowed_df = windowed_df[windowed_df["session_id"] == vid]
    else:
        windowed_df = pd.DataFrame()

    _apply_style()

    figure_tasks = [
        ("sentiment_flow.png",       lambda: fig_sentiment_flow(df, vid, out_dir)),
        ("sycophancy_flow.png",      lambda: fig_sycophancy_flow(df, vid, out_dir)),
        ("prosody_flow.png",         lambda: fig_prosody_flow(df, vid, out_dir)),
        ("turn_length_flow.png",     lambda: fig_turn_length_flow(pairs_df, vid, out_dir)),
        ("windowed_dynamics.png",    lambda: fig_windowed_dynamics(windowed_df, vid, out_dir)),
        ("cascade_map.png",          lambda: fig_cascade_map(df, vid, out_dir)),
        ("entrainment_flow.png",     lambda: fig_entrainment_flow(df, vid, out_dir)),
        ("hedging_vs_sycophancy.png",lambda: fig_hedging_vs_sycophancy(df, vid, out_dir)),
        ("metric_summary_panel.png", lambda: fig_metric_summary_panel(df, vid, out_dir)),
    ]

    for fname, fn in tqdm(figure_tasks, desc=f"[09] Figures for {vid}", unit="fig", leave=False):
        try:
            fn()
        except Exception as exc:
            log.error("[09] Failed to generate %s for %s: %s", fname, vid, exc, exc_info=True)


# ── Cross-session orchestration ───────────────────────────────────────────────

def run_cross_session() -> None:
    """Generate all cross-session comparison figures."""
    cross_dir = config.FIGURES_DIR / "cross_session"
    cross_dir.mkdir(parents=True, exist_ok=True)

    srhi_p   = config.MERGED_DIR / "srhi_summary.csv"
    all_p    = config.MERGED_DIR / "all_videos.csv"
    wind_p   = config.MERGED_DIR / "windowed_metrics.csv"

    if not srhi_p.exists():
        log.warning("[09] srhi_summary.csv not found — skipping cross-session figures")
        return

    srhi_df = pd.read_csv(srhi_p)
    log.info("[09] Generating cross-session figures for %d sessions", len(srhi_df))

    _apply_style()

    # Load all_videos if available
    all_df    = pd.read_csv(all_p) if all_p.exists() else pd.DataFrame()

    fig_srhi_radar(srhi_df, cross_dir)
    fig_srhi_bars(srhi_df, cross_dir)
    fig_subject_comparison(srhi_df, cross_dir)
    fig_ams_vs_fdi(srhi_df, cross_dir)

    if not all_df.empty:
        fig_correlation_matrix(all_df, cross_dir)

        _time_normalised_overlay(
            all_df, col="sentiment_score",
            title="AI Sentiment Trajectories (all sessions, time-normalised)",
            filename="sentiment_trajectories_aligned.png",
            ylabel="Sentiment score (EWMA)", out_dir=cross_dir,
            role_filter="ai",
        )
        _time_normalised_overlay(
            all_df, col="sycophancy_score",
            title="AI Sycophancy Evolution (all sessions, time-normalised)",
            filename="sycophancy_evolution.png",
            ylabel="Sycophancy score (EWMA)", out_dir=cross_dir,
            role_filter="ai",
        )

    # turn_length_ratio lives in per-session turn_length_pairs.csv, not all_videos.csv.
    # Load and concatenate all available files for the cross-session overlay.
    pairs_frames = []
    for pairs_p in sorted(config.PER_VIDEO_DIR.glob("*/turn_length_pairs.csv")):
        try:
            pairs_frames.append(pd.read_csv(pairs_p))
        except Exception as exc:
            log.warning("[09] Could not load %s: %s", pairs_p, exc)

    if pairs_frames:
        all_pairs_df = pd.concat(pairs_frames, ignore_index=True)
        # Rename turn_start → start so _time_normalised_overlay can use it
        all_pairs_df = all_pairs_df.rename(columns={"turn_start": "start"})
        log.info("[09] Turn-length overlay: %d sessions, %d turn pairs",
                 all_pairs_df["session_id"].nunique(), len(all_pairs_df))
        _time_normalised_overlay(
            all_pairs_df, col="turn_length_ratio",
            title="Turn Length Ratio Evolution (all sessions, time-normalised)",
            filename="turn_length_evolution.png",
            ylabel="Word-count ratio (EWMA)", out_dir=cross_dir,
            role_filter=None,
        )
    else:
        log.warning("[09] No turn_length_pairs.csv files found — skipping turn_length_evolution")

    log.info("[09] Cross-session figures complete")


# ── Public API ────────────────────────────────────────────────────────────────

def run(video_path: str | Path | None = None, force: bool = False) -> None:
    """
    Step 09 entry point.

    If video_path is provided, generates per-session figures for that video
    plus updated cross-session figures.

    If video_path is None, generates cross-session figures only (useful when
    all individual videos have already been visualised).

    Args:
        video_path: MP4 source path, or None for cross-session only.
        force:      Bypass per-session figure cache.
    """
    if video_path is not None:
        run_per_session(Path(video_path), force=force)
    run_cross_session()


def run_batch(video_paths: list[Path], force: bool = False) -> None:
    """Run step 09 for all videos, then cross-session figures."""
    log.info("[09] Starting visualization for %d video(s)", len(video_paths))
    _apply_style()

    for video_path in tqdm(video_paths, desc="[09] Per-session figures", unit="video"):
        try:
            run_per_session(video_path, force=force)
        except Exception as exc:
            log.error("[09] FAILED per-session for %s: %s", video_path.name, exc, exc_info=True)

    run_cross_session()
    log.info("[09] Visualization complete")


if __name__ == "__main__":
    import sys
    _force = "--force" in sys.argv
    _video_arg = next((a for a in sys.argv[1:] if a != "--force"), None)
    if _video_arg:
        run(Path(_video_arg), force=_force)
    else:
        run_cross_session()
