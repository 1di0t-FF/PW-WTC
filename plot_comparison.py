"""Paper figures for the comparison archive: one per attack mode.

    python plot_comparison.py            # all three figures
    python plot_comparison.py JAM-SP     # one figure

Reads  results/<MODE>/<scene>/  (satellites/Gxx.csv, clkbm.csv)
       input/<MODE>/<scene>/thresholds.csv   (decision values)
       config.json                           (scenes + display satellites)
Writes figures/<MODE>_comparison.png

"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
METHODS = ["PW-WTC", "E-CCDC", "CD2-P", "clkBM"]
COLUMN = {"PW-WTC": "PW_WTC", "E-CCDC": "E_CCDC", "CD2-P": "CD2_P", "clkBM": "clk"}
THRESHOLD_COLUMN = {"E-CCDC": "E_CCDC_threshold", "CD2-P": "CD2_P_threshold"}
COLORS = [(0, .447, .741), (.85, .325, .098), (.929, .694, .125)]
THRESHOLD_COLOR = (.45, .18, .55)
AU_ROWS = 900
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 8})

CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))


def scenes_of(mode):
    """[(scene, display satellites)] in config order."""
    return [(s, c["display"]) for s, c in CONFIG.items() if c["mode"] == mode]


def load_common(mode, scene, satellites):
    """Per satellite and segment: rows where all four quantities are present.

    x axis: FCS/PCS re-bases end_index by -598 (first common second = 1);
    JAM-SP segments use compact per-segment coordinates (1..n).
    """
    stats = {s: pd.read_csv(RESULTS / mode / scene / "satellites" / f"{s}.csv")
             for s in satellites}
    clock = pd.read_csv(RESULTS / mode / scene / "clkbm.csv")
    thr = pd.read_csv(ROOT / "input" / mode / scene / "thresholds.csv").set_index("satellite")
    series = {}
    for sat in satellites:
        merged = stats[sat].merge(
            clock[["segment", "end_index", "DeltaClkB_nsps"]], on=["segment", "end_index"])
        per = {}
        for seg in merged.segment.unique():
            sub = merged[merged.segment == seg]
            sub = sub[sub.PW_WTC.notna() & sub.E_CCDC.notna()
                      & sub.CD2_P.notna() & sub.DeltaClkB_nsps.notna()].reset_index(drop=True)
            if mode == "JAM-SP":
                x = np.arange(1, len(sub) + 1, dtype=float)
            else:
                x = sub.end_index.to_numpy(float) - 598.0
            per[seg] = {"x": x,
                        **{c: sub[c].to_numpy(float) for c in ("PW_WTC", "E_CCDC", "CD2_P")},
                        "clk": sub.DeltaClkB_nsps.to_numpy(float)}
        series[sat] = per
    return series, thr, clock


def joint_scale(segments, refs):
    """One min-max scale shared across everything drawn in the panel (v1 rule)."""
    values = np.concatenate([s[np.isfinite(s)] for s in segments] + [np.asarray(refs, float)])
    lo, hi = float(values.min()), float(values.max())
    return (lambda v: (v - lo) / (hi - lo)) if hi > lo else (lambda v: v * 0.0)


def segment_scale(signal, lower, upper):
    values = np.concatenate([signal[np.isfinite(signal)], [lower, upper]])
    lo, hi = float(values.min()), float(values.max())
    hi = max(hi, lo + 1.0)
    return lambda v: (v - lo) / (hi - lo)


def style_axes(ax, show_x):
    ax.set_ylim(-0.03, 1.03)
    ax.tick_params(labelsize=8, length=3)
    ax.set_yticks([0, 0.5, 1])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.55)
    if not show_x:
        ax.set_xticklabels([])


def new_figure(legend_labels):
    fig = plt.figure(figsize=(8.40, 8.55))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    legend_axis = fig.add_axes([.10, .935, .80, .03])
    legend_axis.axis("off")
    handles = [plt.Line2D([], [], color=c, lw=1.1) for c in COLORS]
    handles.append(plt.Line2D([], [], color=THRESHOLD_COLOR, lw=.9, ls="--"))
    legend_axis.legend(handles, legend_labels, ncol=4, loc="upper center",
                       frameon=False, fontsize=8)
    return fig


PANEL = dict(left=[.071, .546], bottom=[.497, .048], width=.435, height=.415,
             title=.030, gap=.008)   # gap: vertical row spacing; column pitch = width + .040
# Shared figure-level margins so all three figures align identically.
LABEL_DX = .045    # method name: distance left of the panel edge (figure units)
TIME_DY = .034     # "Time (s)" baseline below the last row
SPOOF_COLOR, SPOOF_LW = '0.35', .9
BOUND_COLOR, BOUND_LW = '0.45', .55


def decorate(fig, left, bottom, row, method, scene, y, show_method):
    """Scene title over the first row, method name left of the panel, time label."""
    if row == 0:
        fig.text(left + PANEL["width"] / 2, bottom + PANEL["height"] - PANEL["title"] / 2,
                 scene, ha="center", va="center", fontsize=9, fontweight="bold")
    if show_method:
        fig.text(left - LABEL_DX, y + row_height() / 2, method, rotation=90,
                 ha="center", va="center", fontsize=8)
    if row == 3:
        fig.text(left + PANEL["width"] / 2, y - TIME_DY, "Time (s)",
                 ha="center", va="center", fontsize=8)


def row_height():
    return (PANEL["height"] - PANEL["title"] - 3 * PANEL["gap"]) / 4


def row_y(bottom, row):
    h = row_height()
    return bottom + PANEL["height"] - PANEL["title"] - (row + 1) * h - row * PANEL["gap"], h


def save(fig, stem):
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    print(f"saved {stem}.png")


def stream_figure(mode):
    """Shared continuous-axis layout for the FCS and PCS figures."""
    fig = new_figure(["#1", "#2", "#3(target)", "threshold"] if mode == "PCS"
                     else ["#1", "#2", "#3", "threshold"])
    for panel, (scene, satellites) in enumerate(scenes_of(mode)):
        series, thr, clock = load_common(mode, scene, satellites)
        col, prow = panel % 2, panel // 2
        left, bottom = PANEL["left"][col], PANEL["bottom"][prow]
        lo_clock, hi_clock = float(clock.LowerBound_nsps.iloc[0]), float(clock.UpperBound_nsps.iloc[0])
        x_end = max(s["stream"]["x"][-1] for s in series.values())
        for row, method in enumerate(METHODS):
            y, h = row_y(bottom, row)
            ax = fig.add_axes([left, y, PANEL["width"], h])
            if method == "clkBM":
                finite = clock.DeltaClkB_nsps.to_numpy(float)
                scale = segment_scale(finite, lo_clock, hi_clock)
                ax.plot(clock.end_index.to_numpy(float) - 598.0, scale(finite),
                        lw=.85, color=COLORS[0])
                for bound in (lo_clock, hi_clock):
                    ax.axhline(scale(bound), ls="--", lw=.65, color=THRESHOLD_COLOR)
            else:
                column = COLUMN[method]
                curves = [series[s]["stream"][column] for s in satellites]
                if method == "PW-WTC":
                    scale = lambda v: v  # noqa: E731
                    level = float(thr.PW_WTC_threshold.iloc[0])
                else:
                    refs = [float(thr.loc[s, THRESHOLD_COLUMN[method]]) for s in satellites]
                    scale = joint_scale(curves, refs)
                    level = scale(max(refs))
                for index, sat in enumerate(satellites):
                    s = series[sat]["stream"]
                    ax.plot(s["x"], scale(s[column]), lw=.85, color=COLORS[index])
                ax.axhline(level, ls="--", lw=.65, color=THRESHOLD_COLOR)
            ax.axvline(301.5, color=SPOOF_COLOR, lw=SPOOF_LW, ls="--")   # spoof start
            ax.set_xlim(1, x_end)
            style_axes(ax, row == 3)
            ax.set_xticks([t for t in range(200, int(x_end) + 1, 200)])
            if row == 0:
                ax.text(301.5 - 8, .48, "spoof start", fontsize=7.5, color='0.2',
                        ha="right", va="center", clip_on=True)
            decorate(fig, left, bottom, row, method, scene, y, col == 0)
    save(fig, FIGURES / f"{mode}_comparison")


def plot_fcs():
    stream_figure("FCS")


def plot_pcs():
    stream_figure("PCS")


def plot_jamsp():
    fig = new_figure(["#1", "#2", "#3", "threshold"])
    inner_gap = .030
    inner_width = (1 - inner_gap) / 2
    for panel, (scene, satellites) in enumerate(scenes_of("JAM-SP")):
        series, thr, clock = load_common("JAM-SP", scene, satellites)
        col, prow = panel % 2, panel // 2
        left, bottom = PANEL["left"][col], PANEL["bottom"][prow]
        lo_clock, hi_clock = float(clock.LowerBound_nsps.iloc[0]), float(clock.UpperBound_nsps.iloc[0])
        au_len = int(len(series[satellites[0]]["au"]["x"]))
        sp_len = int(len(series[satellites[0]]["sp"]["x"]))
        for sat in satellites:      # right side continues the relative timeline
            series[sat]["sp"]["x"] = np.arange(au_len + 660 + 1,
                                               au_len + 660 + 1 + sp_len, dtype=float)
        for row, method in enumerate(METHODS):
            y, h = row_y(bottom, row)
            ax_l = fig.add_axes([left, y, PANEL["width"] * inner_width, h])
            ax_r = fig.add_axes([left + PANEL["width"] * (inner_width + inner_gap), y,
                                 PANEL["width"] * inner_width, h])
            pos_l, pos_r = ax_l.get_position(), ax_r.get_position()
            fig.add_artist(matplotlib.patches.Rectangle(          # pink jam band
                (pos_l.x1, y), pos_r.x0 - pos_l.x1, h,
                facecolor=(1, .88, .88), alpha=.35, edgecolor="none", zorder=0))
            column = COLUMN[method]
            if method == "clkBM":
                for ax, seg in ((ax_l, "au"), (ax_r, "sp")):
                    s = series[satellites[0]][seg]
                    scale = segment_scale(s["clk"], lo_clock, hi_clock)
                    ax.plot(s["x"], scale(s["clk"]), lw=.85, color=COLORS[0])
                    for bound in (lo_clock, hi_clock):
                        ax.plot(s["x"], scale(np.full_like(s["x"], bound)),
                                ls="--", lw=.65, color=THRESHOLD_COLOR)
            else:
                curves = [np.concatenate([series[s]["au"][column], series[s]["sp"][column]])
                          for s in satellites]
                if method == "PW-WTC":
                    scale = lambda v: v  # noqa: E731
                    level = float(thr.PW_WTC_threshold.iloc[0])
                else:
                    refs = [float(thr.loc[s, THRESHOLD_COLUMN[method]]) for s in satellites]
                    scale = joint_scale(curves, refs)
                    level = scale(max(refs))
                for index, sat in enumerate(satellites):
                    for ax, seg in ((ax_l, "au"), (ax_r, "sp")):
                        s = series[sat][seg]
                        ax.plot(s["x"], scale(s[column]), lw=.85, color=COLORS[index])
                for ax in (ax_l, ax_r):
                    ax.axhline(level, ls="--", lw=.65, color=THRESHOLD_COLOR)
            for ax, x_lo, x_hi in ((ax_l, 1, au_len), (ax_r, au_len + 660 + 1, au_len + 660 + sp_len)):
                ax.axvline(x_hi if ax is ax_l else x_lo, color=BOUND_COLOR, lw=BOUND_LW, ls="--")
                ax.set_xlim(x_lo, x_hi)
                style_axes(ax, row == 3)
                if ax is ax_r:          # right segment carries no y labels
                    ax.set_yticks([])
                    ax.spines["left"].set_visible(False)
                ax.set_xticks([t for t in range(100, x_hi + 1, 100) if x_lo <= t <= x_hi])
            for edge in (pos_l.x1, pos_r.x0):                      # break slashes
                dx, dy = .012 * pos_l.width, .012 * pos_l.height
                fig.add_artist(plt.Line2D([edge - dx, edge + dx], [pos_l.y0 - dy, pos_l.y0 + dy],
                                          transform=fig.transFigure, color='0.15', lw=.8))
            if row == 0:
                ax_l.text(au_len - 35, .48, "JAM", fontsize=7.5, color='0.2',
                          ha="left", va="center", clip_on=True)
            decorate(fig, left, bottom, row, method, scene, y, col == 0)
    save(fig, FIGURES / "JAM-SP_comparison")


PLOTS = {"FCS": plot_fcs, "PCS": plot_pcs, "JAM-SP": plot_jamsp}

if __name__ == "__main__":
    for mode in (sys.argv[1:] or list(PLOTS)):
        PLOTS[mode]()
