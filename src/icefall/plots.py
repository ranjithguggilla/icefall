"""README charts, drawn from the numbers in metrics.json and the saved tables."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INK = "#1f2a36"
BLUE = "#2f6f9f"
ORANGE = "#d9822b"
GREY = "#8c97a3"
RED = "#b4443c"
GREEN = "#3f8a5a"


def _style(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel(ylabel, color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(axis="y", color="#e6eaee", lw=0.7)
    ax.set_axisbelow(True)


def era_drift(by_year, path):
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    ax.plot(by_year["year"], by_year["observed"], color=INK, lw=2, marker="o", ms=4, label="observed")
    ax.plot(by_year["year"], by_year["model"], color=BLUE, lw=1.8, marker="s", ms=3.5, label="shipped model")
    ax.plot(by_year["year"], by_year["lookup"], color=ORANGE, lw=1.5, ls="--", label="peak history lookup")
    ax.set_ylim(0, 0.8)
    _style(ax, "Held-out years: share of climbers who summit", "expedition year", "summit rate")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def model_compare(tab, path):
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    names = tab["model"].tolist()
    ll = tab["valid_log_loss"].to_numpy()
    se = tab["se"].to_numpy()
    colors = [GREY if n == "lookup" else BLUE for n in names]
    ax.barh(names, ll, xerr=se, color=colors, ecolor=INK, capsize=3)
    ax.invert_yaxis()
    ax.set_xlim(0.58, max(ll + se) + 0.01)
    for i, v in enumerate(ll):
        ax.text(v + se[i] + 0.0015, i, f"{v:.4f}", va="center", fontsize=9, color=INK)
    _style(ax, "Validation log loss (lower is better), bars show 1 SE", "log loss")
    ax.grid(axis="x", color="#e6eaee", lw=0.7)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def leakage(split_tab, leak_tab, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    ax = axes[0]
    ax.barh(split_tab["split"], split_tab["auc"], color=[RED, ORANGE, BLUE])
    ax.invert_yaxis()
    ax.set_xlim(0.5, 0.9)
    for i, v in enumerate(split_tab["auc"]):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
    _style(ax, "Same model, three ways to hold data out", "test AUC")
    ax.grid(axis="x", color="#e6eaee", lw=0.7)
    ax.grid(axis="y", visible=False)
    ax = axes[1]
    colors = [BLUE] + [RED] * (len(leak_tab) - 1)
    ax.barh(leak_tab["scenario"], leak_tab["auc"], color=colors)
    ax.invert_yaxis()
    ax.set_xlim(0.5, 1.02)
    for i, v in enumerate(leak_tab["auc"]):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
    _style(ax, "Outcome columns that must stay out", "test AUC")
    ax.grid(axis="x", color="#e6eaee", lw=0.7)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def reliability(y, p, path, bins=10):
    y = np.asarray(y)
    p = np.asarray(p)
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    xs = [p[idx == b].mean() for b in range(bins) if (idx == b).any()]
    ys = [y[idx == b].mean() for b in range(bins) if (idx == b).any()]
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([0, 1], [0, 1], color=GREY, ls="--", lw=1)
    ax.plot(xs, ys, color=BLUE, marker="o", lw=2)
    ax.set_xlim(0, 0.9)
    ax.set_ylim(0, 0.9)
    _style(ax, "Test reliability (deciles of predicted risk)", "mean predicted", "observed")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def oxygen_effect(rows, path):
    """rows: list of (label, naive, adjusted, lo, hi)."""
    fig, ax = plt.subplots(figsize=(7.6, 2.6))
    ypos = np.arange(len(rows))
    for i, (_, naive, adj, lo, hi) in enumerate(rows):
        ax.plot([lo * 100, hi * 100], [i, i], color=BLUE, lw=3)
        ax.scatter(
            [adj * 100], [i], color=BLUE, s=55, zorder=3, label="adjusted (AIPW) with 95% CI" if i == 0 else None
        )
        ax.scatter([naive * 100], [i], color=ORANGE, marker="D", s=45, zorder=3, label="naive gap" if i == 0 else None)
    ax.set_yticks(ypos)
    ax.set_yticklabels([r[0] for r in rows])
    ax.invert_yaxis()
    ax.axvline(0, color=GREY, lw=1)
    _style(ax, "Summit rate difference attributed to oxygen (percentage points)", "points")
    ax.grid(axis="x", color="#e6eaee", lw=0.7)
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, fontsize=9, loc="center right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def oxygen_cost(policy_tab, sweep, path):
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    colors = [GREY, ORANGE, BLUE, GREEN, "#c9ced4"]
    ax.barh(policy_tab["policy"], policy_tab["total_cost"] / 1e6, color=colors[: len(policy_tab)])
    ax.invert_yaxis()
    for i, v in enumerate(policy_tab["total_cost"] / 1e6):
        ax.text(v + 0.02, i, f"${v:.2f}M", va="center", fontsize=9)
    _style(ax, "Oxygen ordering cost over the held-out seasons", "USD millions")
    ax.grid(axis="x", color="#e6eaee", lw=0.7)
    ax.grid(axis="y", visible=False)
    ax = axes[1]
    ax.plot(sweep["ratio"], sweep["saving_vs_rate"] / 1e3, color=BLUE, marker="o", label="vs peak history rate")
    ax.plot(sweep["ratio"], sweep["saving_vs_full"] / 1e3, color=ORANGE, marker="s", label="vs order for everyone")
    ax.axhline(0, color=GREY, lw=1)
    ax.set_xscale("log")
    _style(ax, "Saving from the model plus offset", "shortage cost / unused-set cost", "USD thousands")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def death_lift(tiers, path):
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    ax.bar(tiers["tier"].astype(str), tiers["share_of_deaths"] * 100, color=BLUE)
    ax.axhline(10, color=GREY, ls="--", lw=1)
    _style(ax, "Held-out deaths by risk tier (tier 1 = highest 10%)", "risk tier", "% of all deaths")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
