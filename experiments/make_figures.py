"""
Review 1 figures, generated ONLY from results/**/*.json (CLAUDE.md rule 1). 300 DPI PNGs -> paper/figures/.

  fig_a_baselines_bar.png        pooled test macro-F1, every main-table method, both baseline recipes
  fig_b_convergence.png          federated val loss and pooled val macro-F1 vs round (FedAvg E=1, E=5), mean ± std
  fig_c_per_client_scatter.png   per-client D4 macro-F1: local-only (x) vs FedAvg E=1 and FedAvg+FT<=2 (y)
  fig_d_label_distribution.png   per-client label distribution (non-IID evidence)
  fig_e_matched_baselines.png    table: AdamW baselines vs matched-SGD baselines, same FedAvg runs

Colours follow the validated reference palette (blue / orange / aqua for series, greys for references; the
aqua series is always direct-labelled because it sits below 3:1 contrast). Static images: no hover layer.

Usage:  python -m experiments.make_figures
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from scipy import stats  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
OUT = REPO / "paper" / "figures"
SEEDS = [0, 1, 2, 3, 4]
DPI = 300

# Reference palette (light mode) — see the dataviz skill's palette.md
SURFACE, INK, INK_2, INK_3, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
REF_SGD, REF_ADAMW, REF_MAJ = "#7a7974", "#b9b8b2", "#d9d8d2"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

CLIENT_SHORT = {
    "2013_Pakistan_eq": "Pakistan EQ '13", "2014_California_Earthquake": "California EQ '14",
    "2014_Chile_Earthquake_cl": "Chile EQ '14 (es)", "2014_Chile_Earthquake_en": "Chile EQ '14 (en)",
    "2014_Hurricane_Odile_Mexico_en": "Hurricane Odile '14", "2014_India_floods": "India floods '14",
    "2014_Pakistan_floods": "Pakistan floods '14", "2014_Philippines_Typhoon_Hagupit_en": "Typhoon Hagupit '14",
    "2015_Cyclone_Pam_en": "Cyclone Pam '15", "2015_Nepal_Earthquake_en": "Nepal EQ '15",
}
HIGHLIGHT = ("2014_Hurricane_Odile_Mexico_en", "2014_Philippines_Typhoon_Hagupit_en")

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 9, "text.color": INK, "axes.labelcolor": INK_2,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.edgecolor": INK_3, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "axes.titleweight": "bold",
    "axes.titlesize": 10, "axes.titlelocation": "left",
})


# ── data access (results JSON only) ────────────────────────────────────────

def _metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))["metrics"]


def runs(group: str, name: str) -> list[dict]:
    return [_metrics(RESULTS / group / f"{name}_seed{s}" / "metrics.json") for s in SEEDS]


def pooled(group: str, name: str) -> np.ndarray:
    return np.array([m["pooled_test_macro_f1"] for m in runs(group, name)])


def ms(x) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    return float(x.mean()), float(x.std(ddof=1))


def save(fig, name: str, provenance: str, caption: str | None = None) -> Path:
    """Footer below the axes: optional caption, then the provenance line (stacked so they never overlap)."""
    y = -0.01
    if caption:
        fig.text(0.01, y, caption, ha="left", va="top", fontsize=7, color=INK_2)
        line = 7 * 1.45 / 72 / fig.get_size_inches()[1]          # one caption line, in figure-height units
        y -= line * (caption.count("\n") + 1) + line * 0.5
    fig.text(0.01, y, provenance, ha="left", va="top", fontsize=6, color=INK_3)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def provenance_of(*groups_names: tuple[str, str]) -> str:
    commits = set()
    for g, n in groups_names:
        for s in SEEDS:
            d = json.loads((RESULTS / g / f"{n}_seed{s}" / "metrics.json").read_text(encoding="utf-8"))
            commits.add(d["git"]["commit"][:7] + ("" if not d["git"]["dirty"] else "-dirty"))
    return "source: results/*.json · commit " + ", ".join(sorted(commits)) + " · seeds 0–4"


# ── (a) bar chart ──────────────────────────────────────────────────────────

MAIN_METHODS = [  # (label, group, run name, family)
    ("Majority class", "silo1_baselines", "majority_global", "majority"),
    ("Local-only · SGD", "silo1_baselines", "local_only", "ref_sgd"),
    ("Local-only · AdamW", "silo1_baselines", "local_only_adamw", "ref_adamw"),
    ("FedAvg E=1", "silo1_fl", "fedavg_E1", "fl"),
    ("FedProx μ=0.001 E=1", "silo1_fl", "fedprox_mu0.001_E1", "fl"),
    ("FedAvg E=5, 200 rounds †", "silo1_fl", "fedavg_E5_R200", "fl"),
    ("FedAvg + FT ≤1 epoch", "silo1_fl", "ft1_fedavg_E1", "ft"),
    ("FedAvg + FT ≤2 epochs", "silo1_fl", "ft2_fedavg_E1", "ft"),
    ("Centralized · SGD", "silo1_baselines", "centralized", "ref_sgd"),
    ("Centralized · AdamW", "silo1_baselines", "centralized_adamw", "ref_adamw"),
]
FAMILY = {"majority": (REF_MAJ, None, "Majority class"), "ref_sgd": (REF_SGD, None, "Baseline · matched SGD"),
          "ref_adamw": (REF_ADAMW, "////", "Baseline · AdamW (strong reference)"),
          "fl": (BLUE, None, "Federated"), "ft": (AQUA, None, "Federated + local fine-tuning")}


def fig_a() -> Path:
    labels, means, stds, fams = [], [], [], []
    for label, g, n, fam in MAIN_METHODS:
        m, s = ms(pooled(g, n))
        labels.append(label); means.append(m); stds.append(s); fams.append(fam)
    y = np.arange(len(labels))[::-1]
    fig, ax = plt.subplots(figsize=(9.4, 4.4))
    for yi, m, s, fam in zip(y, means, stds, fams):
        color, hatch, _ = FAMILY[fam]
        ax.barh(yi, m, height=0.62, color=color, hatch=hatch, edgecolor=SURFACE if hatch is None else INK_3,
                linewidth=0.0 if hatch is None else 0.4)
        ax.errorbar(m, yi, xerr=s, fmt="none", ecolor=INK, elinewidth=0.9, capsize=2.5)
        ax.text(m + s + 0.008, yi, f"{m:.3f}", va="center", ha="left", fontsize=8, color=INK)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 0.75)
    ax.set_xlabel("Pooled test macro-F1 (9 classes), mean ± std over 5 seeds")
    ax.grid(axis="y", visible=False)
    ax.set_title("Silo 1 — federated learning vs non-federated baselines")
    handles = [plt.Rectangle((0, 0), 1, 1, color=FAMILY[f][0], hatch=FAMILY[f][1],
                             ec=INK_3 if FAMILY[f][1] else FAMILY[f][0], lw=0.4) for f in FAMILY]
    ax.legend(handles, [FAMILY[f][2] for f in FAMILY], loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7.5)
    return save(fig, "fig_a_baselines_bar.png", provenance_of(*[(g, n) for _, g, n, _ in MAIN_METHODS]),
                "† budget-limited: federated val loss still decreasing at round 200.  "
                "FL checkpoints and all lr choices selected on validation loss only.")


# ── (b) convergence ────────────────────────────────────────────────────────

def fig_b() -> Path:
    series = [("FedAvg E=1 (100 rounds)", "fedavg_E1", BLUE), ("FedAvg E=5 (200 rounds)", "fedavg_E5_R200", ORANGE)]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))
    for key, ax, ylab in (("fed_val_loss", axes[0], "Federated val loss (size-weighted, class-weighted CE)"),
                          ("pooled_val_macro_f1", axes[1], "Pooled val macro-F1 (logged only)")):
        for label, name, color in series:
            hist = [m["history"] for m in runs("silo1_fl", name)]
            arr = np.array([[h[key] for h in hs] for hs in hist])[:, 1:]          # rounds 1..R
            r = np.arange(1, arr.shape[1] + 1)
            mean, sd = arr.mean(0), arr.std(0, ddof=1)
            ax.fill_between(r, mean - sd, mean + sd, color=color, alpha=0.18, linewidth=0)
            ax.plot(r, mean, color=color, linewidth=2, label=label)
            ax.text(r[-1] + 3, mean[-1], label.split(" (")[0], color=INK_2, fontsize=7.5, va="center")
        ax.set_xlabel("Communication round")
        ax.set_ylabel(ylab, fontsize=8)
        ax.set_xlim(0, 228)
    axes[0].set_ylim(0.88, 1.25)
    axes[0].set_title("Convergence of the selection signal")
    axes[1].set_title("Validation macro-F1 (not used for selection)")
    best = [m["fl"]["best_round"] for m in runs("silo1_fl", "fedavg_E5_R200")]
    axes[0].annotate(f"E=5 best round {np.mean(best):.0f}/200 → budget-limited", xy=(200, 0.905),
                     xytext=(70, 0.98), fontsize=7, color=INK_2,
                     arrowprops={"arrowstyle": "-", "color": INK_3, "lw": 0.7})
    axes[1].legend(loc="lower right", fontsize=7.5)
    fig.tight_layout()
    return save(fig, "fig_b_convergence.png", provenance_of(*[("silo1_fl", n) for _, n, _ in series]))


# ── (c) per-client scatter ─────────────────────────────────────────────────

def per_client_d4(group: str, name: str) -> dict[str, tuple[float, float]]:
    rs = runs(group, name)
    clients = rs[0]["d4"]["per_client"]
    return {c: ms([r["d4"]["per_client"][c]["macro_f1"] for r in rs]) for c in clients}


def fig_c() -> Path:
    x = per_client_d4("silo1_baselines", "local_only")
    ys = [("FedAvg E=1", per_client_d4("silo1_fl", "fedavg_E1"), BLUE, "o"),
          ("FedAvg + FT ≤2 epochs", per_client_d4("silo1_fl", "ft2_fedavg_E1"), AQUA, "D")]
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    lo, hi = 0.35, 0.78
    ax.plot([lo, hi], [lo, hi], color=INK_3, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.text(0.715, 0.672, "y = x  (no change)", ha="center", va="center", rotation=45, rotation_mode="anchor",
            fontsize=7, color=INK_3)
    for label, y, color, marker in ys:
        xs = [x[c][0] for c in x]
        yv = [y[c][0] for c in x]
        ax.scatter(xs, yv, s=46, color=color, marker=marker, edgecolor=SURFACE, linewidth=1.6, label=label, zorder=3)
    fedavg = ys[0][1]
    for c, (dx, dy) in zip(HIGHLIGHT, ((0.03, -0.055), (0.035, -0.06))):   # point at the FedAvg E=1 marker
        ax.annotate(CLIENT_SHORT[c], xy=(x[c][0], fedavg[c][0]), xytext=(x[c][0] + dx, fedavg[c][0] + dy),
                    fontsize=7.5, color=INK, arrowprops={"arrowstyle": "-", "color": INK_3, "lw": 0.7})
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("Local-only (matched SGD): per-client D4 macro-F1")
    ax.set_ylabel("Federated: per-client D4 macro-F1")
    ax.set_title("Per-client effect of federation")
    ax.legend(loc="lower right", fontsize=7.5)
    return save(fig, "fig_c_per_client_scatter.png",
                provenance_of(("silo1_baselines", "local_only"), ("silo1_fl", "fedavg_E1"), ("silo1_fl", "ft2_fedavg_E1")),
                "Each point = one event client, mean over 5 seeds. Above the diagonal = federation helps.\n"
                "D4: macro-F1 over the classes with ≥3 test examples in that client.")


# ── (d) label distribution heatmap ─────────────────────────────────────────

def fig_d() -> Path:
    d = json.loads((RESULTS / "silo1_data" / "label_distribution.json").read_text(encoding="utf-8"))
    m, git = d["metrics"], d["git"]
    clients = sorted(m["client_totals"])
    counts = np.array([m["client_totals"][c] for c in clients], dtype=float)
    share = counts / counts.sum(1, keepdims=True)
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    im = ax.imshow(share, cmap=cmap, vmin=0, vmax=0.5, aspect="auto")
    for i in range(share.shape[0]):
        for j in range(share.shape[1]):
            v = share[i, j]
            ax.text(j, i, f"{v * 100:.0f}", ha="center", va="center", fontsize=7,
                    color="#ffffff" if v > 0.28 else INK)
    ax.xaxis.tick_top()
    ax.set_xticks(range(len(m["label_short"])), [lab.replace("_", "\n") for lab in m["label_short"]], fontsize=8)
    ax.set_yticks(range(len(clients)), [f"{CLIENT_SHORT[c]}  (n={int(counts[i].sum()):,})" for i, c in enumerate(clients)])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("Share of client's tweets (%)", fontsize=8)
    cb.set_ticks([0, 0.1, 0.2, 0.3, 0.4, 0.5], labels=["0", "10", "20", "30", "40", "≥50"])
    cb.outline.set_visible(False)
    ax.set_title("Label distribution per client (% of each client's tweets) — strongly non-IID", pad=30)
    return save(fig, "fig_d_label_distribution.png",
                f"source: results/silo1_data/label_distribution.json · commit {git['commit'][:7]}"
                f"{'-dirty' if git['dirty'] else ''} · after D1/D3, all splits")


# ── (e) before vs after matched baselines (table image) ────────────────────

def fig_e() -> Path:
    fed = pooled("silo1_fl", "fedavg_E1")
    cols = {"AdamW baselines\n(strong reference)": ("local_only_adamw", "centralized_adamw"),
            "Matched-SGD baselines\n(same recipe as FL clients)": ("local_only", "centralized")}
    rows = ["Local-only", "Centralized", "FedAvg E=1 (same runs)", "Δ FedAvg − local (p)",
            "Δ FedAvg − centralized (p)", "Gap recovered by FedAvg"]
    cells = []
    for loc_name, cen_name in cols.values():
        loc, cen = pooled("silo1_baselines", loc_name), pooled("silo1_baselines", cen_name)
        f = lambda a: "{:.4f} ± {:.4f}".format(*ms(a))
        dl, dc = fed - loc, fed - cen
        gap = (fed - loc) / (cen - loc)
        pfmt = lambda p: "p<0.001" if p < 0.001 else f"p={p:.3f}"
        cells.append([f(loc), f(cen), f(fed),
                      f"{dl.mean():+.4f}  ({pfmt(stats.ttest_rel(fed, loc).pvalue)})",
                      f"{dc.mean():+.4f}  ({pfmt(stats.ttest_rel(fed, cen).pvalue)})",
                      "{:.0%} ± {:.0%}".format(*ms(gap))])
    table = np.array(cells).T
    fig, ax = plt.subplots(figsize=(7.6, 2.5))
    ax.axis("off")
    tb = ax.table(cellText=table, rowLabels=rows, colLabels=list(cols), loc="center", cellLoc="center",
                  rowLoc="left", colWidths=[0.34, 0.34])
    tb.auto_set_font_size(False)
    tb.set_fontsize(8.5)
    tb.scale(1, 1.55)
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_facecolor(SURFACE)
        if r == 0:
            cell.set_text_props(weight="bold", color=INK)
            cell.set_height(cell.get_height() * 1.45)
        if c == -1:
            cell.set_text_props(color=INK_2)
    fig.text(0.01, 0.99, "Before vs after matched baselines — pooled test macro-F1, seeds 0–4 (paired t-tests)",
             ha="left", va="top", fontsize=10, weight="bold", color=INK)
    return save(fig, "fig_e_matched_baselines.png",
                provenance_of(("silo1_fl", "fedavg_E1"), *[("silo1_baselines", n) for v in cols.values() for n in v]),
                "Only the baselines change between columns; FedAvg E=1 is the same 5 runs. Both recipes: 100 epochs,\n"
                "lr chosen on seed 0 by validation loss, val-loss checkpoints. The earlier prototype-era baselines\n"
                "(AdamW + val-F1 selection) were run from an uncommitted tree and are not shown.")


def main() -> None:
    for fn in (fig_a, fig_b, fig_c, fig_d, fig_e):
        print("wrote", fn().relative_to(REPO))


if __name__ == "__main__":
    main()
