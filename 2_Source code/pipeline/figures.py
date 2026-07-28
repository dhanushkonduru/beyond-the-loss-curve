"""
=============================================================
 FORGET-ME-NOT : Machine Unlearning Capstone
 Figure generator — thesis-quality visualizations
=============================================================

Parses results/ artifacts and produces the full figure set:

  fig_training_loss.png     memorization loss curve         (Ch 3)
  fig_methods_trajectory.png GA vs GD unlearning dynamics   (Ch 4)
  fig_tradeoff_frontier.png forgetting vs collateral damage (Ch 4)
  fig_trust_scores.png      F / U / Trust per method        (Ch 5)
  fig_leak_heatmap.png      20 countries x 5 probes leaks    (Ch 5)
  fig_mia_auc.png           membership attack AUC           (Ch 5/6)
  fig_sequential.png        damage under repeated requests  (Ch 6)

Colors follow a validated, colorblind-safe palette; series identity
is fixed across every figure (GA=blue, GD=aqua, oracle=neutral).

Run:  python figures.py     (skips figures whose inputs are missing)
"""

import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGS = os.path.join(RESULTS, "figures")
os.makedirs(FIGS, exist_ok=True)

# ---- validated palette (light surface) ----
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
BLUE = "#2a78d6"    # series: Gradient Ascent
AQUA = "#1baf7a"    # series: Gradient Difference
VIOLET = "#4a3aa7"  # series: NPO
YELLOW = "#eda100"  # series: trained model
GRAY = "#9a988f"    # reference lines / neutral

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "text.color": INK,
    "axes.edgecolor": INK2, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": INK2, "grid.alpha": 0.15,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "figure.titlesize": 13,
})


def save(fig, name):
    out = os.path.join(FIGS, name)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


def maybe(path):
    p = os.path.join(RESULTS, path)
    return p if os.path.exists(p) else None


# ---------------- parsers ----------------
def parse_epochs(path):
    pat = re.compile(r"Epoch (\d+)/\d+ done — avg loss ([\d.]+)")
    return [(int(m[0]), float(m[1])) for m in pat.findall(open(path).read())]


def parse_ga(path):
    text = open(path).read()
    base = re.search(r"Baseline(?: losses)? — forget: ([\d.]+), retain: ([\d.]+)", text)
    rows = re.findall(r"step\s+(\d+)/\d+ \| forget-loss ([\d.]+).*?retain-loss ([\d.]+)", text)
    steps = [0] + [int(r[0]) for r in rows]
    fl = [float(base[1])] + [float(r[1]) for r in rows]
    rl = [float(base[2])] + [float(r[2]) for r in rows]
    return steps, fl, rl


def oracle_losses():
    p = maybe("comparison.txt")
    if not p:
        return None
    m = re.search(r"ORACLE.*?\s([\d.]+)\s+([\d.]+)", open(p).read())
    return (float(m[1]), float(m[2])) if m else None


# ---------------- figures ----------------
def fig_training_loss():
    p = maybe("train_log.txt")
    if not p:
        return
    xs, ys = zip(*parse_epochs(p))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.axhline(10.825, ls=":", lw=1.4, color=GRAY)
    ax.annotate("random init ≈ ln(50257)", xy=(xs[-1], 10.825), ha="right",
                va="bottom", fontsize=9, color=INK2)
    ax.plot(xs, ys, lw=2, color=BLUE, marker="o", ms=4)
    ax.annotate(f"{ys[-1]:.2f}", xy=(xs[-1], ys[-1]), xytext=(4, 4),
                textcoords="offset points", fontsize=9, color=INK)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average training loss")
    ax.set_title("Training on WHO data: the model memorizes 100 countries")
    save(fig, "fig_training_loss.png")


def fig_methods_trajectory():
    ga, gd = maybe("ga_log.txt"), maybe("gd_log.txt")
    if not (ga and gd):
        return
    orc = oracle_losses()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=False)
    for ax, idx, title in [(axes[0], 1, "Forget-loss (higher = more forgotten)"),
                           (axes[1], 2, "Retain-loss (flat = utility intact)")]:
        for path, color, label in [(ga, BLUE, "Gradient Ascent"),
                                   (gd, AQUA, "Gradient Difference")]:
            data = parse_ga(path)
            ax.plot(data[0], data[idx], lw=2, color=color, label=label)
        if orc:
            ref = orc[0] if idx == 1 else orc[1]
            ax.axhline(ref, ls=":", lw=1.4, color=GRAY)
            ax.annotate("oracle", xy=(ax.get_xlim()[1], ref), ha="right",
                        va="bottom", fontsize=9, color=INK2)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Unlearning step")
    axes[0].set_ylabel("Loss")
    axes[0].legend(frameon=False, loc="lower right")
    fig.suptitle("Two unlearning methods, same budget: the retain anchor changes everything")
    save(fig, "fig_methods_trajectory.png")


def fig_tradeoff_frontier():
    ga, gd = maybe("ga_log.txt"), maybe("gd_log.txt")
    if not (ga and gd):
        return
    fig, ax = plt.subplots(figsize=(6.5, 5))
    top = 0
    for path, color, label in [(ga, BLUE, "Gradient Ascent"),
                               (gd, AQUA, "Gradient Difference")]:
        steps, fl, rl = parse_ga(path)
        drift = [r - rl[0] for r in rl]
        rise = [f - fl[0] for f in fl]
        top = max(top, max(rise))
        ax.plot(drift, rise, lw=2, color=color, marker="o", ms=3.5, label=label)
        ax.annotate(label, xy=(drift[-1], rise[-1]), xytext=(6, 0),
                    textcoords="offset points", fontsize=9, color=color)
    ax.plot([0, top], [0, top], ls=":", lw=1.4, color=GRAY)
    ax.annotate("no selectivity (1:1 damage)", xy=(top * 0.55, top * 0.5),
                rotation=38, fontsize=9, color=INK2)
    ax.set_xlabel("Retain-loss drift (collateral damage) →")
    ax.set_ylabel("Forget-loss rise (forgetting) →")
    ax.set_title("Forgetting–utility frontier: up and LEFT is better")
    ax.legend(frameon=False, loc="upper left")
    save(fig, "fig_tradeoff_frontier.png")


def load_matrix(tag):
    p = maybe(f"audit_matrix_{tag}.json")
    return json.load(open(p)) if p else None


def _methods():
    out = []
    for tag, color, label in [("unlearned_model", BLUE, "Gradient Ascent"),
                              ("unlearned_gd", AQUA, "Gradient Difference"),
                              ("unlearned_npo", VIOLET, "NPO")]:
        m = load_matrix(tag)
        if m:
            out.append((m, color, label))
    return out


def fig_trust_scores():
    methods = _methods()
    if len(methods) < 2:
        return
    metrics = ["ForgetScore", "UtilityScore", "Trust"]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    x = range(len(metrics))
    n = len(methods)
    w = 0.8 / n
    for k, (data, color, label) in enumerate(methods):
        off = (k - (n - 1) / 2) * w
        vals = [data[m] for m in metrics]
        bars = ax.bar([i + off for i in x], vals, width=w - 0.03, color=color,
                      label=label, zorder=3)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.2f}", xy=(b.get_x() + b.get_width() / 2, v),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(list(x), ["ForgetScore F", "UtilityScore U", "Trust T = 2FU/(F+U)"])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score (1.0 = ideal)")
    ax.set_title("Audit scorecard: every method, identical interrogation")
    ax.legend(frameon=False, loc="upper left", ncol=len(methods))
    save(fig, "fig_trust_scores.png")


def fig_leak_heatmap():
    ga = load_matrix("unlearned_model")
    gd = load_matrix("unlearned_gd")
    if not (ga and gd):
        return
    probes = ["Direct", "Indirect", "Fill-in-blank", "Yes/No", "Rephrased"]
    cmap = ListedColormap(["#e8e7e2", BLUE])  # clean vs leaking
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 6.5), sharey=True)
    for ax, data, title in [(axes[0], ga, "Gradient Ascent"),
                            (axes[1], gd, "Gradient Difference")]:
        rows = data["forget"]
        grid = [[1 if r["probe_leaks"][p] else 0 for p in probes] for r in rows]
        ax.imshow(grid, cmap=cmap, aspect="auto", vmin=0, vmax=1)
        n_leak = sum(sum(g) for g in grid)
        ax.set_title(f"{title}\n{n_leak}/100 probes still leak", fontsize=10)
        ax.set_xticks(range(5), probes, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(rows)), [r["name"] for r in rows], fontsize=7.5)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.legend(handles=[Patch(color=BLUE, label="knowledge leaks"),
                        Patch(color="#e8e7e2", label="clean (no leak)")],
               loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Per-country leak map: every forgotten country × every probe")
    fig.subplots_adjust(bottom=0.16)
    fig.savefig(os.path.join(FIGS, "fig_leak_heatmap.png"), dpi=150)
    plt.close(fig)
    print("wrote", os.path.join(FIGS, "fig_leak_heatmap.png"))


def fig_mia_auc():
    p = maybe("mia.json")
    if not p:
        return
    data = json.load(open(p))
    order = [("trained (before unlearning)", YELLOW, "Trained"),
             ("Gradient Ascent", BLUE, "Gradient Ascent"),
             ("Gradient Difference", AQUA, "Gradient Difference"),
             ("NPO", VIOLET, "NPO"),
             ("ORACLE (retrained w/o forget)", GRAY, "Oracle")]
    rows = [(lbl, data[k], c) for k, c, lbl in order if k in data]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar([r[0] for r in rows], [r[1] for r in rows],
                  color=[r[2] for r in rows], width=0.6, zorder=3)
    for b, (_, v, _) in zip(bars, rows):
        ax.annotate(f"{v:.3f}", xy=(b.get_x() + b.get_width() / 2, v),
                    xytext=(0, 3), textcoords="offset points", ha="center",
                    fontsize=9, color=INK)
    ax.axhline(0.5, ls=":", lw=1.4, color=GRAY)
    ax.annotate("coin flip (0.5)", xy=(len(rows) - 0.45, 0.5), ha="right",
                va="bottom", fontsize=9, color=INK2)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Membership attack AUC")
    ax.set_title("Privacy: can an attacker still detect the deleted data?")
    save(fig, "fig_mia_auc.png")


def fig_sequential():
    p = maybe("sequential_report.txt")
    if not p:
        return
    rows = re.findall(r"^\s+(\d)\s+\S+\s+[\d.]+\s+[\d.]+\s+\+?(-?[\d.]+)\s*$",
                      open(p).read(), re.M)
    if not rows:
        return
    rounds = [int(r[0]) for r in rows]
    drift = [float(r[1]) for r in rows]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    bars = ax.bar([str(r) for r in rounds], drift, color=BLUE, width=0.55, zorder=3)
    for b, v in zip(bars, drift):
        ax.annotate(f"+{v:.2f}", xy=(b.get_x() + b.get_width() / 2, v),
                    xytext=(0, 3), textcoords="offset points", ha="center",
                    fontsize=9, color=INK)
    ax.set_xlabel("Deletion request round (5 countries each)")
    ax.set_ylabel("Retain-loss drift (collateral damage)")
    ax.set_title("Sequential requests: damage accelerates round over round")
    save(fig, "fig_sequential.png")


def fig_relearn():
    p = maybe("relearn.json")
    if not p:
        return
    d = json.load(open(p))
    step = d["steps_per_point"]
    colors = {"Gradient Ascent": BLUE, "Gradient Difference": AQUA, "NPO": VIOLET,
              "Oracle (never saw data)": GRAY}
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for name, curve in d["curves"].items():
        xs = [i * step for i in range(len(curve))]
        ax.plot(xs, curve, lw=2, marker="o", ms=3.5,
                color=colors.get(name, INK2), label=name)
    ax.axhline(d["orig_forget"], ls=":", lw=1.4, color=GRAY)
    ax.annotate("original memorized level", xy=(ax.get_xlim()[1], d["orig_forget"]),
                ha="right", va="bottom", fontsize=9, color=INK2)
    ax.set_xlabel("Relearning steps (attacker fine-tunes on a few records)")
    ax.set_ylabel("Forget-loss (lower = knowledge back)")
    ax.set_title("Relearning attack: does the 'forgotten' data snap back?")
    ax.legend(frameon=False, loc="upper right")
    save(fig, "fig_relearn.png")


def fig_sweep():
    p = maybe("sweep.json")
    if not p:
        return
    d = json.load(open(p))
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for method, color, label in [("GA", BLUE, "Gradient Ascent"),
                                 ("GD", AQUA, "Gradient Difference")]:
        pts = sorted(d[method], key=lambda z: z["retain_drift"])
        xs = [z["retain_drift"] for z in pts]
        ys = [z["forget_rise"] for z in pts]
        ax.plot(xs, ys, lw=2, marker="o", ms=5, color=color, label=label)
    ax.set_xlabel("Retain-loss drift (collateral damage) →")
    ax.set_ylabel("Forget-loss rise (forgetting) →")
    ax.set_title("Operating-point sweep: GD's curve sits above GA's at every budget")
    ax.legend(frameon=False, loc="upper left")
    save(fig, "fig_sweep.png")


def fig_sync():
    p = maybe("sync.json")
    if not p:
        return
    d = json.load(open(p))
    sets = [("added", "ADDED\n(learn new)", AQUA, "loss should fall ↓"),
            ("removed", "REMOVED\n(unlearn)", BLUE, "loss should rise ↑"),
            ("kept", "KEPT\n(retain)", GRAY, "loss should stay low")]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    x = range(len(sets))
    w = 0.38
    for off, when, alpha, lbl in [(-w / 2, "loss_before", 0.45, "before sync"),
                                  (w / 2, "loss_after", 1.0, "after sync")]:
        for i, (key, _, color, _) in enumerate(sets):
            v = d[when][key]
            ax.bar(i + off, v, width=w - 0.03, color=color, alpha=alpha, zorder=3,
                   label=lbl if i == 0 else None)
            ax.annotate(f"{v:.2f}", xy=(i + off, v), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=9, color=INK)
    ax.set_xticks(list(x), [s[1] for s in sets])
    for i, (_, _, _, note) in enumerate(sets):
        ax.annotate(note, xy=(i, -0.5), ha="center", fontsize=8.5, color=INK2,
                    annotation_clip=False)
    ax.set_ylabel("Answer-loss (low = model knows the fact)")
    ax.set_title("Dataset-model sync on real WHO data — no retraining")
    ax.legend(frameon=False, loc="upper center", ncol=2)
    ax.set_ylim(0, max(max(d["loss_before"].values()), max(d["loss_after"].values())) * 1.18)
    save(fig, "fig_sync.png")


if __name__ == "__main__":
    for f in (fig_training_loss, fig_methods_trajectory, fig_tradeoff_frontier,
              fig_trust_scores, fig_leak_heatmap, fig_mia_auc, fig_sequential,
              fig_relearn, fig_sweep, fig_sync):
        try:
            f()
        except Exception as e:
            print(f"skip {f.__name__}: {e}")
    print("✅ figures in", FIGS)
