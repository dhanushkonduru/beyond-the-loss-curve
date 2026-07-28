"""Aggregate the multi-seed runs into mean +/- std for every reported metric,
compute the behavioural consistency metric (BCM) from real probe data, and render
the data-driven paper figures at ~3 MP."""
import json, os, math, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "results")
S = os.path.join(R, "seeds")
OUT = os.path.join(HERE, "figures")   # keep the paper figures beside the manuscript
os.makedirs(OUT, exist_ok=True)

SEEDS = [42, 43, 44]
METHODS = [("Gradient Ascent", "unlearned_model"), ("Gradient Difference", "unlearned_gd"), ("NPO", "unlearned_npo")]
MIA_KEY = {"Gradient Ascent": "Gradient Ascent", "Gradient Difference": "Gradient Difference", "NPO": "NPO"}
ORACLE_MIA = "ORACLE (retrained w/o forget)"
ORACLE_REL = "Oracle (never saw data)"

INK = "#141414"
# same distinct identity as the conceptual figures: umber, slate, forest, indigo
COL = {"Gradient Ascent": "#9c6644", "Gradient Difference": "#5c677d", "NPO": "#386641", "Oracle": "#3d348b"}
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.grid": True,
                     "grid.color": "#e6e8ec", "axes.axisbelow": True})
FIGSIZE, DPI = (10, 7.5), 200


def ms(xs):
    """mean and sample standard deviation"""
    return (st.mean(xs), st.stdev(xs) if len(xs) > 1 else 0.0)


def bcm_from(matrix, P=5):
    """Behavioural consistency: 1 - 2*min(k, P-k)/P averaged over forget records."""
    vals = []
    for rec in matrix["forget"]:
        k = sum(1 for v in rec["probe_leaks"].values() if v)
        vals.append(1 - 2 * min(k, P - k) / P)
    return sum(vals) / len(vals)


# ---------------- collect ----------------
raw = {m: {"UES": [], "KRR": [], "TS": [], "BCM": [], "PLI": [], "R": []} for m, _ in METHODS}
oracle_R = []
missing = []
for s in SEEDS:
    try:
        mia = json.load(open(os.path.join(S, f"seed{s}_mia.json")))
        rel = json.load(open(os.path.join(S, f"seed{s}_relearn.json")))
    except FileNotFoundError:
        missing.append(s); continue
    orig = rel["orig_forget"]
    oc = rel["curves"].get(ORACLE_REL)
    if oc:
        oracle_R.append((oc[0] - oc[-1]) / (oc[0] - orig))
    for name, tag in METHODS:
        d = json.load(open(os.path.join(S, f"seed{s}_{tag}.json")))
        raw[name]["UES"].append(d["ForgetScore"])
        raw[name]["KRR"].append(d["UtilityScore"])
        raw[name]["TS"].append(d["Trust"])
        raw[name]["BCM"].append(bcm_from(d))
        raw[name]["PLI"].append(mia[MIA_KEY[name]] - mia[ORACLE_MIA])
        c = rel["curves"][name]
        raw[name]["R"].append((c[0] - c[-1]) / (c[0] - orig))

if missing:
    print("WARNING: missing seeds", missing)

summary = {m: {k: ms(v) for k, v in d.items() if v} for m, d in raw.items()}
summary["_oracle_recovery"] = ms(oracle_R) if oracle_R else (0, 0)
summary["_seeds"] = [s for s in SEEDS if s not in missing]
json.dump(summary, open(os.path.join(R, "seeds_summary.json"), "w"), indent=1)

# ---------------- text table ----------------
hdr = f"{'method':<22}{'UES':>14}{'KRR':>14}{'TS':>14}{'BCM':>14}{'PLI':>14}{'ARRM':>14}"
lines = ["MULTI-SEED SUMMARY  (mean +/- sd over seeds " + str(summary["_seeds"]) + ")", "=" * len(hdr), hdr, "-" * len(hdr)]
for name, _ in METHODS:
    d = summary[name]
    f = lambda k, pct=False: (f"{d[k][0]*100:.1f}+/-{d[k][1]*100:.1f}" if pct else f"{d[k][0]:.3f}+/-{d[k][1]:.3f}")
    arrm = (1 - d["R"][0], d["R"][1])
    lines.append(f"{name:<22}{f('UES'):>14}{f('KRR'):>14}{f('TS'):>14}{f('BCM'):>14}{f('PLI'):>14}"
                 f"{arrm[0]:.3f}+/-{arrm[1]:.3f}".rjust(14))
orc = summary["_oracle_recovery"]
lines += ["-" * len(hdr), f"reference model recovery R = {orc[0]*100:.1f}% +/- {orc[1]*100:.1f}%  "
          f"(ARRM = {1-orc[0]:.3f})"]
report = "\n".join(lines)
open(os.path.join(R, "seeds_summary.txt"), "w").write(report)
print(report)

# ---------------- figures ----------------
def clip01(mu, sd):
    """Asymmetric whiskers for scores defined on [0, 1], so an error bar never
    runs past a value the metric cannot take. The exact sd stays in the table."""
    lo = [m - max(0.0, m - s) for m, s in zip(mu, sd)]
    hi = [min(1.0, m + s) - m for m, s in zip(mu, sd)]
    return [lo, hi]


def bars(ax, names, means, sds, colors, fmt="{:.2f}", pad=0.015):
    b = ax.bar(names, means, yerr=sds, capsize=6, color=colors, width=0.6,
               error_kw=dict(ecolor="#444", lw=1.4))
    # sit the value above the whisker, not across it
    for rect, m, s in zip(b, means, sds):
        ax.text(rect.get_x() + rect.get_width() / 2, m + s + pad,
                fmt.format(m), ha="center", fontsize=11)
    for s in ("top", "right"): ax.spines[s].set_visible(False)


names = [m for m, _ in METHODS]
cols = [COL[n] for n in names]

# FIGURE 6: trust scores with error bars
fig, ax = plt.subplots(figsize=FIGSIZE)
w = 0.26; x = range(len(names))
for i, (k, lab, c) in enumerate([("UES", "UES (forgetting)", "#9c6644"),
                                 ("KRR", "KRR (retention)", "#5c677d"),
                                 ("TS", "TS (trustworthiness)", "#386641")]):
    mu = [summary[n][k][0] for n in names]; sd = [summary[n][k][1] for n in names]
    bb = ax.bar([p + (i - 1) * w for p in x], mu, w, yerr=clip01(mu, sd), capsize=4, label=lab, color=c,
                error_kw=dict(ecolor="#444", lw=1.2))
    for rect, m, s in zip(bb, mu, sd):
        ax.text(rect.get_x() + rect.get_width() / 2, min(1.0, m + s) + 0.02, f"{m:.2f}",
                ha="center", fontsize=9)
ax.set_xticks(list(x)); ax.set_xticklabels(names, fontsize=12)
ax.set_ylim(0, 1.28); ax.set_ylabel("score", fontsize=12)
ax.set_title("Unlearning effectiveness, knowledge retention and trustworthiness\n(mean and standard deviation over three seeds)",
             fontsize=14, fontweight="bold", color=INK)
# keep the legend clear of the tallest whisker, which now stops at 1.0
ax.legend(frameon=False, fontsize=11, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0))
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig6_trust.png"), dpi=DPI, facecolor="white"); plt.close()
print("wrote fig6_trust.png")

# FIGURE 7: trade-off scatter (UES vs KRR)
fig, ax = plt.subplots(figsize=FIGSIZE)
for n in names:
    kx, ky = summary[n]["KRR"], summary[n]["UES"]
    ax.errorbar(kx[0], ky[0],
                xerr=clip01([kx[0]], [kx[1]]), yerr=clip01([ky[0]], [ky[1]]),
                fmt="o", ms=14, color=COL[n], capsize=5, lw=1.6, label=n)
    # gradient ascent is labelled to the left so its text clears the other whiskers
    off, ha = ((-16, -30), "right") if n == "Gradient Ascent" else ((16, 12), "left")
    ax.annotate(n, (kx[0], ky[0]), textcoords="offset points", xytext=off,
                ha=ha, fontsize=12, color=COL[n], fontweight="bold")
ax.plot([0, 1], [1, 1], ls=":", color="#aaa"); ax.plot([1, 1], [0, 1], ls=":", color="#aaa")
ax.scatter([1], [1], marker="*", s=420, color="#3d348b", zorder=5)
ax.annotate("ideal", (1, 1), textcoords="offset points", xytext=(-46, -22), fontsize=12, color="#3d348b", fontweight="bold")
ax.set_xlabel("Knowledge retention ratio (utility preserved)", fontsize=12)
ax.set_ylabel("Unlearning effectiveness score (forgetting)", fontsize=12)
ax.set_xlim(-0.04, 1.10); ax.set_ylim(-0.04, 1.10)
ax.set_title("Forgetting versus utility trade-off across unlearning objectives",
             fontsize=14, fontweight="bold", color=INK)
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig7_tradeoff.png"), dpi=DPI, facecolor="white"); plt.close()
print("wrote fig7_tradeoff.png")

# FIGURE 10: recovery under relearning attack
fig, ax = plt.subplots(figsize=FIGSIZE)
labs = names + ["Reference model"]
mu = [summary[n]["R"][0] * 100 for n in names] + [orc[0] * 100]
sd = [summary[n]["R"][1] * 100 for n in names] + [orc[1] * 100]
bars(ax, labs, mu, sd, cols + [COL["Oracle"]], fmt="{:.0f}%", pad=2.0)
ax.set_ylabel("share of deleted knowledge recovered (%)", fontsize=12)
ax.set_ylim(0, 124)
ax.set_title("Knowledge recovered by a brief relearning attack\n(the reference model defines genuine absence)",
             fontsize=14, fontweight="bold", color=INK)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig10_relearn.png"), dpi=DPI, facecolor="white"); plt.close()
print("wrote fig10_relearn.png")

# FIGURE 9: privacy leakage index
fig, ax = plt.subplots(figsize=FIGSIZE)
mu = [summary[n]["PLI"][0] for n in names]; sd = [summary[n]["PLI"][1] for n in names]
b = ax.bar(names, mu, yerr=sd, capsize=6, color=cols, width=0.6, error_kw=dict(ecolor="#444", lw=1.4))
# place each value clear of its whisker, above for positive bars and below for negative ones
for rect, m, s in zip(b, mu, sd):
    ax.text(rect.get_x() + rect.get_width() / 2,
            (m + s + 0.018) if m >= 0 else (m - s - 0.045),
            f"{m:+.2f}", ha="center", fontsize=11)
ax.axhline(0, color="#444", lw=1.3)
# leave room for the value labels that sit outside the whiskers
ax.set_ylim(min(m - s for m, s in zip(mu, sd)) - 0.10,
            max(m + s for m, s in zip(mu, sd)) + 0.07)
ax.text(2.44, 0.015, "reference level", fontsize=10, color="#555", ha="right")
ax.set_ylabel("privacy leakage index  (attack AUC minus reference AUC)", fontsize=12)
ax.set_title("Residual privacy leakage relative to the retrained reference model",
             fontsize=14, fontweight="bold", color=INK)
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig9_pli.png"), dpi=DPI, facecolor="white"); plt.close()
print("wrote fig9_pli.png")
