"""Summarise a keyframe-noise sweep CSV into the tables that go in the writeup.

    .venv-rocm/bin/python experiments/keyframe-noise/analyze.py results/full-*.csv
"""

import sys
import json
import pathlib

import numpy as np
import pandas as pd

METRICS = [
    ("kf_err_to_perturbed", 1000, "kf_err_pert (mm)", "%7.1f"),
    ("kf_err_to_clean", 1000, "kf_err_clean (mm)", "%7.1f"),
    ("skate_ratio", 1, "foot skate", "%7.4f"),
    ("jerk", 1, "jerk (m/s^3)", "%7.0f"),
    ("ground_pen", 1000, "gnd pen (mm)", "%7.1f"),
    ("div_from_ref", 1000, "divergence (mm)", "%7.1f"),
]


def pivot(df, col, scale, fmt):
    thetas = sorted(df.theta.unique())
    tols = sorted(df.tolerance.unique(), reverse=True)
    head = "  c \\ theta |" + "".join(f"{t:>9d}" for t in thetas)
    lines = [head, "  " + "-" * (len(head) - 2)]
    for c in tols:
        cells = []
        for th in thetas:
            v = df[(df.tolerance == c) & (df.theta == th)][col]
            cells.append(fmt % (v.mean() * scale) if len(v) else "      --")
        lines.append(f"  {c:>9.2f} |" + "".join(f"{x:>9}" for x in cells))
    return "\n".join(lines)


def load(paths):
    """Concatenate one or more sweep CSVs. Older files predate the seed column."""
    frames = []
    for i, path in enumerate(paths):
        d = pd.read_csv(path)
        if "seed" not in d.columns:
            meta = pathlib.Path(path).with_suffix(".json")
            d["seed"] = json.loads(meta.read_text())["seed"] if meta.exists() else i
        if "mode" not in d.columns:
            d["mode"] = "independent"  # the only mode before the coherent control
        d["source"] = pathlib.Path(path).name
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def mode_contrast(df):
    """Independent vs coherent keyframe error, paired on (seed, sample).

    Both modes inject exactly theta degrees per joint. They differ only in
    whether consecutive keyframes are wrong in the same direction. If the damage
    is about pose error, they should agree; if it is about keyframes disagreeing
    with each other, only 'independent' should hurt.
    """
    modes = sorted(df["mode"].unique())
    if len(modes) < 2:
        return
    seeds = sorted(set.intersection(*(set(df[df["mode"] == m].seed.unique()) for m in modes)))
    d = df[df.seed.isin(seeds)]
    print("\n== independent vs coherent error " + "=" * 30)
    print(f"   seeds {seeds}; both inject exactly theta deg/joint\n")
    for c in sorted(d.tolerance.unique(), reverse=True):
        sub = d[d.tolerance == c]
        if sub["mode"].nunique() < 2:
            continue
        print(f"  c = {c:.2f}")
        for col, scale, label, _f in METRICS[2:4]:
            for m in modes:
                vals = [sub[(sub["mode"] == m) & (sub.theta == th)][col].mean() * scale
                        for th in sorted(sub.theta.unique())]
                print(f"    {label:<14} {m:<12} " + " ".join(f"{v:8.4g}" for v in vals))
        print()


def main(paths):
    df = load(paths)
    n = df.groupby(["theta", "tolerance"]).size().min()
    print(", ".join(pathlib.Path(p).name for p in paths))
    print(f"{len(df)} rows, {n} motions per cell, "
          f"{df.theta.nunique()} thetas x {df.tolerance.nunique()} tolerances, "
          f"seeds {sorted(df.seed.unique())}\n")

    for col, scale, label, fmt in METRICS:
        print(f"== {label} " + "=" * (58 - len(label)))
        print(pivot(df, col, scale, fmt))
        print()

    # headline contrasts
    def cell(th, c, col):
        return df[(df.theta == th) & (df.tolerance == c)][col]

    thetas = sorted(df.theta.unique())
    tols = sorted(df.tolerance.unique(), reverse=True)
    hard, soft = max(tols), min(tols)
    t0, tmax = min(thetas), max(thetas)

    print("== headline " + "=" * 52)

    def pct(a, b):
        a, b = a.mean(), b.mean()
        return f"{a:.4g} -> {b:.4g} ({(b - a) / a * 100:+.0f}%)"

    print(f"  hard imputation cost at theta=0 (c={hard} vs c={soft}):")
    for col, _s, label, _f in METRICS[2:4]:
        print(f"    {label:<18} {pct(cell(t0, soft, col), cell(t0, hard, col))}")

    print(f"\n  angular noise cost at c={hard} (theta={t0} -> {tmax}):")
    for col, _s, label, _f in METRICS[2:4]:
        print(f"    {label:<18} {pct(cell(t0, hard, col), cell(tmax, hard, col))}")

    print(f"\n  tolerance recovery at theta={tmax} (c={hard} -> c={soft}):")
    for col, _s, label, _f in METRICS[2:4]:
        print(f"    {label:<18} {pct(cell(tmax, hard, col), cell(tmax, soft, col))}")
    print(f"    {'kf fidelity lost':<18} "
          f"{cell(tmax, hard, 'kf_err_to_perturbed').mean() * 1000:.1f} -> "
          f"{cell(tmax, soft, 'kf_err_to_perturbed').mean() * 1000:.1f} mm")

    # does the model pull noisy keyframes back toward the clean pose?
    print("\n  correction test -- output-to-clean error vs the noise injected:")
    for th in thetas:
        if th == 0:
            continue
        injected = cell(th, hard, "kf_err_to_clean").mean() * 1000
        for c in tols:
            got = cell(th, c, "kf_err_to_clean").mean() * 1000
            if c == hard:
                base = got
        best = min(cell(th, c, "kf_err_to_clean").mean() * 1000 for c in tols)
        print(f"    theta={th:2d}: injected {base:6.1f} mm, best over c = {best:6.1f} mm "
              f"({(best - base) / base * 100:+.0f}%)")

    # per-sample spread, so the reader knows what is signal
    print("\n  spread (std over motions) at c=%.2f:" % hard)
    for col, scale, label, _f in METRICS[2:4]:
        s = [cell(th, hard, col).std() * scale for th in thetas]
        print(f"    {label:<18} " + " ".join(f"{v:8.4g}" for v in s))

    # Every cell shares prompts and diffusion noise, so differences are paired.
    # Unpaired means drown the tolerance effect in between-motion variance;
    # the paired delta is the honest test of whether it is real.
    print("\n== paired differences (same motion, same noise) " + "=" * 16)
    print("   mean delta +/- SE over motions; |t| > 2 is the bar\n")

    def paired(col, scale, a, b, key):
        """b - a, matched on (seed, sample)."""
        ka = df[key(a)].set_index(["seed", "sample"])[col]
        kb = df[key(b)].set_index(["seed", "sample"])[col]
        idx = ka.index.intersection(kb.index)
        d = (kb.loc[idx] - ka.loc[idx]) * scale
        se = d.std(ddof=1) / np.sqrt(len(d))
        return d.mean(), se, (d.mean() / se if se > 0 else np.inf), len(d)

    for label_txt, a, b, key in [
        (f"hard imputation, theta=0 (c={soft} -> c={hard})", soft, hard,
         lambda c: (df.theta == t0) & (df.tolerance == c)),
        (f"hard imputation, theta={tmax} (c={soft} -> c={hard})", soft, hard,
         lambda c: (df.theta == tmax) & (df.tolerance == c)),
        (f"angular noise at c={hard} (theta=0 -> {tmax})", t0, tmax,
         lambda th: (df.theta == th) & (df.tolerance == hard)),
        (f"angular noise at c={soft} (theta=0 -> {tmax})", t0, tmax,
         lambda th: (df.theta == th) & (df.tolerance == soft)),
    ]:
        print(f"  {label_txt}")
        for col, scale, label, _f in METRICS[2:4]:
            m, se, t, k = paired(col, scale, a, b, key)
            print(f"    {label:<18} {m:+9.4g} +/- {se:.4g}  |t|={abs(t):5.1f}  n={k}")
        print()

    mode_contrast(df)


if __name__ == "__main__":
    main(sys.argv[1:])
