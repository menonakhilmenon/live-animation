"""Summarise the density and partial-pose sweeps.

    .venv-rocm/bin/python experiments/keyframe-noise/analyze_scope.py \
        experiments/keyframe-noise/results/dens-*.csv
    .venv-rocm/bin/python experiments/keyframe-noise/analyze_scope.py \
        experiments/keyframe-noise/results/part-*.csv

Both sweeps answer the same question -- how much work does an upstream keyframe
proposer actually have to do -- from two directions: how *many* keyframes, and
how many *joints* per keyframe. So both get the same treatment: absolute level,
and the degradation slope from theta=0, which is the part that a better proposer
can buy back.
"""

import sys
import pathlib

import numpy as np
import pandas as pd

FPS = 20.0
FRAMES = 196

METRICS = [
    ("jerk", 1, "jerk (m/s^3)", "%8.0f"),
    ("skate_ratio", 1, "foot skate", "%8.4f"),
    ("div_from_ref", 1000, "divergence (mm)", "%8.1f"),
    ("kf_err_obs", 1000, "kf err, observed (mm)", "%8.1f"),
]


def load(paths):
    frames = []
    for path in paths:
        d = pd.read_csv(path)
        d["source"] = pathlib.Path(path).name
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    for col, default in [("trans_length", 10), ("obs_joints", "all"),
                         ("perturb_joints", "all"), ("mode", "independent")]:
        if col not in df.columns:
            df[col] = default
    df["keys_per_s"] = np.ceil(FRAMES / df.trans_length) / (FRAMES / FPS)
    return df


def paired(df, col, scale, sel_a, sel_b):
    """mean(b - a) +/- SE, matched on (seed, sample). Same prompts, same noise."""
    a = df[sel_a].set_index(["seed", "sample"])[col]
    b = df[sel_b].set_index(["seed", "sample"])[col]
    idx = a.index.intersection(b.index)
    if not len(idx):
        return None
    d = (b.loc[idx] - a.loc[idx]) * scale
    se = d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), se, (d.mean() / se if se > 0 else np.inf), len(d)


def table(df, row_key, row_label, fmt_row):
    """One block per metric per tolerance: rows = configuration, cols = theta."""
    thetas = sorted(df.theta.unique())
    for c in sorted(df.tolerance.unique(), reverse=True):
        print(f"\n--- c = {c:.2f} " + "-" * 52)
        for col, scale, label, fmt in METRICS:
            head = f"  {label:<22}" + "".join(f"{'th=%d' % t:>9}" for t in thetas)
            head += f"{'th0->max':>11}"
            print(head)
            for rv in sorted(df[row_key].unique(), key=lambda v: (str(type(v)), v)):
                sub = df[(df[row_key] == rv) & (df.tolerance == c)]
                if sub.empty:
                    continue
                vals = [sub[sub.theta == t][col].mean() * scale for t in thetas]
                base, top = vals[0], vals[-1]
                # A percentage off a base that rounds to zero (kf error under
                # hard imputation) is noise amplified to look like a result.
                rel = (f"{(top - base) / base * 100:+9.0f}%"
                       if base > 0.05 else "        --")
                cells = "".join(fmt % v for v in vals)
                print(f"    {fmt_row(rv):<20}" + cells + rel)
            print()


def density(df):
    print("\n" + "=" * 74)
    print("A. KEYFRAME DENSITY -- how many keyframes must the proposer emit?")
    print("=" * 74)
    n = df.groupby(["trans_length", "theta", "tolerance"]).size().min()
    print(f"n = {n} motions per cell; {FRAMES} frames at {FPS:g} fps")
    print("  trans_length 5/10/20/40 = 40/20/10/5 keyframes = "
          "4.08/2.04/1.02/0.51 keys/s")

    table(df, "trans_length", "keys",
          lambda tl: f"tl={tl:<3d} ({np.ceil(FRAMES / tl) / (FRAMES / FPS):.2f}/s)")

    # The question is not "is sparse worse" -- it obviously is, the model has
    # less information. It is whether sparse specification *amplifies* a given
    # per-joint error. Compare the theta penalty, not the level.
    print("\n  theta=0 -> 30 penalty, paired within each density:")
    for c in sorted(df.tolerance.unique(), reverse=True):
        print(f"    c = {c:.2f}")
        for tl in sorted(df.trans_length.unique()):
            m = df.trans_length == tl
            out = []
            for col, scale, label, _f in METRICS[:3]:
                r = paired(df, col, scale,
                           m & (df.theta == 0) & (df.tolerance == c),
                           m & (df.theta == df.theta.max()) & (df.tolerance == c))
                if r:
                    out.append(f"{label.split()[0]} {r[0]:+8.4g} |t|={abs(r[2]):4.1f}")
            print(f"      tl={tl:<3d} " + "   ".join(out))
        print()


def partial(df):
    print("\n" + "=" * 74)
    print("B. PARTIAL POSE -- does error stay in the body part it was injected into?")
    print("=" * 74)
    df = df.copy()
    df["config"] = df.obs_joints + "/" + df.perturb_joints
    n = df.groupby(["config", "theta", "tolerance"]).size().min()
    print(f"n = {n} motions per cell; config is observed/perturbed "
          f"(the root is always observed)")

    table(df, "config", "config", lambda s: s)

    # Locality: divergence at joints the perturbation actually moves, vs joints
    # it cannot reach through the kinematic tree. If the second one tracks the
    # first, the densifier is spreading the error body-wide.
    print("\n  locality -- divergence (mm) at moved vs untouched joints:")
    thetas = sorted(df.theta.unique())
    for c in sorted(df.tolerance.unique(), reverse=True):
        print(f"    c = {c:.2f}")
        for cfg in sorted(df.config.unique()):
            sub = df[(df.config == cfg) & (df.tolerance == c)]
            if sub.empty or sub.div_spared.isna().all():
                continue
            hit = [sub[sub.theta == t].div_hit.mean() * 1000 for t in thetas]
            spa = [sub[sub.theta == t].div_spared.mean() * 1000 for t in thetas]
            g = paired(df, "div_spared", 1000,
                       (df.config == cfg) & (df.theta == 0) & (df.tolerance == c),
                       (df.config == cfg) & (df.theta == max(thetas)) & (df.tolerance == c))
            h = paired(df, "div_hit", 1000,
                       (df.config == cfg) & (df.theta == 0) & (df.tolerance == c),
                       (df.config == cfg) & (df.theta == max(thetas)) & (df.tolerance == c))
            print(f"      {cfg:<12} moved   " + "".join(f"{v:9.1f}" for v in hit)
                  + (f"   d={h[0]:+7.1f} |t|={abs(h[2]):4.1f}" if h else ""))
            print(f"      {'':<12} spared  " + "".join(f"{v:9.1f}" for v in spa)
                  + (f"   d={g[0]:+7.1f} |t|={abs(g[2]):4.1f}" if g else ""))
        print()

    # Partial specification: is leaving joints free cheaper than specifying them
    # badly? Compare obs=all vs obs=<part> at the same perturbation.
    print("  cost of leaving joints unspecified (same perturbation, "
          "fewer joints observed):")
    for c in sorted(df.tolerance.unique(), reverse=True):
        for pj in sorted(df.perturb_joints.unique()):
            cfgs = sorted(df[df.perturb_joints == pj].obs_joints.unique())
            if len(cfgs) < 2 or "all" not in cfgs:
                continue
            other = [o for o in cfgs if o != "all"]
            for o in other:
                for th in thetas:
                    sel = lambda ob: ((df.obs_joints == ob) & (df.perturb_joints == pj)
                                      & (df.theta == th) & (df.tolerance == c))
                    out = []
                    for col, scale, label, _f in METRICS[:3]:
                        r = paired(df, col, scale, sel("all"), sel(o))
                        if r:
                            out.append(f"{label.split()[0]} {r[0]:+8.4g} |t|={abs(r[2]):4.1f}")
                    if out:
                        print(f"    c={c:.2f} perturb={pj:<5} obs all->{o:<6} "
                              f"th={th:<3d} " + "  ".join(out))
        print()


def main(paths):
    df = load(paths)
    print(", ".join(sorted({pathlib.Path(p).name for p in paths})))
    print(f"{len(df)} rows, thetas {sorted(df.theta.unique())}, "
          f"tolerances {sorted(df.tolerance.unique())}")
    if df.trans_length.nunique() > 1:
        density(df)
    if df.obs_joints.nunique() > 1 or df.perturb_joints.nunique() > 1:
        partial(df)


if __name__ == "__main__":
    main(sys.argv[1:])
