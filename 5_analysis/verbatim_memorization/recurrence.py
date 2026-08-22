#!/usr/bin/env python3
"""
% of memorized regions recurring across multiple patients (K_corpus > 1).

% of clinically revealing, patient-specific regions (K_corpus == 1) that
recur across multiple notes of the same patient (K_within_patient > 1).

Plot: breakdown of patient-timeline duplication for clinically revealing, patient-specific regions (Fig 12)
"""

import argparse
import json
import re
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "1_priors"))
import build_prompt_from_priors as _priors  # noqa: E402

# Paths and constants  (kept identical to make_paper_figures.py)

RESULTS_DIR = config.RESULTS_DIR
FIGURE_DIR  = config.FIGURES_DIR / "verbatim_memorization"

PRIOR_ORDER = list(_priors.PRIOR_TIERS)
PRIORS_INCLUDED = set(PRIOR_ORDER)
PROBES_INCLUDED = set(_priors.PROBE_TYPES)

FNAME_RE      = re.compile(r"^(?P<prior>[a-z_]+)__(?P<probe>[a-z_]+)\.jsonl$")


# Styling / layout helpers (mirrors make_paper_figures.py so figures match)

def set_paper_style(font_scale=1.5):
    sns.set_context("paper", font_scale=font_scale)
    base = 12 * font_scale
    plt.rcParams.update({
        "axes.titlesize":  base + 1,
        "axes.labelsize":  base,
        "xtick.labelsize": base - 1,
        "ytick.labelsize": base - 1,
        "legend.fontsize": base - 1,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "savefig.dpi":       200,
        "figure.dpi":        110,
    })


def finalize(fig, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"  saved {out_path}")


def out_dir_for_ckpt(ckpt_tag):
    out = FIGURE_DIR / ckpt_tag / "paper"
    out.mkdir(parents=True, exist_ok=True)
    return out


def percent_yaxis(ax, decimals=0):
    ax.yaxis.set_major_formatter(
        mticker.PercentFormatter(xmax=1.0, decimals=decimals))
    ax.set_ylim(bottom=0)


# Eval-file discovery (copied from make_paper_figures.py for self-containment)

def _eval_files(ckpt_tag, priors=PRIORS_INCLUDED, probes=PROBES_INCLUDED):
    root = RESULTS_DIR / ckpt_tag
    if not root.exists():
        return []
    out = []
    for fp in sorted(root.rglob("*.jsonl")):
        m_fname = FNAME_RE.match(fp.name)
        if not m_fname:
            continue
        prior, probe = m_fname.group("prior"), m_fname.group("probe")
        if prior in priors and probe in probes:
            out.append(fp)
    return out


def _iter_train_records(fp):
    with open(fp) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (rec.get("eval_split") or "unknown") != "train":
                continue
            if not rec.get("patient_id"):
                continue
            yield rec


# Worker: scan one file for cross-patient and within-patient recurrence

def _scan_one(fp_str):
    """For every region with a valid K_corpus:

      (1) count it toward cross-patient recurrence (K_corpus > 1).
      (2) if it is clinically revealing and patient-specific (K_corpus == 1),
          credit it once (region-weighted) to kw_rev_k1_regions, keyed by
          K_within_patient.
    """
    fp = Path(fp_str)

    kw_rev_k1_regions = Counter()   # K_within_patient -> region count
    n_train = n_regions = n_segments = 0
    n_regions_no_kcorpus = 0
    n_regions_cross_patient_recur = 0   # K_corpus > 1

    for rec in _iter_train_records(fp):
        n_train += 1
        attr = rec.get("source_attribution")
        if not isinstance(attr, dict):
            continue
        for r in (attr.get("regions") or []):
            n_regions += 1
            k_corpus = r.get("K_corpus")
            if not k_corpus:                      # None or 0 -> can't bucket
                n_regions_no_kcorpus += 1
                continue
            if k_corpus > 1:
                n_regions_cross_patient_recur += 1
            is_k1 = (k_corpus == 1)
            k_within = r.get("K_within_patient")
            region_counted = False                # credit a region once

            for seg in (r.get("segments") or []):
                if "n_revealing" not in seg:
                    continue
                n_segments += 1
                n_rev = seg.get("n_revealing", 0)

                # Region-weighted: credit this region exactly once if it is
                # K_corpus==1, has any revealing content, and has a within-
                # patient count. A region is counted on the first revealing
                # segment encountered.
                if (is_k1 and n_rev > 0 and k_within is not None
                        and not region_counted):
                    kw_rev_k1_regions[int(k_within)] += 1
                    region_counted = True

    return {
        "kw_rev_k1_regions":              dict(kw_rev_k1_regions),
        "n_train":                        n_train,
        "n_regions":                      n_regions,
        "n_segments":                     n_segments,
        "n_regions_no_kcorpus":           n_regions_no_kcorpus,
        "n_regions_cross_patient_recur":  n_regions_cross_patient_recur,
        "file":                           str(fp),
    }


# (1) Cross-patient recurrence

def report_cross_patient_recurrence(results):
    n_with_kcorpus = sum(r["n_regions"] - r["n_regions_no_kcorpus"] for r in results)
    n_recur = sum(r["n_regions_cross_patient_recur"] for r in results)
    pct = 100 * n_recur / n_with_kcorpus if n_with_kcorpus else 0.0
    print(f"[cross_patient_recurrence] {pct:.1f}% of memorized regions recur "
          f"across multiple patients (K_corpus > 1); n={n_with_kcorpus:,}")
    return pct


# (2) & (3) Within-patient recurrence: aggregate stat + histogram plot

def _pooled_fraction(counter, xmax):
    """counter: {K_within_patient: weight}. Returns xs, ys (fractions), total."""
    by_k = Counter()
    total = 0
    for k, w in counter.items():
        if k is None or k <= 0:
            continue
        by_k[min(k, xmax)] += w
        total += w
    xs = list(range(1, xmax + 1))
    ys = [by_k.get(k, 0) / total if total else 0.0 for k in xs]
    return xs, ys, total


def report_within_patient_recurrence(kw_regions):
    total = sum(v for k, v in kw_regions.items() if k and k > 0)
    n_multi = sum(v for k, v in kw_regions.items() if k and k > 1)
    pct = 100 * n_multi / total if total else 0.0
    print(f"[within_patient_recurrence] {pct:.1f}% of clinically-revealing, "
          f"patient-specific (K_corpus==1) regions recur across multiple "
          f"notes of the same patient (K_within_patient > 1); n={total:,}")
    return pct


def plot_within_patient_revealing_k1(kw_regions, ckpt_tag, xmax=10):
    if not kw_regions:
        print("[within_patient_recurrence] no revealing K=1 regions found, skipping plot")
        return

    xs, ys, total_reg = _pooled_fraction(kw_regions, xmax)
    bar_color = sns.color_palette("colorblind")[3]   # vermillion = revealing

    set_paper_style()
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(xs, ys, color=bar_color, edgecolor="black",
           linewidth=0.5, width=0.85)
    for x, y in zip(xs, ys):
        if y >= 0.005:
            ax.text(x, y + 0.005, f"{100*y:.1f}%",
                    ha="center", va="bottom", fontsize=12)

    ax.set_xticks(xs)
    ax.set_xticklabels([str(k) for k in xs[:-1]] + [f"≥{xmax}"])
    ax.set_xlim(0.5, xmax + 0.5)
    ax.set_xlabel("Number of patient notes containing memorized region")
    ax.set_ylabel("Fraction of memorized regions with clinically-revealing text")
    percent_yaxis(ax)

    finalize(fig, out_dir_for_ckpt(ckpt_tag) / "within_patient_revealing_k1.png")

    print(f"[within_patient_recurrence] region-weighted total = {total_reg:,} revealing K=1 regions")
    print(f"[within_patient_recurrence] region-weighted fraction by K_within_patient "
          f"(1..>= {xmax}):")
    for k, y in zip(xs, ys):
        label = f"{k}" if k < xmax else f">={xmax}"
        print(f"          K_within={label:>4}: {100*y:5.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-tag", dest="ckpt_tag", default=config.DEFAULT_CHECKPOINT,
                    help="Results subdirectory name, e.g. 'final' or 'checkpoint-14178'")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--xmax", type=int, default=10,
                    help="Right-side bin cutoff for the K-within histogram")
    ap.add_argument("--priors", nargs="+", default=None,
                    help="Restrict to these priors. Default: all priors.")
    ap.add_argument("--font-scale", type=float, default=1.5)
    args = ap.parse_args()

    global set_paper_style
    _orig = set_paper_style
    set_paper_style = lambda fs=args.font_scale: _orig(fs)  # noqa: E731

    ckpt_tag = args.ckpt_tag
    priors = set(args.priors) if args.priors else PRIORS_INCLUDED
    files = _eval_files(ckpt_tag, priors=priors)
    print(f"[main] ckpt_tag={ckpt_tag}, {len(files)} eval files, "
          f"priors={sorted(priors)}, workers={args.workers}")
    if not files:
        raise SystemExit("[main] no eval files in scope")

    file_strs = sorted((str(fp) for fp in files),
                       key=lambda s: -Path(s).stat().st_size)

    t0 = time.time()
    results = []
    with Pool(processes=args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_scan_one, file_strs)):
            results.append(res)
            print(f"  [{i+1}/{len(files)}] {res['n_train']:,} train recs, "
                  f"{res['n_regions']:,} regions, "
                  f"{res['n_segments']:,} segments", flush=True)
    print(f"[main] scanned in {time.time()-t0:.0f}s")

    n_no_k = sum(r["n_regions_no_kcorpus"] for r in results)
    if n_no_k:
        print(f"[main] note: {n_no_k:,} regions skipped (K_corpus missing/0)")

    # (1) cross-patient recurrence
    report_cross_patient_recurrence(results)

    # (2) & (3) within-patient recurrence: stat + plot
    kw_regions = Counter()
    for r in results:
        for k, w in r["kw_rev_k1_regions"].items():
            kw_regions[int(k)] += w
    report_within_patient_recurrence(kw_regions)
    plot_within_patient_revealing_k1(kw_regions, ckpt_tag, xmax=args.xmax)


if __name__ == "__main__":
    main()
