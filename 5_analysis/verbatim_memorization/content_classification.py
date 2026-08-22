
# Decomposes memorized content under a single note-fragment (Fig 4)

import argparse
import json
import time
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "1_priors"))
import build_prompt_from_priors as _priors  # noqa: E402

RESULTS_DIR  = config.RESULTS_DIR
OUT_DIR      = config.BASE_DIR / "analysis" / "memorized_content"
FIGURE_DIR   = config.FIGURES_DIR / "verbatim_memorization"


NOTE_FRAGMENT_PRIORS = [
    p for p in _priors.PRIOR_TIERS if p in _priors.UNSTRUCTURED_PRIORS
]

PRIOR_TO_PROBE = {p: "note_continuation" for p in NOTE_FRAGMENT_PRIORS}

PRIOR_PRETTY = {
    "encounter_info":        "Encounter Info",
    "encounter_info_cc":     "Encounter Info + CC",
    "encounter_info_cc_hpi": "Encounter Info + CC + HPI",
}


SECTION_PRETTY = {
    "ros":                          "Review of Systems",
    "pe":                           "Physical Exam",
    "hpi":                          "History of Present Illness",
    "cc":                           "Chief Complaint",
    "past medical history":         "Past Medical History",
    "past medical history / family history / social history":
                                    "PMH / FH / SH",
    "family history":               "Family History",
    "social history":               "Social History",
    "surgical history":             "Surgical History",
    "current medical providers":    "Current Medical Providers",
    "current medications":          "Current Medications",
    "current problems":             "Current Problems",
    "tobacco/alcohol/supplements":  "Tobacco/Alcohol/Supplements",
    "gynecological history":        "Gynecological History",
    "marital status":               "Marital Status",
    "occupation":                   "Occupation",
    "exercise":                     "Exercise",
    "children":                     "Children",
    "vaccine":                      "Vaccinations",
    "immunizations":                "Immunizations",
    "allergies":                    "Allergies",
    "assessment":                   "Assessment",
    "plan":                         "Plan",
    "medications":                  "Medications",
    "primary diagnosis":            "Primary Diagnosis",
    "orders":                       "Orders",
    "objective":                    "Objective",
    "vitals":                       "Vitals",
    "subjective":                   "Subjective",
    "history":                      "History",
    "hospitalizations":             "Hospitalizations",
    "mental health history":        "Mental Health History",
    "substance abuse history":      "Substance Abuse History",
    "preventive health maintenance": "Preventive Health Maintenance",
    "functional status":            "Functional Status",
    "hobbies/recreation":           "Hobbies/Recreation",
    "caffeine":                     "Caffeine",
    "lab/test results":             "Lab/Test Results",
    "patient recommendations":      "Patient Recommendations",
    "charge capture":               "Charge Capture",
    "exams":                        "Exams",
    "alcohol":                      "Alcohol",
    "prescriptions":                "Prescriptions",
    "communicable diseases (eg stds)": "Communicable Diseases",
    "(no_header)":                  "(No Header)",
}


def pretty_section(machine_name):
    """Look up a human-readable label for a collapsed section name.
    Falls back to title-casing the machine name."""
    if machine_name in SECTION_PRETTY:
        return SECTION_PRETTY[machine_name]
    # Generic fallback: title-case and replace slashes/underscores
    return machine_name.replace("_", " ").title()


def collapse_section_for_plot(section_label):
    """Collapse pe/<leaf> -> 'pe' and ros/<leaf> -> 'ros'. Other labels
    pass through. None or missing -> '(no_header)'.

    Also handles two bare leaves that find_span_headers.py did not yet
    classify as ROS-only at the time these JSONLs were segmented:
    `endocrine` and `hematologic/lymphatic`. Both are CMS E&M ROS-only
    body systems, so we collapse them into `ros` here for the plot.
    Once find_span_headers.py is re-run with the updated leaf sets,
    these will naturally come through as `ros/endocrine` and
    `ros/hematologic/lymphatic` and the regular `ros/` prefix branch
    will catch them.
    """
    if section_label is None or section_label == "":
        return "(no_header)"
    if section_label.startswith("pe/"):
        return "pe"
    if section_label.startswith("ros/"):
        return "ros"
    if section_label in ("endocrine", "hematologic/lymphatic"):
        return "ros"
    return section_label


def process_file(fp_str):
    """Process one JSONL file. Returns dict with section_counts + stats."""
    fp = Path(fp_str)
    t0 = time.time()

    section_counts = defaultdict(lambda: {
        "rev_k1": 0, "rev_kgt1": 0,
        "tpl_k1": 0, "tpl_kgt1": 0,
    })

    n_scanned             = 0
    n_train               = 0
    n_with_attr           = 0
    n_records_with_regions = 0
    n_regions_total       = 0
    n_regions_k0_skipped  = 0
    n_regions_used        = 0
    n_segments_used       = 0
    n_segments_missing_counts = 0
    blanket_segment_count = 0
    k_dist = Counter()

    with open(fp) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_scanned += 1

            if (rec.get("eval_split") or "unknown") != "train":
                continue
            n_train += 1

            attr = rec.get("source_attribution")
            if not isinstance(attr, dict):
                continue
            n_with_attr += 1

            regions = attr.get("regions") or []
            if not regions:
                continue
            n_records_with_regions += 1

            for r in regions:
                n_regions_total += 1

                k = r.get("K_corpus", 0)
                if not k:
                    n_regions_k0_skipped += 1
                    continue
                is_k1 = (k == 1)
                k_dist["K=1" if is_k1 else "K>1"] += 1
                n_regions_used += 1

                segments = r.get("segments") or []
                for seg in segments:
                    if ("n_templated" not in seg
                            or "n_revealing" not in seg):
                        n_segments_missing_counts += 1
                        continue

                    n_tpl = seg["n_templated"]
                    n_rev = seg["n_revealing"]
                    if n_tpl == 0 and n_rev == 0:
                        continue

                    if seg.get("blanket_section"):
                        blanket_segment_count += 1

                    section = collapse_section_for_plot(seg.get("section"))

                    if is_k1:
                        section_counts[section]["rev_k1"] += n_rev
                        section_counts[section]["tpl_k1"] += n_tpl
                    else:
                        section_counts[section]["rev_kgt1"] += n_rev
                        section_counts[section]["tpl_kgt1"] += n_tpl
                    n_segments_used += 1

    elapsed = time.time() - t0
    return {
        "fp":             str(fp),
        "section_counts": {k: dict(v) for k, v in section_counts.items()},
        "stats": {
            "n_scanned":             n_scanned,
            "n_train":               n_train,
            "n_with_attr":           n_with_attr,
            "n_records_with_regions": n_records_with_regions,
            "n_regions_total":       n_regions_total,
            "n_regions_k0_skipped":  n_regions_k0_skipped,
            "n_regions_used":        n_regions_used,
            "n_segments_used":       n_segments_used,
            "n_segments_missing_counts": n_segments_missing_counts,
            "blanket_segment_count": blanket_segment_count,
            "k_dist":                dict(k_dist),
        },
        "elapsed":        elapsed,
    }


def eval_files_for_prior(ckpt_tag, prior, probe):
    root = RESULTS_DIR / ckpt_tag
    if not root.exists():
        return []
    out = []
    for fp in sorted(root.rglob(f"{prior}__{probe}.jsonl")):
        try:
            rel = fp.relative_to(root)
        except ValueError:
            continue
        if len(rel.parts) != 2:
            continue
        out.append(fp)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-tag", dest="ckpt_tag", default=config.DEFAULT_CHECKPOINT,
                    help="Results subdirectory name, e.g. 'final' or 'checkpoint-14178'")
    ap.add_argument("--prior", required=True, choices=NOTE_FRAGMENT_PRIORS)
    ap.add_argument("--workers", type=int, default=16,
                    help="Parallel worker processes (default: 16)")
    ap.add_argument("--top-n", type=int, default=10,
                    help="Number of top sections to plot (default: 10)")
    args = ap.parse_args()
    ckpt_tag = args.ckpt_tag
    prior = args.prior
    probe = PRIOR_TO_PROBE[prior]
    top_n = args.top_n

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    files = eval_files_for_prior(ckpt_tag, prior, probe)
    print(f"[init] eval files: {len(files)}")
    if not files:
        raise SystemExit(f"no eval files for prior={prior} probe={probe} "
                         f"ckpt_tag={ckpt_tag}")

    file_sizes = [(fp, fp.stat().st_size) for fp in files]
    total_mb = sum(sz for _, sz in file_sizes) / 1024 / 1024
    print(f"[init] total input size: {total_mb:,.0f} MB across {len(files)} files")
    print(f"[init] workers: {args.workers}")
    print()

    print(f"[pass] processing {len(files)} files in parallel...")
    t0 = time.time()
    file_strs = [str(fp) for fp in files]
    # Largest-first for better load balance
    file_strs.sort(key=lambda s: -Path(s).stat().st_size)

    results = []
    with Pool(processes=args.workers) as pool:
        for i, result in enumerate(pool.imap_unordered(process_file, file_strs)):
            results.append(result)
            fp_short = Path(result["fp"]).relative_to(
                RESULTS_DIR / ckpt_tag)
            n_kept = result["stats"]["n_regions_used"]
            print(f"  [{i+1}/{len(files)}] {fp_short}  "
                  f"({result['elapsed']:.0f}s, "
                  f"{result['stats']['n_train']:,} train recs, "
                  f"{n_kept:,} regions used)", flush=True)

    wall = time.time() - t0
    cpu_seconds = sum(r["elapsed"] for r in results)
    print(f"[pass] done in {wall:.0f}s wall "
          f"({cpu_seconds:.0f}s cpu, {cpu_seconds/max(wall,1):.1f}x speedup)")
    print()

    section_counts = defaultdict(lambda: {
        "rev_k1": 0, "rev_kgt1": 0,
        "tpl_k1": 0, "tpl_kgt1": 0,
    })
    n_scanned = 0
    n_train = 0
    n_with_attr = 0
    n_records_with_regions = 0
    n_regions_total = 0
    n_regions_k0_skipped = 0
    n_regions_used = 0
    n_segments_used = 0
    n_segments_missing_counts = 0
    blanket_segment_count = 0
    k_dist = Counter()

    for r in results:
        for section, counts in r["section_counts"].items():
            section_counts[section]["rev_k1"]   += counts["rev_k1"]
            section_counts[section]["rev_kgt1"] += counts["rev_kgt1"]
            section_counts[section]["tpl_k1"]   += counts["tpl_k1"]
            section_counts[section]["tpl_kgt1"] += counts["tpl_kgt1"]
        s = r["stats"]
        n_scanned              += s["n_scanned"]
        n_train                += s["n_train"]
        n_with_attr            += s["n_with_attr"]
        n_records_with_regions += s["n_records_with_regions"]
        n_regions_total        += s["n_regions_total"]
        n_regions_k0_skipped   += s["n_regions_k0_skipped"]
        n_regions_used         += s["n_regions_used"]
        n_segments_used        += s["n_segments_used"]
        n_segments_missing_counts += s["n_segments_missing_counts"]
        blanket_segment_count  += s["blanket_segment_count"]
        for k, v in s["k_dist"].items():
            k_dist[k] += v

    print(f"  records scanned                : {n_scanned:,}")
    print(f"  train-split records            : {n_train:,}")
    print(f"  with source_attribution        : {n_with_attr:,}")
    print(f"  with regions                   : {n_records_with_regions:,}")
    print(f"  regions total                  : {n_regions_total:,}")
    print(f"  regions skipped (K_corpus=0)   : {n_regions_k0_skipped:,}")
    print(f"  regions used                   : {n_regions_used:,}")
    print(f"  segments used                  : {n_segments_used:,}")
    print(f"  segments missing token counts  : {n_segments_missing_counts:,}")
    print(f"  blanket-section segments       : {blanket_segment_count:,}")
    print(f"  K split                        : {dict(k_dist)}")
    print()

    if n_segments_missing_counts > 0:
        print(f"  WARNING: {n_segments_missing_counts:,} segments are missing "
              f"`n_templated`/`n_revealing`. Run "
              f"2_privacy_measures/verbatim_memorization/content_classification/"
              f"content_classification.py --apply first.")
        print()

    if not section_counts:
        raise SystemExit("no usable segments found")

    # Build sorted aggregate (ALL sections — written to JSON)
    aggregate = []
    for section, counts in section_counts.items():
        total = sum(counts.values())
        aggregate.append({
            "section":      section,
            "total_tokens": total,
            "rev_k1":       counts["rev_k1"],
            "rev_kgt1":     counts["rev_kgt1"],
            "tpl_k1":       counts["tpl_k1"],
            "tpl_kgt1":     counts["tpl_kgt1"],
        })
    aggregate.sort(key=lambda x: -x["total_tokens"])
    total_tokens_all = sum(s["total_tokens"] for s in aggregate)

    agg_path = OUT_DIR / f"aggregate_{prior}_{ckpt_tag}_train.json"
    with open(agg_path, "w") as f:
        json.dump({
            "checkpoint":              ckpt_tag,
            "prior":                   prior,
            "probe":                   probe,
            "split":                   "train",
            "total_tokens":            total_tokens_all,
            "n_train_records":         n_train,
            "n_records_with_regions":  n_records_with_regions,
            "n_regions_total":         n_regions_total,
            "n_regions_used":          n_regions_used,
            "n_regions_k0_skipped":    n_regions_k0_skipped,
            "n_segments_used":         n_segments_used,
            "n_sections":              len(aggregate),
            "k_distribution":          dict(k_dist),
            "sections":                aggregate,
        }, f, indent=2)
    print(f"[agg]   → {agg_path}")

    total_rev = sum(s["rev_k1"] + s["rev_kgt1"] for s in aggregate)
    total_tpl = sum(s["tpl_k1"] + s["tpl_kgt1"] for s in aggregate)
    total_k1  = sum(s["rev_k1"] + s["tpl_k1"]   for s in aggregate)
    total_kgt = sum(s["rev_kgt1"] + s["tpl_kgt1"] for s in aggregate)
    print()
    print(f"Total memorized tokens     : {total_tokens_all:,}")
    print(f"  Clinically revealing     : {total_rev:>10,}  "
          f"({100*total_rev/max(total_tokens_all,1):.1f}%)")
    print(f"  Templated                : {total_tpl:>10,}  "
          f"({100*total_tpl/max(total_tokens_all,1):.1f}%)")
    print(f"  K=1 (unique to 1 patient): {total_k1:>10,}  "
          f"({100*total_k1/max(total_tokens_all,1):.1f}%)")
    print(f"  K>1 (≥2 patients)        : {total_kgt:>10,}  "
          f"({100*total_kgt/max(total_tokens_all,1):.1f}%)")
    print()
    print(f"Per-section token totals (top {top_n}):")
    print(f"  {'section':<35}  {'tokens':>10}  {'rev%':>6}  {'tpl%':>6}  "
          f"{'K=1%':>6}  {'K>1%':>6}")
    for s in aggregate[:top_n]:
        tot = s["total_tokens"]
        if tot == 0:
            continue
        rev = s["rev_k1"] + s["rev_kgt1"]
        tpl = s["tpl_k1"] + s["tpl_kgt1"]
        k1  = s["rev_k1"] + s["tpl_k1"]
        kgt = s["rev_kgt1"] + s["tpl_kgt1"]
        print(f"  {s['section']:<35}  {tot:>10,}  "
              f"{100*rev/tot:>5.1f}%  {100*tpl/tot:>5.1f}%  "
              f"{100*k1/tot:>5.1f}%  {100*kgt/tot:>5.1f}%")
    print()

    sections_to_plot = [s for s in aggregate if s["total_tokens"] > 0][:top_n]
    tokens_in_plot = sum(s["total_tokens"] for s in sections_to_plot)
    coverage_pct = 100 * tokens_in_plot / max(total_tokens_all, 1)
    print(f"Top {top_n} sections cover {coverage_pct:.1f}% of all memorized tokens.")
    print()

    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    n_bars = len(sections_to_plot)
    x_pos = np.arange(n_bars)
    section_names = [pretty_section(s["section"]) for s in sections_to_plot]

    # Normalize to fractions of TOTAL tokens so bar heights reflect the
    # section's share of the entire memorized corpus (not just the
    # top-N share).
    denom = max(total_tokens_all, 1)
    rev_k1   = np.array([s["rev_k1"]   / denom for s in sections_to_plot])
    rev_kgt1 = np.array([s["rev_kgt1"] / denom for s in sections_to_plot])
    tpl_k1   = np.array([s["tpl_k1"]   / denom for s in sections_to_plot])
    tpl_kgt1 = np.array([s["tpl_kgt1"] / denom for s in sections_to_plot])

    fig_width = max(10, min(20, 1.0 * n_bars + 4))
    fig, ax = plt.subplots(figsize=(fig_width, 7))

    _cb = sns.color_palette("colorblind")
    COLOR_REV = _cb[3]   # vermillion
    COLOR_TPL = _cb[0]   # blue

    ax.bar(x_pos, rev_k1, color=COLOR_REV, edgecolor="black",
            linewidth=0.5, label="Clinically revealing, K=1")
    ax.bar(x_pos, rev_kgt1, bottom=rev_k1,
            color=COLOR_REV, edgecolor="black", linewidth=0.5,
            hatch="///", label="Clinically revealing, K>1")
    ax.bar(x_pos, tpl_k1, bottom=rev_k1 + rev_kgt1,
            color=COLOR_TPL, edgecolor="black", linewidth=0.5,
            label="Templated, K=1")
    ax.bar(x_pos, tpl_kgt1, bottom=rev_k1 + rev_kgt1 + tpl_k1,
            color=COLOR_TPL, edgecolor="black", linewidth=0.5,
            hatch="///", label="Templated, K>1")

    # Total label on top of each bar
    for i, _ in enumerate(sections_to_plot):
        total = (rev_k1[i] + rev_kgt1[i] + tpl_k1[i] + tpl_kgt1[i])
        if total > 0.005:
            ax.text(i, total + 0.002, f"{100*total:.1f}%",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(section_names, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Fraction of memorized tokens", fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)

    fig.tight_layout()
    out_path = FIGURE_DIR / f"memorized_content_composition_{prior}_{ckpt_tag}_train.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot]  → {out_path}")


if __name__ == "__main__":
    main()
