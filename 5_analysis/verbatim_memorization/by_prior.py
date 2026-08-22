"""
Computes verbatim memorization 
Plots verbatim memorization by prior (Fig 2) - Average share of a generated note copied verbatim from each 
patient's training notes. A token is counted as memorized if it falls within at least one
tau=30-gram match.

Plots Verbatim memorization hit rate by prior (Fig 9) - Fraction of generations
containing at least one verbatim tau=30-gram span matching the patient's
training notes. 
"""

import argparse
import csv
import hashlib
import json
import os
import sys
import textwrap
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "1_priors"))
import build_prompt_from_priors as _priors  # noqa: E402

RESULTS_DIR = config.RESULTS_DIR
OUT_ROOT    = config.FIGURES_DIR / "leakage_features"
FIGURE_DIR  = config.FIGURES_DIR / "verbatim_memorization"
TOKENIZER_PATH = config.TOKENIZER_PATH

PRIOR_ORDER = list(_priors.PRIOR_TIERS)
PROBE_TYPES = list(_priors.PROBE_TYPES)

PRIOR_PRETTY = {
    "public":                "Public",
    "public_named":          "Public + Name",
    "public_named_meds":     "Public + Name + Meds",
    "encounter_info":        "Encounter Info",
    "encounter_info_cc":     "Encounter Info + CC",
    "encounter_info_cc_hpi": "Encounter Info + CC + HPI",
}

_CB = sns.color_palette("colorblind", n_colors=len(PRIOR_ORDER))
PRIOR_COLORS = {p: _CB[i] for i, p in enumerate(PRIOR_ORDER)}

LABEL_FS = 18
TICK_FS  = 20
BAR_FS   = 20

CSV_COLUMNS = [
    "patient_id", "dx", "prior", "probe",
    "mem_token_pct", "mem_token_count", "n_gen_tokens",
    "patient_n_notes", "patient_n_tokens",
]


def _wrap(label, width=8):
    return "\n".join(textwrap.wrap(label, width=width, break_long_words=False))


_tokenizer = None
def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        print(f"[tok] loading {TOKENIZER_PATH}", flush=True)
        _tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_PATH), trust_remote_code=True)
    return _tokenizer


def compute_mem_token_pct(generation, spans):
    """Fraction of generation's tokens covered by >=1 memorized_spans entry."""
    if not generation:
        return None, 0, 0
    tok = get_tokenizer()
    enc = tok(generation, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]
    n_tokens = len(offsets)
    if n_tokens == 0:
        return None, 0, 0
    if not spans:
        return 0.0, 0, n_tokens

    char_ranges = []
    for s in spans:
        st = s.get("text", "") if isinstance(s, dict) else (s or "")
        if not st:
            continue
        start = 0
        while True:
            idx = generation.find(st, start)
            if idx < 0:
                break
            char_ranges.append((idx, idx + len(st)))
            start = idx + 1
    if not char_ranges:
        return 0.0, 0, n_tokens

    char_ranges.sort()
    merged = [list(char_ranges[0])]
    for s, e in char_ranges[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])

    covered = 0
    mi = 0
    for tok_s, tok_e in offsets:
        if tok_e <= tok_s:
            continue
        while mi < len(merged) and merged[mi][1] <= tok_s:
            mi += 1
        if mi == len(merged):
            break
        m_s, m_e = merged[mi]
        if m_s < tok_e and tok_s < m_e:
            covered += 1
    return covered / n_tokens, covered, n_tokens


def iter_eval_files(ckpt_dir, priors, probes):
    for fp in sorted(ckpt_dir.rglob("*.jsonl")):
        if "__" not in fp.stem:
            continue
        prior, probe = fp.stem.split("__", 1)
        if probe not in probes or prior not in priors:
            continue
        yield fp, prior, probe, fp.parent.name


def rec_key(patient_id, dx, prior, probe):
    return (patient_id, dx, prior, probe)


def load_existing(out_path):
    rows, done = [], set()
    if not out_path.exists():
        return rows, done
    with open(out_path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
            done.add(rec_key(row.get("patient_id", ""), row.get("dx", ""),
                             row.get("prior", ""), row.get("probe", "")))
    return rows, done


def scan_missing(ckpt_dir, priors, probes, done_keys):
    files = list(iter_eval_files(ckpt_dir, priors, probes))
    print(f"[scan] {len(files)} in-scope eval files; {len(done_keys):,} records already present", flush=True)

    new_records, needed_patients = [], set()
    n_total = n_skip = n_new = 0
    t0 = time.time()

    for fi, (fp, prior, probe, dx) in enumerate(files, start=1):
        file_new = 0
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("_record_type") or rec.get("error_type"):
                    continue
                if (rec.get("eval_split") or "unknown") != "train":
                    continue
                pid = rec.get("patient_id")
                if not pid:
                    continue
                n_total += 1
                k = rec_key(pid, dx, prior, probe)
                if k in done_keys:
                    n_skip += 1
                    continue

                generation = rec.get("generation") or ""
                spans = rec.get("memorized_spans") or []
                mp, mc, n_gen = compute_mem_token_pct(generation, spans)
                new_records.append({
                    "patient_id": pid, "dx": dx, "prior": prior, "probe": probe,
                    "mem_token_pct": f"{mp:.6f}" if mp is not None else "",
                    "mem_token_count": mc, "n_gen_tokens": n_gen,
                })
                needed_patients.add(pid)
                n_new += 1
                file_new += 1
        print(f"  [{fi}/{len(files)}] {fp.relative_to(ckpt_dir)}  "
              f"({f'+{file_new}' if file_new else 'complete'})", flush=True)

    print(f"[scan] {n_total:,} train records seen, {n_skip:,} already done, "
          f"{n_new:,} new, {len(needed_patients):,} new patients, {time.time()-t0:.0f}s")
    return new_records, needed_patients


def corpus_cache_path(out_dir, needed_patients):
    h = hashlib.sha1("\n".join(sorted(needed_patients)).encode("utf-8")).hexdigest()[:16]
    return out_dir / f".corpus_stats_{h}.json"


def compute_patient_corpus_stats(needed_patients, out_dir, batch_size=512, force=False):
    cache = corpus_cache_path(out_dir, needed_patients)
    if cache.exists() and not force:
        print(f"[corpus] cache hit {cache.name} -- skipping corpus stream", flush=True)
        with open(cache) as f:
            d = json.load(f)
        return defaultdict(int, d["n_notes"]), defaultdict(int, d["n_tokens"])

    tok = get_tokenizer()
    n_notes_by_pid, n_tokens_by_pid = defaultdict(int), defaultdict(int)

    print(f"[corpus] streaming {config.TRAIN_COHORT_PATH} for {len(needed_patients):,} patients", flush=True)
    t0 = time.time()
    n_scanned = n_kept = 0
    pending_pids, pending_texts = [], []

    def flush():
        if not pending_texts:
            return
        enc = tok(pending_texts, add_special_tokens=False)
        for pid, ids in zip(pending_pids, enc["input_ids"]):
            n_notes_by_pid[pid] += 1
            n_tokens_by_pid[pid] += len(ids)
        pending_pids.clear()
        pending_texts.clear()

    with open(config.TRAIN_COHORT_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_scanned += 1
            if n_scanned % 200_000 == 0:
                print(f"  [corpus] {n_scanned:,} scanned, {n_kept:,} kept, {time.time()-t0:.0f}s", flush=True)
            pid = row.get("PatientUid")
            if pid not in needed_patients:
                continue
            note_text = row.get("Note") or ""
            if not note_text:
                continue
            pending_pids.append(pid)
            pending_texts.append(note_text)
            n_kept += 1
            if len(pending_texts) >= batch_size:
                flush()
    flush()
    print(f"[corpus] done: {n_scanned:,} scanned, {n_kept:,} kept, {time.time()-t0:.0f}s")

    tmp = cache.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump({"n_notes": dict(n_notes_by_pid), "n_tokens": dict(n_tokens_by_pid)}, f)
    os.replace(tmp, cache)
    print(f"[corpus] cached -> {cache.name}")
    return n_notes_by_pid, n_tokens_by_pid


def run_data_step(ckpt_tag: str, priors: list[str], probes: list[str],
                  batch_size: int, force_corpus: bool, rebuild: bool) -> Path:
    ckpt_dir = RESULTS_DIR / ckpt_tag
    if not ckpt_dir.exists():
        raise SystemExit(f"missing {ckpt_dir}")

    out_dir = OUT_ROOT / ckpt_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "train_features_simple.csv"

    if rebuild:
        existing_rows, done_keys = [], set()
        print("[init] --rebuild: ignoring any existing CSV")
    else:
        existing_rows, done_keys = load_existing(out_path)
        print(f"[init] loaded {len(existing_rows):,} existing rows from CSV")

    new_records, needed_patients = scan_missing(ckpt_dir, priors, set(probes), done_keys)

    if not new_records and existing_rows:
        print("[done] nothing missing -- CSV already complete, leaving as-is")
        print(f"[done] -> {out_path}")
        return out_path

    if needed_patients:
        n_notes_by_pid, n_tokens_by_pid = compute_patient_corpus_stats(
            needed_patients, out_dir, batch_size=batch_size, force=force_corpus)
    else:
        n_notes_by_pid, n_tokens_by_pid = defaultdict(int), defaultdict(int)

    n_missing_corpus = 0
    for r in new_records:
        pn = n_notes_by_pid.get(r["patient_id"], 0)
        pt = n_tokens_by_pid.get(r["patient_id"], 0)
        if pn == 0:
            n_missing_corpus += 1
        r["patient_n_notes"], r["patient_n_tokens"] = pn, pt

    tmp_path = out_path.with_suffix(".csv.tmp")
    with open(tmp_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in existing_rows:
            w.writerow({c: r.get(c, "") for c in CSV_COLUMNS})
        for r in new_records:
            w.writerow({c: r.get(c, "") for c in CSV_COLUMNS})
    os.replace(tmp_path, out_path)

    total = len(existing_rows) + len(new_records)
    print(f"[done] {len(new_records):,} new rows appended ({len(existing_rows):,} kept; {total:,} total); "
          f"{n_missing_corpus:,} new rows with no corpus notes (likely held-out patients)")
    print(f"[done] -> {out_path}")
    return out_path


# Steps: mem_pct_by_prior / hit_rate_by_prior

def _load_csv_by_prior(csv_path: Path, priors: list[str]):
    """Returns {prior: [(mem_token_pct, mem_token_count), ...]}."""
    by_prior = defaultdict(list)
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            prior = r.get("prior")
            if prior not in priors:
                continue
            pct_str = r.get("mem_token_pct", "")
            if not pct_str:
                continue
            try:
                pct = float(pct_str)
            except ValueError:
                continue
            count_str = r.get("mem_token_count", "")
            try:
                count = int(count_str) if count_str != "" else (1 if pct > 0 else 0)
            except ValueError:
                count = 1 if pct > 0 else 0
            by_prior[prior].append((pct, count))
    return by_prior


def _bar_plot(priors_present, values, sems, ylabel, out_path, fmt_pct=True):
    x_pos = np.arange(len(priors_present))
    colors = [PRIOR_COLORS.get(p, "#666666") for p in priors_present]

    sns.set_context("notebook")
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.bar(x_pos, values, yerr=sems, capsize=4, color=colors,
          edgecolor="black", linewidth=0.5, width=0.6)
    for i, v in enumerate(values):
        ax.text(i, v + sems[i] + 0.002, f"{100*v:.1f}%",
                ha="center", va="bottom", fontsize=BAR_FS)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([_wrap(PRIOR_PRETTY.get(p, p)) for p in priors_present],
                       rotation=0, ha="center", fontsize=TICK_FS)
    ax.set_ylabel(ylabel, fontsize=LABEL_FS, labelpad=10)
    ax.tick_params(axis="y", labelsize=TICK_FS)

    y_top = max(v + s for v, s in zip(values, sems)) * 1.18
    ax.set_ylim(0, min(y_top, 1.02) if not fmt_pct else y_top)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"[saved] {out_path}")

def run_mem_pct_by_prior(csv_path: Path, priors: list[str], out_dir: Path) -> None:
    """Mean % of generated tokens memorized, by prior."""
    print(f"[mem_pct] load {csv_path}")
    by_prior = _load_csv_by_prior(csv_path, priors)
    priors_present = [p for p in priors if p in by_prior]
    if not priors_present:
        print("[mem_pct] no data for the requested priors — skipping")
        return

    means, sems = [], []
    for p in priors_present:
        vals = np.array([pct for pct, _ in by_prior[p]])
        means.append(float(np.mean(vals)))
        sems.append(float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0)
        print(f"  {p:<28}  n={len(vals):,}  mean={np.mean(vals):.4f}")

    _bar_plot(priors_present, means, sems,
             "Mean % of generated tokens memorized",
             out_dir / "mem_pct_by_prior.png")

def run_hit_rate_by_prior(csv_path: Path, priors: list[str], out_dir: Path) -> None:
    """Fraction of generations with >=1 tau=30-gram match, by prior."""
    print(f"[hit_rate] load {csv_path}")
    by_prior = _load_csv_by_prior(csv_path, priors)
    priors_present = [p for p in priors if p in by_prior]
    if not priors_present:
        print("[hit_rate] no data for the requested priors — skipping")
        return

    rates, sems = [], []
    for p in priors_present:
        hits = np.array([1.0 if count > 0 else 0.0 for _, count in by_prior[p]])
        n = len(hits)
        p_hat = float(np.mean(hits))
        rates.append(p_hat)
        sems.append(float(np.sqrt(p_hat * (1 - p_hat) / n)) if n > 0 else 0.0)
        print(f"  {p:<28}  n={n:,}  hit_rate={p_hat:.4f}")

    _bar_plot(priors_present, rates, sems,
             "% of generations with ≥ 1 τ-30 match",
             out_dir / "mem_hit_rate_by_prior.png", fmt_pct=False)


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-tag", dest="ckpt_tag", default=config.DEFAULT_CHECKPOINT,
                    help="Results subdirectory name, e.g. 'final' or 'checkpoint-14178'")
    ap.add_argument("--priors", nargs="+", default=PRIOR_ORDER, choices=PRIOR_ORDER)
    ap.add_argument("--probes", nargs="+", default=PROBE_TYPES, choices=PROBE_TYPES)
    ap.add_argument("--steps", nargs="+", default=["data", "mem_pct", "hit_rate"],
                    choices=["data", "mem_pct", "hit_rate"])
    ap.add_argument("--csv", type=Path, default=None,
                    help="Override the train_features_simple.csv path for mem_pct/hit_rate "
                         "(default: the path 'data' would write/read)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: <FIGURES_DIR>/verbatim_memorization")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--force-corpus", action="store_true")
    ap.add_argument("--rebuild", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    priors = [p for p in PRIOR_ORDER if p in args.priors]
    out_dir = args.out_dir or FIGURE_DIR

    csv_path = args.csv or (OUT_ROOT / args.ckpt_tag / "train_features_simple.csv")

    if "data" in args.steps:
        csv_path = run_data_step(args.ckpt_tag, priors, args.probes,
                                 args.batch, args.force_corpus, args.rebuild)

    if not csv_path.exists() and ("mem_pct" in args.steps or "hit_rate" in args.steps):
        raise SystemExit(f"missing {csv_path}\nRun with --steps data first "
                         f"(or pass --csv to point at an existing file).")

    if "mem_pct" in args.steps:
        run_mem_pct_by_prior(csv_path, priors, out_dir)
    if "hit_rate" in args.steps:
        run_hit_rate_by_prior(csv_path, priors, out_dir)


if __name__ == "__main__":
    main()
