#!/usr/bin/env python3
"""
The resolver walks backward through HEADER_RE matches in a source note text
to find the most recent parent-defining marker, returning a section
label such as "pe/cardiovascular", "ros/respiratory", "hpi", etc.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import config  # noqa: E402

KNOWN_HEADERS = config.KNOWN_HEADERS


def _build_header_regex(headers):
    parts = []
    for h in sorted(set(headers), key=lambda x: -len(x)):
        toks = h.split()
        pat = r"\s+".join(re.escape(t) for t in toks)
        parts.append(pat)
    alt = "|".join(parts)
    return re.compile(
        r"(?ix)"
        r"(?:(?<=^)|(?<=\s))"
        r"(?P<header>" + alt + r")"
        r"\s*:"
    )


HEADER_RE = _build_header_regex(KNOWN_HEADERS)


SECTION_HEADER_TAXONOMY     = config.SECTION_HEADER_TAXONOMY
BLANKET_TEMPLATED_SECTIONS  = config.BLANKET_TEMPLATED_SECTIONS


def classify_header_match(header_text):
    h = header_text.strip().lower()
    return SECTION_HEADER_TAXONOMY.get(h, ("other", None))


def resolve_section_for_region(source_note_text, region_pos,
                                 default_for_unresolved_both=None):
    if not source_note_text or region_pos <= 0:
        return None
    matches = list(HEADER_RE.finditer(source_note_text, 0, region_pos))
    if not matches:
        return None
    last_match = matches[-1]
    leaf = last_match.group("header").strip().lower()
    role, parent = classify_header_match(leaf)
    if role in ("pe_leaf", "ros_leaf"):
        return f"{parent}/{leaf}"
    if role == "both_leaf":
        for m in reversed(matches[:-1]):
            _, parent = classify_header_match(m.group("header"))
            if parent is not None:
                return f"{parent}/{leaf}"
        if default_for_unresolved_both:
            return f"{default_for_unresolved_both}/{leaf}"
        return leaf
    return leaf


def segments_in_region(region_text, source_note_text, region_pos_in_note):
    if not region_text:
        return
    internal_matches = list(HEADER_RE.finditer(region_text))

    if not internal_matches:
        section = resolve_section_for_region(source_note_text, region_pos_in_note)
        yield (section, 0, len(region_text), True)
        return

    first = internal_matches[0]
    if first.start() > 0:
        section = resolve_section_for_region(source_note_text, region_pos_in_note)
        yield (section, 0, first.start(), True)

    for i, m in enumerate(internal_matches):
        seg_start = m.start()
        seg_end = (internal_matches[i + 1].start()
                   if i + 1 < len(internal_matches)
                   else len(region_text))
        section = resolve_section_for_region(
            source_note_text,
            (region_pos_in_note if region_pos_in_note is not None else 0) + m.end()
        )
        is_leading = (i == 0 and m.start() == 0)
        yield (section, seg_start, seg_end, is_leading)

_WORKER_APPLY = False
_WORKER_FORCE = False


def _worker_init(apply_flag, force_flag):
    global _WORKER_APPLY, _WORKER_FORCE
    _WORKER_APPLY = apply_flag
    _WORKER_FORCE = force_flag


def _process_file_worker(fp_str):
    """One file → enrich regions with `segments`, write back if apply."""
    import json
    from collections import Counter
    from pathlib import Path

    fp = Path(fp_str)
    apply_flag = _WORKER_APPLY
    force_flag = _WORKER_FORCE

    records = []
    with open(fp) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    stats = {
        "file_path":                     str(fp),
        "file_changed":                  False,
        "records_processed":             0,
        "records_skipped_non_train":     0,
        "records_skipped_no_attr":       0,
        "records_skipped_fully_done":    0,
        "regions_skipped_done":          0,
        "total_regions":                 0,
        "total_segments":                0,
        "section_counts":                Counter(),
        "pe_leaf_collapsed":             Counter(),
        "ros_leaf_collapsed":            Counter(),
    }

    n_changed_in_file = 0
    for rec in records:
        if (rec.get("eval_split") or "unknown") != "train":
            stats["records_skipped_non_train"] += 1
            continue
        attr = rec.get("source_attribution")
        if not isinstance(attr, dict):
            stats["records_skipped_no_attr"] += 1
            continue
        regions = attr.get("regions") or []
        if not regions:
            continue

        # Quick check: are ALL regions already enriched?
        if not force_flag and all("segments" in r for r in regions):
            stats["records_skipped_fully_done"] += 1
            continue

        record_touched = False
        for r in regions:
            # Per-region skip: leave already-enriched regions alone unless --force
            if not force_flag and "segments" in r:
                stats["regions_skipped_done"] += 1
                # Still count its segments toward stats so per-prior totals are
                # consistent — but skip the regex work.
                existing_segs = r.get("segments") or []
                stats["total_regions"] += 1
                stats["total_segments"] += len(existing_segs)
                for seg in existing_segs:
                    sec = seg.get("section")
                    stats["section_counts"][sec or "(no_header)"] += 1
                    if sec and sec.startswith("pe/"):
                        stats["pe_leaf_collapsed"][sec] += 1
                    elif sec and sec.startswith("ros/"):
                        stats["ros_leaf_collapsed"][sec] += 1
                continue

            region_text       = r.get("region_text") or ""
            source_note_text  = r.get("winning_source_note_text")
            region_pos        = r.get("winning_source_note_pos")
            if not region_text:
                r["segments"] = []
                record_touched = True
                continue
            segs = []
            for (section, seg_start, seg_end, is_leading) in (
                segments_in_region(region_text, source_note_text, region_pos)
            ):
                segs.append({
                    "section":             section,
                    "seg_start_in_region": seg_start,
                    "seg_end_in_region":   seg_end,
                    "is_leading":          is_leading,
                })
                stats["section_counts"][section or "(no_header)"] += 1
                if section and section.startswith("pe/"):
                    stats["pe_leaf_collapsed"][section] += 1
                elif section and section.startswith("ros/"):
                    stats["ros_leaf_collapsed"][section] += 1
            r["segments"] = segs
            record_touched = True
            stats["total_regions"] += 1
            stats["total_segments"] += len(segs)

        if record_touched:
            stats["records_processed"] += 1
            n_changed_in_file += 1

    if apply_flag and n_changed_in_file > 0:
        with open(fp, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        stats["file_changed"] = True

    stats["section_counts"]     = dict(stats["section_counts"])
    stats["pe_leaf_collapsed"]  = dict(stats["pe_leaf_collapsed"])
    stats["ros_leaf_collapsed"] = dict(stats["ros_leaf_collapsed"])
    return stats


def _cli_main():
    import argparse
    import multiprocessing as mp
    import time
    from collections import Counter
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "1_priors"))
    import build_prompt_from_priors as _priors

    RESULTS_DIR = config.RESULTS_DIR

    DEFAULT_PRIORS = _priors.PRIOR_TIERS
    DEFAULT_PROBES = list(_priors.PROBE_TYPES)
    ALL_PROBES     = set(_priors.PROBE_TYPES)

    FNAME_RE = re.compile(r"^(?P<prior>[a-z_]+)__(?P<probe>[a-z_]+)\.jsonl$")

    def iter_target_files(ckpt_tag, target_priors, target_probes):
        ckpt_dir = RESULTS_DIR / ckpt_tag
        if not ckpt_dir.exists():
            return
        for dx_dir in sorted(p for p in ckpt_dir.iterdir() if p.is_dir()):
            for fp in sorted(dx_dir.glob("*.jsonl")):
                m = FNAME_RE.match(fp.name)
                if not m:
                    continue
                prior, probe = m.group("prior"), m.group("probe")
                if prior in target_priors and probe in target_probes:
                    yield fp

    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="Re-enrich regions that already have `segments` "
                         "(default: skip).")
    ap.add_argument("--ckpt-tag", dest="ckpt_tag", default=config.DEFAULT_CHECKPOINT,
                    help="Results subdirectory name, e.g. 'final' or 'checkpoint-14178'")
    ap.add_argument("--priors",      nargs="+", default=DEFAULT_PRIORS)
    ap.add_argument("--probes",      nargs="+", default=DEFAULT_PROBES,
                    choices=sorted(ALL_PROBES))
    ap.add_argument("--workers",     type=int, default=16,
                    help="Number of parallel worker processes (default: 16)")
    args = ap.parse_args()

    target_priors = set(args.priors)
    target_probes = set(args.probes)

    print(f"[scan] ckpt_tag   : {args.ckpt_tag}")
    print(f"[scan] priors     : {sorted(target_priors)}")
    print(f"[scan] probes     : {sorted(target_probes)}")
    print(f"[scan] workers    : {args.workers}")
    print(f"[scan] mode       : {'APPLY' if args.apply else 'DRY RUN'}")
    print(f"[scan] force      : {args.force}")
    print()

    matched = list(iter_target_files(args.ckpt_tag, target_priors, target_probes))
    print(f"[scan] {len(matched)} files match scope")
    if not matched:
        return

    matched_sorted = sorted(matched, key=lambda p: -p.stat().st_size)
    n_workers = max(1, min(args.workers, len(matched_sorted)))
    worker_args = [str(p) for p in matched_sorted]

    print(f"[parallel] launching {n_workers} worker(s) for {len(worker_args)} files")
    t0 = time.time()

    with mp.Pool(
        processes=n_workers,
        initializer=_worker_init,
        initargs=(args.apply, args.force),
    ) as pool:
        all_stats = []
        for i, stats in enumerate(pool.imap_unordered(_process_file_worker, worker_args)):
            all_stats.append(stats)
            if (i + 1) % 5 == 0 or (i + 1) == len(worker_args):
                el = time.time() - t0
                print(f"  [{i+1}/{len(worker_args)}] {el:.0f}s elapsed", flush=True)

    elapsed = time.time() - t0

    files_changed = sum(1 for s in all_stats if s["file_changed"])
    agg = {
        "records_processed":          0,
        "records_skipped_non_train":  0,
        "records_skipped_no_attr":    0,
        "records_skipped_fully_done": 0,
        "regions_skipped_done":       0,
        "total_regions":              0,
        "total_segments":             0,
    }
    section_counts     = Counter()
    pe_leaf_collapsed  = Counter()
    ros_leaf_collapsed = Counter()
    for s in all_stats:
        for k in agg:
            agg[k] += s[k]
        for k, v in s["section_counts"].items():
            section_counts[k] += v
        for k, v in s["pe_leaf_collapsed"].items():
            pe_leaf_collapsed[k] += v
        for k, v in s["ros_leaf_collapsed"].items():
            ros_leaf_collapsed[k] += v

    print()
    print("=" * 72)
    print(f"[done] in {elapsed:.0f}s (parallel, {n_workers} workers)")
    print(f"  files in scope                : {len(matched)}")
    print(f"  files changed                 : {files_changed}")
    print(f"  train records processed       : {agg['records_processed']:,}")
    print(f"  skipped (non-train)           : {agg['records_skipped_non_train']:,}")
    print(f"  skipped (no source_attribution): {agg['records_skipped_no_attr']:,}")
    print(f"  skipped (all regions done)    : {agg['records_skipped_fully_done']:,}")
    print(f"  regions skipped (already done): {agg['regions_skipped_done']:,}")
    print(f"  total regions enriched        : {agg['total_regions']:,}")
    print(f"  total segments emitted        : {agg['total_segments']:,}")
    if agg['total_regions'] > 0:
        print(f"  mean segments per region      : "
              f"{agg['total_segments']/agg['total_regions']:.2f}")
    if not args.apply:
        print()
        print(f"  DRY RUN — pass --apply to write back")
    print("=" * 72)
    print()
    print(f"=== Top 30 sections by segment count ===")
    for section, ct in section_counts.most_common(30):
        print(f"  {section[:55]:<55}  {ct:>8,}")
    print()
    print(f"=== PE-leaf segments (collapsed to pe/<leaf>) ===")
    for sec, ct in pe_leaf_collapsed.most_common():
        print(f"  {sec:<35}  {ct:>8,}")
    print()
    print(f"=== ROS-leaf segments (collapsed to ros/<leaf>) ===")
    for sec, ct in ros_leaf_collapsed.most_common():
        print(f"  {sec:<35}  {ct:>8,}")


if __name__ == "__main__":
    _cli_main()
