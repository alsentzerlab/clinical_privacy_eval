#!/usr/bin/env python3
"""
Mean # of distinct source notes from generations with at least one
memorized region, and % of contiguous memorized regions which stitch
together verbatim spans from multiple source notes.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config  # noqa: E402

RESULTS_DIR = config.RESULTS_DIR


def eval_files_for(ckpt_tag, prior, probe):
    root = RESULTS_DIR / ckpt_tag
    if not root.exists():
        return []
    out = []
    for fp in sorted(root.rglob(f"{prior}__{probe}.jsonl")):
        rel = fp.relative_to(root)
        if len(rel.parts) != 2:
            continue
        dx, _ = rel.parts
        if dx not in config.SENSITIVE_DIAGNOSES:
            continue
        out.append(fp)
    return out


def note_key_tuple(k):
    if not isinstance(k, dict):
        return None
    return (k.get("encounter_date"), k.get("row_idx"))


# Reconstruct pre-split (contiguous) regions and detect cross-note stitching

def _finalize_run(run, original_regions):
    if not run:
        return
    attributed_pieces = [p for p in run if p.get("attributed")]
    distinct_notes = set()
    for p in attributed_pieces:
        nk = note_key_tuple(p.get("winning_source_note_key"))
        if nk is not None:
            distinct_notes.add(nk)
    original_regions.append({"pieces": run, "distinct_notes": len(distinct_notes)})


def reconstruct_pre_split_regions(regions):
    """Walk regions in gen order; contiguous (touching) pieces collapse
    into one pre-split region."""
    out = []
    sorted_r = sorted(regions, key=lambda x: x.get("gen_start", 0))
    cur_run = []
    cur_end = None
    for r in sorted_r:
        gs = r.get("gen_start")
        ge = r.get("gen_end")
        if gs is None or ge is None:
            continue
        if cur_end is None or gs == cur_end:
            cur_run.append(r)
        else:
            _finalize_run(cur_run, out)
            cur_run = [r]
        cur_end = ge
    if cur_run:
        _finalize_run(cur_run, out)
    return out


def is_cross_note_stitch(orig):
    """A pre-split (contiguous) region stitches spans from multiple source
    notes together if it has >1 piece, at least one attributed piece, and
    those pieces resolve to more than one distinct source note."""
    pieces = orig["pieces"]
    attributed_pieces = [p for p in pieces if p.get("attributed")]
    return len(pieces) > 1 and bool(attributed_pieces) and orig["distinct_notes"] != 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-tag", dest="ckpt_tag", default=config.DEFAULT_CHECKPOINT,
                    help="Results subdirectory name, e.g. 'final' or 'checkpoint-14178'")
    ap.add_argument("--prior", default="encounter_info")
    ap.add_argument("--probe", default="note_continuation")
    args = ap.parse_args()

    ckpt_tag, prior, probe = args.ckpt_tag, args.prior, args.probe
    files = eval_files_for(ckpt_tag, prior, probe)
    print(f"[main] ckpt_tag={ckpt_tag} prior={prior} probe={probe}, {len(files)} eval files")
    if not files:
        raise SystemExit("[main] no eval files in scope")

    n_contiguous_total = 0
    n_cross_note = 0
    n_records_with_attr = 0
    sum_distinct_notes = 0
    n_scanned = n_train = 0

    for fp in files:
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
                if not rec.get("patient_id"):
                    continue
                n_train += 1

                attr = rec.get("source_attribution")
                regions = attr.get("regions") if isinstance(attr, dict) else []
                if not regions:
                    continue

                # % of contiguous regions stitching across source notes
                original_regions = reconstruct_pre_split_regions(regions)
                for orig in original_regions:
                    n_contiguous_total += 1
                    if is_cross_note_stitch(orig):
                        n_cross_note += 1

                # mean # distinct source notes, for records w/ >=1 attributed region
                attributed_keys = [note_key_tuple(r.get("winning_source_note_key"))
                                    for r in regions if r.get("attributed")]
                attributed_keys = [k for k in attributed_keys if k is not None]
                if attributed_keys:
                    n_records_with_attr += 1
                    sum_distinct_notes += len(set(attributed_keys))

    print(f"[scan] {n_scanned:,} records scanned, {n_train:,} train records")
    print()

    mean_distinct_notes = sum_distinct_notes / n_records_with_attr if n_records_with_attr else 0.0
    print(f"[distinct_source_notes] generations with >=1 memorized region: {n_records_with_attr:,}")
    print(f"[distinct_source_notes] mean distinct source notes: {mean_distinct_notes:.2f}")

    pct_cross = 100 * n_cross_note / n_contiguous_total if n_contiguous_total else 0.0
    print()
    print(f"[cross_note_stitching] contiguous memorized regions: {n_contiguous_total:,}")
    print(f"[cross_note_stitching] {pct_cross:.1f}% stitch together verbatim spans "
          f"from multiple source notes  (n={n_cross_note:,})")


if __name__ == "__main__":
    main()
