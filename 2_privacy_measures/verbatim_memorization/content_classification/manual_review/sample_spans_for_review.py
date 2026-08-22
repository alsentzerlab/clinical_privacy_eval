#!/usr/bin/env python3
"""
sample_spans_for_review.py
===========================

Draws a blinded random sample of templated / clinically-revealing text
spans for manual review: 
  - review_blind.csv
  - review_key.csv   
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
import config  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "1_priors"))
import build_prompt_from_priors as _priors  # noqa: E402

RESULTS_DIR = config.RESULTS_DIR

FNAME_RE = re.compile(r"^(?P<prior>[a-z_]+)__(?P<probe>[a-z_]+)\.jsonl$")

DEFAULT_PRIORS = ["encounter_info"]
DEFAULT_PROBES = ["note_continuation"]

_INFORMATIVE_RE = re.compile(r"[A-Za-z0-9]")


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
                yield fp, prior, probe


def _rule_summary(seg):
    if seg.get("blanket_section"):
        return "BLANKET_SECTION"
    rc = seg.get("rule_counts") or {}
    if rc:
        return "|".join(sorted(rc.keys()))
    return None


def collect_pools(matched_files, ckpt_tag, min_span_chars, contiguous_only, n_per_class, context_chars):
    templated_pool = []
    revealing_pool = []
    stats = {
        "records_non_train":     0,
        "records_no_attr":       0,
        "regions_not_enriched":  0,
        "segments_not_enriched": 0,
        "segments_seen":         0,
    }

    def full(pool):
        return len(pool) >= n_per_class

    for fp, prior, probe in matched_files:
        if full(templated_pool) and full(revealing_pool):
            break
        dx = fp.parent.name
        with open(fp) as f:
            for line_idx, line in enumerate(f):
                if full(templated_pool) and full(revealing_pool):
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (rec.get("eval_split") or "unknown") != "train":
                    stats["records_non_train"] += 1
                    continue
                attr = rec.get("source_attribution")
                if not isinstance(attr, dict):
                    stats["records_no_attr"] += 1
                    continue
                regions = attr.get("regions") or []
                record_uid = f"{fp.name}:{line_idx}"

                for region_idx, r in enumerate(regions):
                    if full(templated_pool) and full(revealing_pool):
                        break
                    region_text = r.get("region_text") or ""
                    if "n_tokens" not in r:
                        stats["regions_not_enriched"] += 1
                        continue
                    for seg_idx, seg in enumerate(r.get("segments") or []):
                        if full(templated_pool) and full(revealing_pool):
                            break
                        if "n_templated" not in seg:
                            stats["segments_not_enriched"] += 1
                            continue
                        stats["segments_seen"] += 1

                        n_tpl = seg.get("n_templated", 0)
                        n_rev = seg.get("n_revealing", 0)
                        tpl_text = seg.get("templated_text") or ""
                        rev_text = seg.get("revealing_text") or ""

                        want_tpl = (
                            not full(templated_pool)
                            and tpl_text
                            and len(_INFORMATIVE_RE.findall(tpl_text)) >= min_span_chars
                        )
                        want_rev = (
                            not full(revealing_pool)
                            and rev_text
                            and len(_INFORMATIVE_RE.findall(rev_text)) >= min_span_chars
                        )
                        if not (want_tpl or want_rev):
                            continue

                        seg_start = seg.get("seg_start_in_region", 0)
                        seg_end = seg.get("seg_end_in_region", 0)
                        section = seg.get("section")
                        is_pure_templated = (n_rev == 0 and n_tpl > 0)
                        is_pure_revealing = (n_tpl == 0 and n_rev > 0)

                        # Slice small context here; don't retain full region.
                        segment_text = region_text[seg_start:seg_end]
                        cb = region_text[max(0, seg_start - context_chars):seg_start] if context_chars else ""
                        ca = region_text[seg_end:seg_end + context_chars] if context_chars else ""

                        base_meta = {
                            "dx": dx,
                            "record_uid": record_uid,
                            "region_idx": region_idx,
                            "segment_idx": seg_idx,
                            "section": section,
                            "checkpoint": ckpt_tag,
                            "prior": prior,
                            "probe": probe,
                            "n_templated": n_tpl,
                            "n_revealing": n_rev,
                            "blanket_section": bool(seg.get("blanket_section")),
                            "rule_matched": _rule_summary(seg),
                            "segment_text": segment_text,
                            "context_before": cb,
                            "context_after": ca,
                        }

                        if want_tpl and ((not contiguous_only) or is_pure_templated):
                            item = dict(base_meta)
                            item["predicted_label"] = "templated"
                            item["span_text"] = tpl_text
                            item["is_pure_segment"] = is_pure_templated
                            templated_pool.append(item)

                        if want_rev and ((not contiguous_only) or is_pure_revealing):
                            item = dict(base_meta)
                            item["predicted_label"] = "revealing"
                            item["span_text"] = rev_text
                            item["is_pure_segment"] = is_pure_revealing
                            revealing_pool.append(item)

    return templated_pool, revealing_pool, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-tag", dest="ckpt_tag", default=config.DEFAULT_CHECKPOINT,
                    help="Results subdirectory name, e.g. 'final' or 'checkpoint-14178'")
    ap.add_argument("--priors", nargs="+", default=DEFAULT_PRIORS,
                    choices=_priors.PRIOR_TIERS)
    ap.add_argument("--probes", nargs="+", default=DEFAULT_PROBES,
                    choices=_priors.PROBE_TYPES)
    ap.add_argument("--n-per-class", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42,
                     help="(Unused in first-N mode; kept for backward compatibility.)")
    ap.add_argument("--min-span-chars", type=int, default=3,
                     help="Minimum informative (alnum) characters for a span to be eligible.")
    ap.add_argument("--context-chars", type=int, default=300,
                     help="Characters of surrounding context shown on each side of the segment.")
    ap.add_argument("--contiguous-only", action="store_true",
                     help="Only sample segments that are entirely one label (span == full "
                          "segment, guaranteed contiguous). Excludes mixed segments.")
    ap.add_argument("--out-dir", default="analysis/manual_review")
    args = ap.parse_args()

    target_priors = set(args.priors)
    target_probes = set(args.probes)

    matched = list(iter_target_files(args.ckpt_tag, target_priors, target_probes))
    print(f"[scan] {len(matched)} files match scope "
          f"(ckpt_tag={args.ckpt_tag}, priors={sorted(target_priors)}, probes={sorted(target_probes)})")
    if not matched:
        print("[scan] nothing matched -- check --ckpt-tag/--priors/--probes and RESULTS_DIR")
        return

    print("[collect] reading pipeline-written classification (first-N, early-stop)...")
    templated_pool, revealing_pool, stats = collect_pools(
        matched, args.ckpt_tag, args.min_span_chars, args.contiguous_only,
        args.n_per_class, args.context_chars)
    print(f"[collect] segments read: {stats['segments_seen']:,}  "
          f"templated items: {len(templated_pool):,}  revealing items: {len(revealing_pool):,}")
    if stats["segments_not_enriched"] or stats["regions_not_enriched"]:
        print(f"[collect] WARNING: skipped {stats['regions_not_enriched']:,} un-enriched region(s) "
              f"and {stats['segments_not_enriched']:,} un-enriched segment(s). "
              f"Did content_classification.py --apply finish for this scope?")

    for pool, label in ((templated_pool, "templated"), (revealing_pool, "revealing")):
        if len(pool) < args.n_per_class:
            print(f"[sample] WARNING: only found {len(pool)} {label} items in the whole "
                  f"scope, requested {args.n_per_class}. Using all of them.")

    # First-N: keep the first n_per_class of each class, in the order the
    # corpus yielded them. Interleave templated/revealing for review order.
    templated_sample = templated_pool[:args.n_per_class]
    revealing_sample = revealing_pool[:args.n_per_class]

    combined = []
    for t, r in zip(templated_sample, revealing_sample):
        combined.append(t)
        combined.append(r)
    # append any leftover if the two classes are unequal length
    extra = templated_sample[len(revealing_sample):] + revealing_sample[len(templated_sample):]
    combined.extend(extra)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    blind_path = out_dir / "review_blind.csv"
    key_path = out_dir / "review_key.csv"

    with open(blind_path, "w", newline="") as f_blind, open(key_path, "w", newline="") as f_key:
        blind_writer = csv.DictWriter(
            f_blind,
            fieldnames=["item_id", "context_before", "segment_text", "span_text",
                        "context_after", "human_label", "reviewer_notes"],
        )
        blind_writer.writeheader()

        key_writer = csv.DictWriter(
            f_key,
            fieldnames=["item_id", "predicted_label", "is_pure_segment", "rule_matched",
                        "section", "blanket_section", "n_templated", "n_revealing", "dx",
                        "record_uid", "region_idx", "segment_idx", "checkpoint", "prior", "probe"],
        )
        key_writer.writeheader()

        for i, item in enumerate(combined):
            item_id = f"item_{i:04d}"
            blind_writer.writerow({
                "item_id": item_id,
                "context_before": item["context_before"],
                "segment_text": item["segment_text"],
                "span_text": item["span_text"],
                "context_after": item["context_after"],
                "human_label": "",
                "reviewer_notes": "",
            })
            key_writer.writerow({
                "item_id": item_id,
                "predicted_label": item["predicted_label"],
                "is_pure_segment": item["is_pure_segment"],
                "rule_matched": item["rule_matched"],
                "section": item["section"],
                "blanket_section": item["blanket_section"],
                "n_templated": item["n_templated"],
                "n_revealing": item["n_revealing"],
                "dx": item["dx"],
                "record_uid": item["record_uid"],
                "region_idx": item["region_idx"],
                "segment_idx": item["segment_idx"],
                "checkpoint": item["checkpoint"],
                "prior": item["prior"],
                "probe": item["probe"],
            })

    print()
    print("=" * 72)
    print(f"[done] wrote {len(combined)} items ({len(templated_sample)} templated, "
          f"{len(revealing_sample)} revealing)")
    print(f"  reviewer file (fill this in): {blind_path}")
    print(f"  scoring key   (keep aside)  : {key_path}")
    print()
    print("  Reviewer: in review_blind.csv, judge `span_text` (using segment_text +")
    print("  context for reference) and fill `human_label` with exactly 'templated'")
    print("  or 'revealing' for every row. Then:")
    print(f"    python score_review.py --blind {blind_path} --key {key_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
