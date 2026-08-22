"""
Every token in a segment is classified as either templated or
revealing. 

Classification rules:
  1. Alphanumeric tokens: classified by regex span overlap with one of
     the templated rules. Templated if any overlap; revealing otherwise.
  2. Whitespace-only and pure-punctuation tokens are first classified
     by overlap too. Anything still unclassified inherits from its
     nearest classified neighbor (forward first, then backward).
     If a segment is entirely filler, default to templated.
  3. Blanket sections (BLANKET_TEMPLATED_SECTIONS from headers.py):
     all tokens forced to templated.
     
Requires `segments` to already exist (run find_span_headers.py first).
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import config  # noqa: E402

KNOWN_HEADERS = config.KNOWN_HEADERS

def _build_header_prefix_regex():
    parts = []
    for h in sorted(set(KNOWN_HEADERS), key=lambda x: -len(x)):
        pat = r"\s+".join(re.escape(t) for t in h.split())
        parts.append(pat)
    alt = "|".join(parts)
    return re.compile(
        rf"(?ix)(?:^|(?<=\n))\s*(?:{alt})\s*:",
        re.DOTALL,
    )


HEADER_PREFIX_RE = _build_header_prefix_regex()

TEMPLATED_ARTIFACT_RULES = config.TEMPLATED_ARTIFACT_RULES


def _is_informative_piece(piece):
    return any(c.isalnum() for c in piece)


def find_templated_spans(text):
    if not text:
        return []
    spans = []
    for m in HEADER_PREFIX_RE.finditer(text):
        spans.append((m.start(), m.end(), "HEADER_PREFIX"))
    for rule_name, rule_re in TEMPLATED_ARTIFACT_RULES.items():
        for m in rule_re.finditer(text):
            spans.append((m.start(), m.end(), rule_name))
    return spans


def classify_tokens(text, spans, tokenizer):
    if not text:
        return []
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    if not spans:
        return [(s, e, None) for (s, e) in offsets if e > s]
    spans_sorted = sorted(spans, key=lambda x: (x[0], x[1]))
    out = []
    j = 0
    for tok_s, tok_e in offsets:
        if tok_e <= tok_s:
            continue
        while j < len(spans_sorted) and spans_sorted[j][1] <= tok_s:
            j += 1
        matched = None
        k = j
        while k < len(spans_sorted) and spans_sorted[k][0] < tok_e:
            if spans_sorted[k][1] > tok_s:
                matched = spans_sorted[k][2]
                break
            k += 1
        out.append((tok_s, tok_e, matched))
    return out


def _classify_with_inheritance(text, tok_list):
    if not tok_list:
        return []
    classified = []
    for (ts, te, rule) in tok_list:
        piece = text[ts:te]
        informative = _is_informative_piece(piece)
        if informative:
            is_tpl = (rule is not None)
        else:
            is_tpl = True if rule is not None else None
        classified.append({
            "start":        ts,
            "end":          te,
            "piece":        piece,
            "rule":         rule,
            "is_templated": is_tpl,
            "_informative": informative,
        })
    if not any(t["_informative"] for t in classified):
        for t in classified:
            t["is_templated"] = True
        for t in classified:
            del t["_informative"]
        return classified
    last_classified = None
    for tok in classified:
        if tok["is_templated"] is not None:
            last_classified = tok["is_templated"]
        else:
            if last_classified is not None:
                tok["is_templated"] = last_classified
    next_classified = None
    for tok in reversed(classified):
        if tok["is_templated"] is not None:
            next_classified = tok["is_templated"]
        else:
            if next_classified is not None:
                tok["is_templated"] = next_classified
    for tok in classified:
        del tok["_informative"]
    return classified


# Module-level globals each worker initializes once via the pool initializer.
_WORKER_TOK = None
_WORKER_APPLY = False
_WORKER_FORCE = False
_WORKER_BLANKET_SECTIONS = None


def _worker_init(tokenizer_path, apply_flag, force_flag, blanket_sections):
    """Pool initializer — load tokenizer + globals once per worker."""
    global _WORKER_TOK, _WORKER_APPLY, _WORKER_FORCE, _WORKER_BLANKET_SECTIONS
    from transformers import AutoTokenizer
    _WORKER_TOK = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    _WORKER_APPLY = apply_flag
    _WORKER_FORCE = force_flag
    _WORKER_BLANKET_SECTIONS = blanket_sections


def _process_file_worker(args_tuple):
    """One file → updates JSONL in place (if apply) + returns shard audit path + stats."""
    import json
    from collections import Counter
    from pathlib import Path

    fp_str, prior, audit_dir_str, worker_idx = args_tuple
    fp = Path(fp_str)
    audit_dir = Path(audit_dir_str) if audit_dir_str else None

    tok = _WORKER_TOK
    apply_flag = _WORKER_APPLY
    force_flag = _WORKER_FORCE
    BLANKET_TEMPLATED_SECTIONS = _WORKER_BLANKET_SECTIONS

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

    dx = fp.parent.name

    # We always emit audit rows for newly-processed regions. For already-done
    # regions, we skip them entirely (no audit row, no re-tokenize).
    audit_shard_path = None
    audit_fh = None
    if apply_flag and audit_dir is not None:
        audit_shard_path = audit_dir / f"_shard_prior-{prior}_worker-{worker_idx:03d}_{fp.name}.jsonl"
        audit_fh = open(audit_shard_path, "w")

    stats = {
        "records_skipped_non_train":    0,
        "records_skipped_no_attr":      0,
        "records_skipped_no_segments":  0,
        "records_skipped_fully_done":   0,
        "regions_skipped_done":         0,
        "records_processed":            0,
        "total_regions":                0,
        "total_segments_enriched":      0,
        "total_tokens":                 0,
        "total_templated_tokens":       0,
        "rule_token_totals":            Counter(),
        "section_token_totals":         Counter(),
        "section_templated_totals":     Counter(),
        "blanket_segment_count":        0,
        "audit_shard_path":             str(audit_shard_path) if audit_shard_path else None,
        "audit_shard_prior":            prior,
        "file_changed":                 False,
        "file_path":                    str(fp),
    }

    n_changed_in_file = 0
    for line_idx, rec in enumerate(records):
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
        patient_id = rec.get("patient_id") or ""
        record_uid = f"{fp.name}:{line_idx}"

        has_segments = any(("segments" in r) for r in regions)
        if not has_segments:
            stats["records_skipped_no_segments"] += 1
            continue

        # Quick check: are ALL regions already enriched (have n_tokens)?
        # If so, skip the whole record so we don't bother tokenizing.
        if not force_flag and all(("n_tokens" in r) for r in regions):
            stats["records_skipped_fully_done"] += 1
            continue

        record_touched = False
        for region_idx, r in enumerate(regions):
            if not force_flag and "n_tokens" in r:
                stats["regions_skipped_done"] += 1
                # Still count totals so cross-run aggregates stay consistent.
                stats["total_regions"]          += 1
                stats["total_tokens"]           += r.get("n_tokens", 0)
                stats["total_templated_tokens"] += r.get("n_templated_tokens", 0)
                for seg in r.get("segments") or []:
                    if "n_tokens" not in seg:
                        continue
                    n_total = seg.get("n_tokens", 0)
                    n_tpl   = seg.get("n_templated", 0)
                    sec_key = seg.get("section") or "(no_header)"
                    stats["total_segments_enriched"] += 1
                    stats["section_token_totals"][sec_key] += n_total
                    stats["section_templated_totals"][sec_key] += n_tpl
                    for rule, ct in (seg.get("rule_counts") or {}).items():
                        stats["rule_token_totals"][rule] += ct
                    if seg.get("blanket_section"):
                        stats["blanket_segment_count"] += 1
                continue

            region_text = r.get("region_text") or ""
            if not region_text:
                r["templated_spans"]    = []
                r["n_tokens"]           = 0
                r["n_templated_tokens"] = 0
                r["n_revealing_tokens"] = 0
                record_touched = True
                continue

            spans_region = find_templated_spans(region_text)
            tok_list_region = classify_tokens(region_text, spans_region, tok)
            classified_region = _classify_with_inheritance(region_text, tok_list_region)
            n_tok_region = len(classified_region)
            n_tpl_region = sum(1 for t in classified_region if t["is_templated"])
            n_rev_region = n_tok_region - n_tpl_region

            r["templated_spans"] = [
                {"start": s, "end": e, "rule": rule}
                for (s, e, rule) in spans_region
            ]
            r["n_tokens"]            = n_tok_region
            r["n_templated_tokens"]  = n_tpl_region
            r["n_revealing_tokens"]  = n_rev_region
            record_touched = True

            stats["total_regions"]          += 1
            stats["total_tokens"]           += n_tok_region
            stats["total_templated_tokens"] += n_tpl_region

            segments = r.get("segments") or []
            for seg_idx, seg in enumerate(segments):
                seg_start = seg.get("seg_start_in_region", 0)
                seg_end   = seg.get("seg_end_in_region", 0)
                section   = seg.get("section")
                seg_text  = region_text[seg_start:seg_end]
                if not seg_text:
                    continue

                seg_spans = find_templated_spans(seg_text)
                seg_tok_list = classify_tokens(seg_text, seg_spans, tok)
                classified_seg = _classify_with_inheritance(seg_text, seg_tok_list)

                rule_counts = Counter()
                tpl_pieces = []
                rev_pieces = []
                for tok_d in classified_seg:
                    if tok_d["is_templated"]:
                        tpl_pieces.append(tok_d["piece"])
                        if tok_d["rule"] is not None:
                            rule_counts[tok_d["rule"]] += 1
                    else:
                        rev_pieces.append(tok_d["piece"])

                n_total = len(classified_seg)
                n_tpl = sum(1 for t in classified_seg if t["is_templated"])
                n_rev = n_total - n_tpl

                blanket = (section in BLANKET_TEMPLATED_SECTIONS)
                if blanket:
                    tpl_pieces = tpl_pieces + rev_pieces
                    rev_pieces = []
                    n_tpl = n_total
                    n_rev = 0
                    stats["blanket_segment_count"] += 1

                rule_matched = {}
                for (sp_s, sp_e, sp_rule) in seg_spans:
                    rule_matched.setdefault(sp_rule, []).append(seg_text[sp_s:sp_e])

                seg["n_tokens"]        = n_total
                seg["n_templated"]     = n_tpl
                seg["n_revealing"]     = n_rev
                seg["templated_text"]  = "".join(tpl_pieces)
                seg["revealing_text"]  = "".join(rev_pieces)
                seg["rule_counts"]     = dict(rule_counts)
                seg["rule_matched"]    = rule_matched
                seg["blanket_section"] = blanket

                stats["total_segments_enriched"] += 1
                sec_key = section or "(no_header)"
                stats["section_token_totals"][sec_key] += n_total
                stats["section_templated_totals"][sec_key] += n_tpl
                for rule, ct in rule_counts.items():
                    stats["rule_token_totals"][rule] += ct

                if audit_fh is not None:
                    audit_row = {
                        "patient_id":     patient_id,
                        "dx":             dx,
                        "record_uid":     record_uid,
                        "region_idx":     region_idx,
                        "segment_idx":    seg_idx,
                        "section":        section,
                        "blanket":        blanket,
                        "segment_text":   seg_text,
                        "templated_text": seg["templated_text"],
                        "revealing_text": seg["revealing_text"],
                        "rule_counts":    dict(rule_counts),
                        "n_tokens":       n_total,
                        "n_templated":    n_tpl,
                        "n_revealing":    n_rev,
                    }
                    audit_fh.write(json.dumps(audit_row) + "\n")

        if record_touched:
            stats["records_processed"] += 1
            n_changed_in_file += 1

    if audit_fh is not None:
        audit_fh.close()

    if apply_flag and n_changed_in_file > 0:
        with open(fp, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        stats["file_changed"] = True

    # Convert Counter -> dict for pickling
    stats["rule_token_totals"]        = dict(stats["rule_token_totals"])
    stats["section_token_totals"]     = dict(stats["section_token_totals"])
    stats["section_templated_totals"] = dict(stats["section_templated_totals"])

    return stats


def _cli_main():
    import argparse
    import multiprocessing as mp
    import time
    from collections import Counter
    from pathlib import Path

    BLANKET_TEMPLATED_SECTIONS = config.BLANKET_TEMPLATED_SECTIONS

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "1_priors"))
    import build_prompt_from_priors as _priors

    RESULTS_DIR = config.RESULTS_DIR
    AUDIT_DIR = config.AUDIT_DIR

    TOKENIZER_PATH = str(config.TOKENIZER_PATH)

    DEFAULT_PRIORS      = _priors.PRIOR_TIERS
    DEFAULT_PROBES      = list(_priors.PROBE_TYPES)
    ALL_PROBES          = set(_priors.PROBE_TYPES)

    FNAME_RE      = re.compile(r"^(?P<prior>[a-z_]+)__(?P<probe>[a-z_]+)\.jsonl$")

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

    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="actually write back (default: dry run)")
    ap.add_argument("--force", action="store_true",
                    help="Re-tokenize regions that already have `n_tokens` "
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

    if args.apply:
        AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    # Sort by size desc for better load balance
    matched_sorted = sorted(matched, key=lambda x: -Path(x[0]).stat().st_size)
    n_workers = max(1, min(args.workers, len(matched_sorted)))

    audit_dir_arg = str(AUDIT_DIR) if args.apply else ""
    worker_args = [
        (str(fp), prior, audit_dir_arg, i)
        for i, (fp, prior, probe) in enumerate(matched_sorted)
    ]

    print(f"[parallel] launching {n_workers} worker(s) for {len(worker_args)} files")
    print(f"[parallel] loading tokenizer in each worker: {TOKENIZER_PATH}")

    t0 = time.time()
    with mp.Pool(
        processes=n_workers,
        initializer=_worker_init,
        initargs=(TOKENIZER_PATH, args.apply, args.force, BLANKET_TEMPLATED_SECTIONS),
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
        "records_processed":            0,
        "records_skipped_non_train":    0,
        "records_skipped_no_attr":      0,
        "records_skipped_no_segments":  0,
        "records_skipped_fully_done":   0,
        "regions_skipped_done":         0,
        "total_regions":                0,
        "total_segments_enriched":      0,
        "total_tokens":                 0,
        "total_templated_tokens":       0,
        "blanket_segment_count":        0,
    }
    rule_token_totals = Counter()
    section_token_totals = Counter()
    section_templated_totals = Counter()

    for s in all_stats:
        for k in agg:
            agg[k] += s[k]
        for k, v in s["rule_token_totals"].items():
            rule_token_totals[k] += v
        for k, v in s["section_token_totals"].items():
            section_token_totals[k] += v
        for k, v in s["section_templated_totals"].items():
            section_templated_totals[k] += v

    # Merge audit shards into per-prior audit files.
    # NOTE: In incremental mode, shards only contain audit rows for the
    # regions that were newly processed. Merging will overwrite any prior
    # audit file for the same (ckpt, prior). If you want the cumulative
    # audit, use --force or merge shards yourself.
    audit_paths = {}
    if args.apply:
        from collections import defaultdict
        shards_by_prior = defaultdict(list)
        for s in all_stats:
            if s["audit_shard_path"]:
                shards_by_prior[s["audit_shard_prior"]].append(s["audit_shard_path"])

        for prior, shards in shards_by_prior.items():
            final = AUDIT_DIR / f"audit_segments_{args.ckpt_tag}_prior-{prior}.jsonl"
            # In incremental mode (not --force), APPEND to existing audit so
            # past rows are preserved. With --force, overwrite from scratch.
            mode = "w" if args.force else "a"
            with open(final, mode) as out_f:
                for shard_path in sorted(shards):
                    if Path(shard_path).exists():
                        with open(shard_path) as in_f:
                            for line in in_f:
                                out_f.write(line)
                        Path(shard_path).unlink()
            audit_paths[prior] = final

    print()
    print("=" * 72)
    print(f"[done] in {elapsed:.0f}s (parallel, {n_workers} workers)")
    print(f"  files in scope                : {len(matched)}")
    print(f"  files changed                 : {files_changed}")
    print(f"  train records processed       : {agg['records_processed']:,}")
    print(f"  skipped (non-train)           : {agg['records_skipped_non_train']:,}")
    print(f"  skipped (no source_attribution): {agg['records_skipped_no_attr']:,}")
    print(f"  skipped (no segments yet)     : {agg['records_skipped_no_segments']:,}")
    print(f"  skipped (all regions done)    : {agg['records_skipped_fully_done']:,}")
    print(f"  regions skipped (already done): {agg['regions_skipped_done']:,}")
    print(f"  total regions enriched        : {agg['total_regions']:,}")
    print(f"  total segments enriched       : {agg['total_segments_enriched']:,}")
    print(f"  blanket segments (forced tpl) : {agg['blanket_segment_count']:,}")
    print(f"  total tokens                  : {agg['total_tokens']:,}")
    print(f"  total templated tokens        : {agg['total_templated_tokens']:,}  "
          f"({100*agg['total_templated_tokens']/max(agg['total_tokens'],1):.1f}%)")
    if not args.apply:
        print()
        print(f"  DRY RUN — pass --apply to write back and create audit JSONLs")
    print("=" * 72)
    print()
    print(f"=== Per-rule token coverage (regex hits only) ===")
    print(f"  {'rule':<18}  {'tokens':>10}  {'% of all':>9}")
    for rule, ct in rule_token_totals.most_common():
        print(f"  {rule:<18}  {ct:>10,}  "
              f"{100*ct/max(agg['total_tokens'],1):>8.1f}%")
    print()
    print(f"=== Top sections (total + templated tokens) ===")
    print(f"  {'section':<50}  {'total':>10}  {'templated':>10}  {'%tpl':>5}")
    for sec, ct in section_token_totals.most_common(30):
        tpl_ct = section_templated_totals.get(sec, 0)
        pct = 100 * tpl_ct / max(ct, 1)
        print(f"  {sec[:50]:<50}  {ct:>10,}  {tpl_ct:>10,}  {pct:>4.0f}%")
    print()
    if args.apply and audit_paths:
        print(f"=== Audit JSONLs written ===")
        for prior, path in audit_paths.items():
            print(f"  {path}  ({'overwritten' if args.force else 'appended'})")


if __name__ == "__main__":
    _cli_main()
