#!/usr/bin/env python3
"""
For every record:
  1. Build a "memorized mask" by .find()-ing every memorized_span in the
     generation; collapse to maximal CONTIGUOUS regions.
  2. For each region, search ALL patient notes in the entire training
     corpus for VERBATIM occurrences. Records EVERY hit as
     (patient_id, encounter_date, row_idx, position).
  3. Pick a "winning" within-patient source note for section attribution --> most recent
  4. Apply post-hoc splitting to unattributed regions.

Output written to "source_attribution" on each record. Each region has:
  - gen_start, gen_end, region_text, region_chars
  - attributed (bool), split_origin
  - winning_source_note_text  (FULL note text; for section lookup)
  - winning_source_note_key   {patient_id, encounter_date, row_idx}
  - winning_source_note_pos   char offset in that note
  - corpus_occurrences        flat list of all corpus hits
  - K_within_patient          distinct row_idxs for this patient
  - K_corpus                  distinct patient_ids in corpus

"""

import argparse
import json
import multiprocessing as mp
import re
import time
from collections import defaultdict
from pathlib import Path

import ahocorasick
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "1_priors"))
import build_prompt_from_priors as _priors  # noqa: E402

CORPUS_PATH = config.TRAIN_COHORT_PATH
RESULTS_DIR = config.RESULTS_DIR

DEFAULT_PRIORS      = _priors.PRIOR_TIERS
DEFAULT_PROBES      = list(_priors.PROBE_TYPES)
ALL_PROBES          = set(_priors.PROBE_TYPES)

FNAME_RE      = re.compile(r"^(?P<prior>[a-z_]+)__(?P<probe>[a-z_]+)\.jsonl$")

MIN_TEXT_LEN          = 30
MIN_SPLIT_PIECE_CHARS = 30


def parse_filename(path):
    m = FNAME_RE.match(path.name)
    if not m:
        return None
    return m.group("prior"), m.group("probe")


def iter_target_files(ckpt_tag, target_priors, target_probes):
    ckpt_dir = RESULTS_DIR / ckpt_tag
    if not ckpt_dir.exists():
        return
    for dx_dir in sorted(p for p in ckpt_dir.iterdir() if p.is_dir()):
        for fp in sorted(dx_dir.glob("*.jsonl")):
            parsed = parse_filename(fp)
            if parsed is None:
                continue
            prior, probe = parsed
            if prior in target_priors and probe in target_probes:
                yield fp


def collect_required_patients(matched_files, force):
    needed = set()
    n_records = 0
    n_eligible = 0
    n_already_done = 0
    for path in matched_files:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                n_records += 1
                if rec.get("error_type") or not rec.get("generation"):
                    continue
                if not rec.get("memorized_spans"):
                    continue
                # Count already-done records separately so we can report them
                if not force and isinstance(rec.get("source_attribution"), dict):
                    n_already_done += 1
                    continue
                pid = rec.get("patient_id")
                if pid:
                    needed.add(pid)
                    n_eligible += 1
    return needed, n_records, n_eligible, n_already_done


def load_corpus(needed_patients):
    print(f"[pass 2] streaming corpus {CORPUS_PATH}", flush=True)
    by_patient = defaultdict(list)
    all_notes = []
    t0 = time.time()
    n_total = 0
    n_kept_for_attribution = 0
    with open(CORPUS_PATH) as f:
        for row_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_total += 1
            if n_total % 200_000 == 0:
                el = time.time() - t0
                print(f"  [pass 2] {n_total:,} rows scanned, "
                      f"{n_kept_for_attribution:,} kept for in-patient "
                      f"attribution, {el:.0f}s", flush=True)
            pid = row.get("PatientUid")
            if not pid:
                continue
            note_text = row.get("Note") or ""
            if not note_text:
                continue
            enc_date = row.get("Encounter_Date") or ""
            all_notes.append((pid, enc_date, note_text, row_idx))
            if pid in needed_patients:
                by_patient[pid].append((enc_date, note_text, row_idx))
                n_kept_for_attribution += 1
    elapsed = time.time() - t0
    print(f"[pass 2] done: {n_total:,} rows scanned, "
          f"{n_kept_for_attribution:,} in-patient notes kept, "
          f"{len(all_notes):,} total notes, "
          f"{elapsed:.0f}s", flush=True)
    return dict(by_patient), all_notes


def build_memorized_mask(generation, raw_spans):
    if not generation or not raw_spans:
        return []
    intervals = []
    for s in raw_spans:
        span_text = s.get("text", "") if isinstance(s, dict) else (s or "")
        span_text = span_text.strip()
        if not span_text:
            continue
        start = 0
        while True:
            idx = generation.find(span_text, start)
            if idx < 0:
                break
            intervals.append((idx, idx + len(span_text)))
            start = idx + 1
    if not intervals:
        return []
    intervals.sort()
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s < merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def find_in_patient_notes(region_text, patient_notes):
    if not region_text:
        return []
    matches = []
    for enc_date, note_text, row_idx in patient_notes:
        positions = []
        start = 0
        while True:
            idx = note_text.find(region_text, start)
            if idx < 0:
                break
            positions.append(idx)
            start = idx + 1
        if not positions:
            continue
        matches.append({
            "note_key":   {"encounter_date": enc_date, "row_idx": row_idx},
            "note_text":  note_text,
            "positions":  positions,
        })
    return matches


def pick_winning_note(matches):
    if not matches:
        return None
    return sorted(matches,
                  key=lambda m: (m["note_key"]["encounter_date"],
                                  m["note_key"]["row_idx"]),
                  reverse=True)[0]


def _longest_prefix_in_notes(region_text, patient_notes):
    if not region_text:
        return 0, None
    note_texts = [n[1] for n in patient_notes]
    if not any(region_text[0] in nt for nt in note_texts):
        return 0, None
    lo, hi = 1, len(region_text)
    best_k, best_idx = 0, None
    while lo <= hi:
        mid = (lo + hi) // 2
        probe = region_text[:mid]
        hit_idx = None
        for i, nt in enumerate(note_texts):
            if probe in nt:
                hit_idx = i
                break
        if hit_idx is not None:
            best_k, best_idx = mid, hit_idx
            lo = mid + 1
        else:
            hi = mid - 1
    return best_k, best_idx


def _longest_suffix_in_notes(region_text, patient_notes):
    if not region_text:
        return 0, None
    note_texts = [n[1] for n in patient_notes]
    if not any(region_text[-1] in nt for nt in note_texts):
        return 0, None
    lo, hi = 1, len(region_text)
    best_k, best_idx = 0, None
    while lo <= hi:
        mid = (lo + hi) // 2
        probe = region_text[-mid:]
        hit_idx = None
        for i, nt in enumerate(note_texts):
            if probe in nt:
                hit_idx = i
                break
        if hit_idx is not None:
            best_k, best_idx = mid, hit_idx
            lo = mid + 1
        else:
            hi = mid - 1
    return best_k, best_idx


def _build_region_row(gs, ge, region_text, in_patient_matches,
                       corpus_occurrences, this_pid, split_origin):
    region_chars = ge - gs
    attributed = bool(in_patient_matches)

    if attributed:
        winning = pick_winning_note(in_patient_matches)
        win_pos = winning["positions"][0]
        winning_source_note_text = winning["note_text"]
        winning_source_note_key  = winning["note_key"]
        winning_source_note_pos  = win_pos
    else:
        winning_source_note_text = None
        winning_source_note_key  = None
        winning_source_note_pos  = None

    distinct_patients = {o["patient_id"] for o in corpus_occurrences}
    in_patient_rows   = {o["row_idx"] for o in corpus_occurrences
                          if o["patient_id"] == this_pid}
    K_corpus         = len(distinct_patients)
    K_within_patient = len(in_patient_rows)

    return {
        "gen_start":                  gs,
        "gen_end":                    ge,
        "region_text":                region_text,
        "region_chars":               region_chars,
        "attributed":                 attributed,
        "split_origin":               split_origin,
        "winning_source_note_key":    winning_source_note_key,
        "winning_source_note_pos":    winning_source_note_pos,
        "winning_source_note_text":   winning_source_note_text,
        "corpus_occurrences":         corpus_occurrences,
        "K_within_patient":           K_within_patient,
        "K_corpus":                   K_corpus,
    }


def split_and_attribute(gs, ge, generation, patient_notes,
                         this_pid,
                         get_corpus_occurrences,
                         split_depth=0, max_depth=4):
    region_text = generation[gs:ge]
    region_chars = ge - gs

    in_patient = find_in_patient_notes(region_text, patient_notes)
    corpus = get_corpus_occurrences(region_text)
    if in_patient:
        return [_build_region_row(
            gs, ge, region_text, in_patient, corpus, this_pid,
            split_origin=("original" if split_depth == 0 else "split_attributed")
        )]

    if split_depth >= max_depth or region_chars < MIN_SPLIT_PIECE_CHARS:
        last_chance = find_in_patient_notes(region_text, patient_notes)
        if last_chance:
            return [_build_region_row(
                gs, ge, region_text, last_chance, corpus, this_pid,
                split_origin=("original" if split_depth == 0 else "split_attributed_tiny")
            )]
        return [_build_region_row(
            gs, ge, region_text, [], corpus, this_pid,
            split_origin=("original" if split_depth == 0 else "split_residual")
        )]

    prefix_k, _ = _longest_prefix_in_notes(region_text, patient_notes)
    suffix_k, _ = _longest_suffix_in_notes(region_text, patient_notes)

    if prefix_k + suffix_k > region_chars:
        suffix_k = region_chars - prefix_k

    if prefix_k == 0 and suffix_k == 0:
        return [_build_region_row(
            gs, ge, region_text, [], corpus, this_pid,
            split_origin=("original" if split_depth == 0 else "split_residual")
        )]

    if prefix_k >= region_chars or suffix_k >= region_chars:
        return [_build_region_row(
            gs, ge, region_text, [], corpus, this_pid,
            split_origin="split_residual"
        )]

    pieces = []

    if prefix_k >= MIN_SPLIT_PIECE_CHARS:
        prefix_text = region_text[:prefix_k]
        prefix_in_patient = find_in_patient_notes(prefix_text, patient_notes)
        prefix_corpus = get_corpus_occurrences(prefix_text)
        pieces.append(_build_region_row(
            gs, gs + prefix_k, prefix_text, prefix_in_patient,
            prefix_corpus, this_pid,
            split_origin=("split_attributed" if prefix_in_patient else "split_residual")
        ))
    elif prefix_k > 0:
        prefix_k = 0

    suffix_piece = None
    if suffix_k >= MIN_SPLIT_PIECE_CHARS:
        suffix_text = region_text[region_chars - suffix_k:]
        suffix_in_patient = find_in_patient_notes(suffix_text, patient_notes)
        suffix_corpus = get_corpus_occurrences(suffix_text)
        suffix_piece = _build_region_row(
            ge - suffix_k, ge, suffix_text, suffix_in_patient,
            suffix_corpus, this_pid,
            split_origin=("split_attributed" if suffix_in_patient else "split_residual")
        )
    elif suffix_k > 0:
        suffix_k = 0

    mid_start = gs + prefix_k
    mid_end   = ge - suffix_k
    if mid_end > mid_start:
        if mid_end - mid_start >= MIN_SPLIT_PIECE_CHARS:
            pieces.extend(split_and_attribute(
                mid_start, mid_end, generation, patient_notes, this_pid,
                get_corpus_occurrences,
                split_depth=split_depth + 1, max_depth=max_depth
            ))
        else:
            mid_text = generation[mid_start:mid_end]
            mid_in_patient = find_in_patient_notes(mid_text, patient_notes)
            mid_corpus = get_corpus_occurrences(mid_text)
            if mid_in_patient:
                pieces.append(_build_region_row(
                    mid_start, mid_end, mid_text, mid_in_patient,
                    mid_corpus, this_pid,
                    split_origin="split_attributed_tiny"
                ))
            else:
                pieces.append(_build_region_row(
                    mid_start, mid_end, mid_text, [],
                    mid_corpus, this_pid,
                    split_origin="split_residual_tiny"
                ))

    if suffix_piece is not None:
        pieces.append(suffix_piece)

    return pieces

_NOTES_BY_PATIENT = None
_ALL_NOTES        = None
_HITS_BY_IDX      = None
_TEXT_TO_IDX      = None
_APPLY            = False
_FORCE            = False


def _worker_init(notes_by_patient, all_notes, hits_by_idx, text_to_idx,
                 apply_flag, force_flag):
    """Pool initializer. On Linux fork, these are shared copy-on-write
    so memory isn't duplicated unless workers write. Worker code only
    reads, so this stays cheap."""
    global _NOTES_BY_PATIENT, _ALL_NOTES, _HITS_BY_IDX, _TEXT_TO_IDX
    global _APPLY, _FORCE
    _NOTES_BY_PATIENT = notes_by_patient
    _ALL_NOTES        = all_notes
    _HITS_BY_IDX      = hits_by_idx
    _TEXT_TO_IDX      = text_to_idx
    _APPLY            = apply_flag
    _FORCE            = force_flag


def _make_get_corpus_occurrences():
    """Closure over the worker's global hits_by_idx + text_to_idx + all_notes."""
    hits_by_idx = _HITS_BY_IDX
    text_to_idx = _TEXT_TO_IDX
    all_notes   = _ALL_NOTES

    def get_corpus_occurrences(region_text):
        ai = text_to_idx.get(region_text)
        if ai is not None:
            return list(hits_by_idx.get(ai, []))
        occs = []
        if not region_text:
            return occs
        for (cpid, c_enc_date, c_note, c_row_idx) in all_notes:
            start = 0
            while True:
                idx = c_note.find(region_text, start)
                if idx < 0:
                    break
                occs.append({
                    "patient_id":     cpid,
                    "encounter_date": c_enc_date,
                    "row_idx":        c_row_idx,
                    "position":       idx,
                })
                start = idx + 1
        return occs

    return get_corpus_occurrences


def _process_file_worker(fp_str):
    """One file → attribute every eligible record, write back."""
    fp = Path(fp_str)

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

    get_corpus_occurrences = _make_get_corpus_occurrences()
    notes_by_patient = _NOTES_BY_PATIENT
    force = _FORCE

    stats = {
        "file_path":              str(fp),
        "file_changed":           False,
        "records_changed":        0,
        "records_skipped_done":   0,
        "total_regions":          0,
        "total_attributed":       0,
        "total_unattributed":     0,
        "total_split_attributed": 0,
        "total_split_residual":   0,
    }

    n_changed_in_file = 0
    for rec in records:
        if rec.get("error_type") or not rec.get("generation"):
            continue
        if not rec.get("memorized_spans"):
            continue
        # Skip records that already have source_attribution unless --force.
        if not force and isinstance(rec.get("source_attribution"), dict):
            stats["records_skipped_done"] += 1
            continue

        pid = rec.get("patient_id")
        patient_notes = notes_by_patient.get(pid, [])
        generation = rec.get("generation") or ""
        raw_spans  = rec.get("memorized_spans") or []
        intervals = build_memorized_mask(generation, raw_spans)

        if not intervals or not patient_notes:
            attribution = {
                "n_regions":              0,
                "n_attributed":           0,
                "n_unattributed":         0,
                "n_split_attributed":     0,
                "n_split_residual":       0,
                "total_memorized_chars":  0,
                "regions":                [],
            }
            if _APPLY:
                rec["source_attribution"] = attribution
            n_changed_in_file += 1
            continue

        regions_out = []
        n_attributed = 0
        n_unattributed = 0
        n_split_attributed = 0
        n_split_residual = 0
        total_mem_chars = 0
        for (gs, ge) in intervals:
            total_mem_chars += (ge - gs)
            pieces = split_and_attribute(
                gs, ge, generation, patient_notes, pid,
                get_corpus_occurrences,
            )
            for p in pieces:
                regions_out.append(p)
                if p["attributed"]:
                    if p["split_origin"] in ("split_attributed", "split_attributed_tiny"):
                        n_split_attributed += 1
                    n_attributed += 1
                else:
                    if p["split_origin"] in ("split_residual", "split_residual_tiny"):
                        n_split_residual += 1
                    n_unattributed += 1

        attribution = {
            "n_regions":              len(regions_out),
            "n_attributed":           n_attributed,
            "n_unattributed":         n_unattributed,
            "n_split_attributed":     n_split_attributed,
            "n_split_residual":       n_split_residual,
            "total_memorized_chars":  total_mem_chars,
            "regions":                regions_out,
        }
        stats["total_regions"]          += len(regions_out)
        stats["total_attributed"]       += n_attributed
        stats["total_unattributed"]     += n_unattributed
        stats["total_split_attributed"] += n_split_attributed
        stats["total_split_residual"]   += n_split_residual

        if _APPLY:
            rec["source_attribution"] = attribution
        n_changed_in_file += 1

    if _APPLY and n_changed_in_file:
        with open(fp, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        stats["file_changed"] = True
    stats["records_changed"] = n_changed_in_file

    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="Reprocess records that already have "
                         "source_attribution (default: skip).")
    ap.add_argument("--ckpt-tag", dest="ckpt_tag", default=config.DEFAULT_CHECKPOINT,
                    help="Results subdirectory name, e.g. 'final' or 'checkpoint-14178'")
    ap.add_argument("--priors",      nargs="+", default=DEFAULT_PRIORS)
    ap.add_argument("--probes",      nargs="+", default=DEFAULT_PROBES,
                    choices=sorted(ALL_PROBES))
    ap.add_argument("--workers",     type=int, default=16,
                    help="Number of parallel worker processes for Pass 4 (default: 16)")
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

    print("[pass 1] collecting required patient_ids...", flush=True)
    needed, n_total, n_eligible, n_already_done = collect_required_patients(
        matched, args.force,
    )
    print(f"[pass 1] {n_eligible:,} eligible records (of {n_total:,}), "
          f"{n_already_done:,} already done (skipped), "
          f"{len(needed):,} unique patients needed")
    if not needed:
        print("[done] nothing to attribute "
              f"({'pass --force to redo' if n_already_done else 'no eligible records'})")
        return

    notes_by_patient, all_notes = load_corpus(needed)
    n_with_notes = sum(1 for p in needed if p in notes_by_patient)
    print(f"[pass 2] {n_with_notes:,} / {len(needed):,} patients found in corpus")
    print()

    # Pass 3a: collect unique region_texts
    # Only collect from records that will actually be processed in Pass 4.
    # This keeps the AC automaton small when most work is already done.
    print("[pass 3a] collecting region_texts for batch K-search...", flush=True)
    region_texts = set()
    t0 = time.time()
    for path in matched:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("error_type") or not rec.get("generation"):
                    continue
                if not rec.get("memorized_spans"):
                    continue
                # Skip already-done records unless --force
                if not args.force and isinstance(rec.get("source_attribution"), dict):
                    continue
                generation = rec.get("generation") or ""
                raw_spans  = rec.get("memorized_spans") or []
                intervals = build_memorized_mask(generation, raw_spans)
                for (gs, ge) in intervals:
                    txt = generation[gs:ge]
                    if len(txt) >= MIN_TEXT_LEN:
                        region_texts.add(txt)
    print(f"[pass 3a] {len(region_texts):,} unique whole-region texts, "
          f"{time.time()-t0:.0f}s")

    # Pass 3b: global Aho-Corasick scan of corpus
    print("[pass 3b] building global Aho-Corasick automaton...", flush=True)
    text_list = sorted(region_texts)
    A = ahocorasick.Automaton()
    for idx, t in enumerate(text_list):
        A.add_word(t, idx)
    A.make_automaton()
    print(f"[pass 3b] automaton built ({len(text_list):,} texts), "
          f"streaming corpus to record flat occurrences...", flush=True)

    hits_by_idx = defaultdict(list)
    t0 = time.time()
    for note_n, (pid, enc_date, note_text, row_idx) in enumerate(all_notes):
        if note_n and note_n % 200_000 == 0:
            print(f"  [pass 3b] {note_n:,} notes scanned, "
                  f"{time.time()-t0:.0f}s", flush=True)
        for end_idx, ai in A.iter(note_text):
            text = text_list[ai]
            start_idx = end_idx - len(text) + 1
            hits_by_idx[ai].append({
                "patient_id":     pid,
                "encounter_date": enc_date,
                "row_idx":        row_idx,
                "position":       start_idx,
            })
    print(f"[pass 3b] done: {time.time()-t0:.0f}s")
    del A
    print()

    # Convert defaultdict -> dict for clean fork semantics
    hits_by_idx = dict(hits_by_idx)
    text_to_idx = {t: i for i, t in enumerate(text_list)}

    # Pass 4: PARALLEL attribution
    print(f"[pass 4] attributing records in parallel ({args.workers} workers)...",
          flush=True)
    matched_sorted = sorted(matched, key=lambda p: -p.stat().st_size)
    n_workers = max(1, min(args.workers, len(matched_sorted)))
    worker_args = [str(p) for p in matched_sorted]

    # Use "fork" start method explicitly so the large globals are shared
    # copy-on-write rather than pickled. On Linux this is the default.
    try:
        ctx = mp.get_context("fork")
    except ValueError:
        print("[pass 4] WARNING: 'fork' start method unavailable, "
              "falling back to default (will pickle big data — slow)")
        ctx = mp.get_context()

    t0 = time.time()
    with ctx.Pool(
        processes=n_workers,
        initializer=_worker_init,
        initargs=(notes_by_patient, all_notes, hits_by_idx, text_to_idx,
                  args.apply, args.force),
    ) as pool:
        all_stats = []
        for i, stats in enumerate(pool.imap_unordered(_process_file_worker, worker_args)):
            all_stats.append(stats)
            if (i + 1) % 5 == 0 or (i + 1) == len(worker_args):
                el = time.time() - t0
                print(f"  [{i+1}/{len(worker_args)}] {el:.0f}s elapsed", flush=True)

    elapsed = time.time() - t0
    print(f"[pass 4] done in {elapsed:.0f}s")

    files_changed         = sum(1 for s in all_stats if s["file_changed"])
    records_changed       = sum(s["records_changed"] for s in all_stats)
    records_skipped_done  = sum(s["records_skipped_done"] for s in all_stats)
    total_regions         = sum(s["total_regions"] for s in all_stats)
    total_attributed      = sum(s["total_attributed"] for s in all_stats)
    total_unattributed    = sum(s["total_unattributed"] for s in all_stats)
    total_split_attributed = sum(s["total_split_attributed"] for s in all_stats)
    total_split_residual  = sum(s["total_split_residual"] for s in all_stats)

    print()
    print("=" * 72)
    print("[summary]")
    print(f"  files in scope          : {len(matched)}")
    print(f"  files changed           : {files_changed}")
    print(f"  records processed       : {records_changed:,}")
    print(f"  records skipped (done)  : {records_skipped_done:,}")
    print(f"  total regions           : {total_regions:,}")
    print(f"  attributed              : {total_attributed:,}  "
          f"({100*total_attributed/max(total_regions,1):.1f}%)")
    print(f"    of which from split   : {total_split_attributed:,}  "
          f"({100*total_split_attributed/max(total_regions,1):.1f}%)")
    print(f"  unattributed (no match) : {total_unattributed:,}  "
          f"({100*total_unattributed/max(total_regions,1):.1f}%)")
    print(f"    of which split residuals: {total_split_residual:,}")
    if not args.apply:
        print()
        print("  DRY RUN — pass --apply to actually write source_attribution")
    print("=" * 72)


if __name__ == "__main__":
    main()
