"""
Measures verbatim memorization using token-level n-gram matching between
each generated note and the target patient's own training notes. 
Reports the fraction of the generated note's tokens that have a tau=30-gram match. 

For every primary record (skips records with `_record_type` or `error_type`,
and records that already have `verbatim_coverage` unless --force), computes
and merges onto the record in place:
    verbatim_vs_train   {"tau_30": bool}
    verbatim_coverage   {"tau_30_coverage": float, "tau_30_covered_tokens": int,
                          "tau_30_total_tokens": int}
    memorized_spans     [{"tau": 30, "text": str}, ...]
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config  # noqa: E402


def check_verbatim(
    gen_text: str, note_texts: list[str], tok, tau_list: tuple = (30,),
) -> dict:
    """Check if generation contains any n-gram from the provided note texts."""
    gen_ids    = tok.encode(gen_text, add_special_tokens=False)
    all_ngrams = {tau: set() for tau in tau_list}

    for note_text in note_texts:
        note_ids = tok.encode(note_text, add_special_tokens=False)
        for tau in tau_list:
            for i in range(len(note_ids) - tau + 1):
                all_ngrams[tau].add(tuple(note_ids[i: i + tau]))

    results = {}
    for tau in tau_list:
        if len(gen_ids) < tau:
            results[f"tau_{tau}"] = False
            continue
        results[f"tau_{tau}"] = any(
            tuple(gen_ids[j: j + tau]) in all_ngrams[tau]
            for j in range(len(gen_ids) - tau + 1)
        )
    return results


def compute_verbatim_coverage(
    gen_text: str, note_texts: list[str], tok, tau_list: tuple = (30,),
) -> dict:
    """Fraction of the generation's tokens covered by at least one tau-gram match."""
    gen_ids = tok.encode(gen_text, add_special_tokens=False)
    n       = len(gen_ids)
    result  = {}

    for tau in tau_list:
        if n < tau:
            result[f"tau_{tau}_coverage"]      = 0.0
            result[f"tau_{tau}_covered_tokens"] = 0
            result[f"tau_{tau}_total_tokens"]   = n
            continue

        ngram_set = set()
        for note_text in note_texts:
            note_ids = tok.encode(note_text, add_special_tokens=False)
            for i in range(len(note_ids) - tau + 1):
                ngram_set.add(tuple(note_ids[i: i + tau]))

        covered = [False] * n
        for j in range(n - tau + 1):
            ngram = tuple(gen_ids[j: j + tau])
            if ngram in ngram_set:
                for k in range(j, j + tau):
                    covered[k] = True

        n_covered = sum(covered)
        result[f"tau_{tau}_coverage"]      = round(n_covered / n, 4) if n else 0.0
        result[f"tau_{tau}_covered_tokens"] = n_covered
        result[f"tau_{tau}_total_tokens"]   = n

    return result


def extract_memorized_spans(
    gen_text: str, note_texts: list[str], tok, verbatim_dict: dict,
    tau_list: tuple = (30,),
) -> list[dict]:
    """Decode the actual matched tau-gram spans, for source-note attribution."""
    gen_ids = tok.encode(gen_text, add_special_tokens=False)
    spans   = []
    for tau in tau_list:
        if not verbatim_dict.get(f"tau_{tau}"):
            continue
        if len(gen_ids) < tau:
            continue
        ngram_set = set()
        for note_text in note_texts:
            note_ids = tok.encode(note_text, add_special_tokens=False)
            for i in range(len(note_ids) - tau + 1):
                ngram_set.add(tuple(note_ids[i: i + tau]))
        seen_texts = set()
        for j in range(len(gen_ids) - tau + 1):
            ngram = tuple(gen_ids[j: j + tau])
            if ngram in ngram_set:
                span_text = tok.decode(list(ngram), skip_special_tokens=True)
                if span_text not in seen_texts:
                    seen_texts.add(span_text)
                    spans.append({"tau": tau, "text": span_text})
    return spans

def load_train_notes() -> dict[str, list[str]]:
    patient_notes = defaultdict(list)
    with open(config.TRAIN_COHORT_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                rec  = json.loads(line)
                uid  = str(rec.get("PatientUid", ""))
                note = rec.get("Note", "").strip()
                if uid and note:
                    patient_notes[uid].append(note)
    return patient_notes


def load_tokenizer():
    from transformers import AutoTokenizer
    tok_candidates = [Path(config.BASE_MODEL_PATH), config.CPT_MODEL_BASE / "final"]
    for tok_source in tok_candidates:
        if not tok_source.exists():
            continue
        try:
            print(f"[tokenizer] Trying {tok_source} ...")
            tok = AutoTokenizer.from_pretrained(str(tok_source), trust_remote_code=True)
            print(f"[tokenizer] Loaded from {tok_source}  vocab={tok.vocab_size}")
            return tok
        except Exception as e:
            print(f"[tokenizer] Failed ({type(e).__name__}: {e}) — trying next path")
    print(f"[tokenizer] All local paths failed. Trying HF hub: {config.TOKENIZER_PATH}")
    tok = AutoTokenizer.from_pretrained(config.TOKENIZER_PATH, trust_remote_code=True)
    print(f"[tokenizer] Loaded from hub  vocab={tok.vocab_size}")
    return tok


def _iter_target_files(ckpt_tag: str, diagnoses: list[str] | None,
                        prior_tier: str | None, probe_type: str | None):
    ckpt_dir = config.RESULTS_DIR / ckpt_tag
    dx_dirs = (
        [ckpt_dir / dx for dx in diagnoses] if diagnoses
        else sorted(p for p in ckpt_dir.iterdir() if p.is_dir())
    )
    pattern = f"{prior_tier or '*'}__{probe_type or '*'}.jsonl"
    for dx_dir in dx_dirs:
        if not dx_dir.exists():
            print(f"  [WARN] diagnosis dir not found: {dx_dir}")
            continue
        yield from sorted(dx_dir.glob(pattern))


def process_file(path: Path, tok, patient_train_notes: dict, force: bool) -> dict:
    records: list[dict | None] = []
    raw_lines: list[str] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            raw_lines.append(line)
            if not line:
                records.append(None)
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                records.append(None)

    stats = {"total": len(records), "computed": 0, "already_done": 0, "skipped_no_gen": 0}
    changed = False

    for rec in records:
        if rec is None or rec.get("_record_type") or rec.get("error_type"):
            continue
        if not rec.get("generation"):
            stats["skipped_no_gen"] += 1
            continue
        if not force and "verbatim_coverage" in rec:
            stats["already_done"] += 1
            continue

        patient_id  = str(rec.get("patient_id", ""))
        train_notes = patient_train_notes.get(patient_id, [])
        gen_clean   = rec["generation"]

        verbatim_train  = check_verbatim(gen_clean, train_notes, tok)
        coverage        = compute_verbatim_coverage(gen_clean, train_notes, tok)
        memorized_spans = extract_memorized_spans(gen_clean, train_notes, tok, verbatim_train)

        rec["verbatim_vs_train"] = verbatim_train
        rec["verbatim_coverage"] = coverage
        if memorized_spans:
            rec["memorized_spans"] = memorized_spans

        stats["computed"] += 1
        changed = True

    if not changed:
        print(f"  [verbatim] {path.name}: nothing to compute "
              f"({stats['already_done']} already done)")
        return stats

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as out_f:
        for raw_line, rec in zip(raw_lines, records):
            out_f.write((json.dumps(rec) if rec is not None else raw_line) + "\n")
    os.replace(tmp_path, path)
    print(f"  [verbatim] {path.name}: computed {stats['computed']} record(s) "
          f"({stats['already_done']} already done)")
    return stats


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_tag", default=config.DEFAULT_CHECKPOINT,
                    help="Results subdirectory name, e.g. 'final' or 'checkpoint-14178'")
    ap.add_argument("--diagnosis", nargs="+", default=None,
                    choices=config.SENSITIVE_DIAGNOSES,
                    help="One or more diagnoses (default: all found under the checkpoint dir)")
    ap.add_argument("--prior_tier", default=None, help="Filter to one prior tier")
    ap.add_argument("--probe_type", default=None, help="Filter to one probe type")
    ap.add_argument("--force", action="store_true",
                    help="Recompute records that already have verbatim_coverage")
    return ap.parse_args()


def main():
    args = parse_args()
    files = list(_iter_target_files(args.ckpt_tag, args.diagnosis,
                                     args.prior_tier, args.probe_type))
    if not files:
        print(f"[verbatim] no matching files under {config.RESULTS_DIR / args.ckpt_tag}")
        return

    print(f"[verbatim] {len(files)} file(s) to process")
    print("[data] Loading training notes ...")
    patient_train_notes = load_train_notes()
    print(f"[data] {len(patient_train_notes)} patients, "
          f"{sum(len(v) for v in patient_train_notes.values())} notes")

    tok = load_tokenizer()

    totals = {"computed": 0, "already_done": 0}
    for path in files:
        stats = process_file(path, tok, patient_train_notes, force=args.force)
        totals["computed"]     += stats["computed"]
        totals["already_done"] += stats["already_done"]

    print(f"\n[verbatim] done — computed {totals['computed']} new record(s), "
          f"{totals['already_done']} were already done")


if __name__ == "__main__":
    main()
