#!/usr/bin/env python3
"""
Sample 50 generations from each diagnosis's `<prior_tier>__<probe_type>.jsonl`
(default: encounter_info__note_continuation.jsonl, 300 total) for manual
inter-rater reliability annotation against the LLM judge.

Outputs:
  1. annotation_sheet.csv   -> generation_id, diagnosis, generation_text, manual_label (blank)
  2. judge_labels_hidden.csv -> generation_id, diagnosis, judge_diagnosis_mentioned,
                                 judge_patient_has_diagnosis, judge_evidence_types
"""

import argparse
import json
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import config  # noqa: E402


DIAGNOSES = config.SENSITIVE_DIAGNOSES
N_PER_DX = 50
SEED = 42

OUT_ANNOTATION_SHEET = Path("annotation_sheet.csv")
OUT_JUDGE_HIDDEN = Path("judge_labels_hidden.csv")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-tag", dest="ckpt_tag", default=config.DEFAULT_CHECKPOINT,
                    help="Results subdirectory name, e.g. 'final' or 'checkpoint-14178'")
    ap.add_argument("--prior_tier", default="encounter_info")
    ap.add_argument("--probe_type", default="note_continuation")
    return ap.parse_args()


def reservoir_sample_jsonl(path: Path, k: int, rng: random.Random):
    """Read the file once, line by line, keeping a uniform random sample of k
    records in memory (Algorithm R). Only JSON-parses the k retained lines,
    not every line in the file."""
    reservoir: list[str] = []
    n_seen = 0
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_seen += 1
            if len(reservoir) < k:
                reservoir.append(line)
            else:
                j = rng.randint(0, n_seen - 1)
                if j < k:
                    reservoir[j] = line
    records = [json.loads(l) for l in reservoir]
    return records, n_seen


def main():
    args = parse_args()
    base_dir = config.RESULTS_DIR / args.ckpt_tag
    filename = f"{args.prior_tier}__{args.probe_type}.jsonl"

    rng = random.Random(SEED)

    annotation_rows = []
    judge_rows = []
    gen_counter = 0

    for dx in DIAGNOSES:
        path = base_dir / dx / filename
        if not path.exists():
            print(f"  [WARN] missing file for {dx}: {path}")
            continue

        sample, n_seen = reservoir_sample_jsonl(path, N_PER_DX, rng)
        if n_seen < N_PER_DX:
            print(f"  [WARN] {dx} has only {n_seen} records, requested {N_PER_DX}")
        print(f"  [{dx}] scanned {n_seen} records, sampled {len(sample)}")

        for rec in sample:
            gen_counter += 1
            gen_id = f"{dx}_{gen_counter:04d}"

            judge = rec.get("llm_judge") or {}

            annotation_rows.append({
                "generation_id": gen_id,
                "diagnosis": dx,
                "patient_id": rec.get("patient_id", ""),
                "generation_text": rec.get("generation", ""),
                "manual_diagnosis_mentioned": "",   # fill: true / false
                "manual_patient_has_diagnosis": "",  # fill: positive / negative / ambiguous / (blank if not mentioned)
                "annotator_notes": "",
            })

            judge_rows.append({
                "generation_id": gen_id,
                "diagnosis": dx,
                "judge_diagnosis_mentioned": judge.get("diagnosis_mentioned"),
                "judge_patient_has_diagnosis": judge.get("patient_has_diagnosis"),
                "judge_evidence_types": ";".join(judge.get("evidence_types", []) or []),
            })

    # rows are already in diagnosis order (built diagnosis-by-diagnosis above)

    with open(OUT_ANNOTATION_SHEET, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "generation_id", "diagnosis", "patient_id", "generation_text",
            "manual_diagnosis_mentioned", "manual_patient_has_diagnosis", "annotator_notes",
        ])
        writer.writeheader()
        writer.writerows(annotation_rows)

    with open(OUT_JUDGE_HIDDEN, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "generation_id", "diagnosis", "judge_diagnosis_mentioned",
            "judge_patient_has_diagnosis", "judge_evidence_types",
        ])
        writer.writeheader()
        writer.writerows(judge_rows)

    print(f"Sampled {gen_counter} generations across {len(DIAGNOSES)} diagnoses.")
    print(f"Annotation sheet (blind): {OUT_ANNOTATION_SHEET}")
    print(f"Judge labels (hidden, don't peek): {OUT_JUDGE_HIDDEN}")


if __name__ == "__main__":
    main()
