
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
random.seed(43)  

ANNOTATED   = config.ANNOTATED_COHORT_PATH
PRE_ANNOT   = config.PRE_ANNOTATION_PATH
TRAIN_PATH  = config.TRAIN_COHORT_PATH
OUTPUT      = config.VAL_COHORT_PATH
TOKENIZER   = config.TOKENIZER_PATH

TARGET_TOKENS   = 10_000_000
MAX_NOTE_TOKENS = 8_096

DIAGNOSES = config.SENSITIVE_DIAGNOSES

print("Loading training cohort patient IDs...")
train_pids = set()
with open(TRAIN_PATH) as f:
    for line in f:
        train_pids.add(json.loads(line)["PatientUid"])
print(f"  {len(train_pids):,} unique train patients to exclude")

print("Loading annotated cohort...")
annotated = {}
with open(ANNOTATED) as f:
    for line in f:
        row = json.loads(line)
        annotated[row["PatientUid"]] = row
print(f"  {len(annotated):,} annotated patients")

print(f"Loading tokenizer from {TOKENIZER}...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=True)

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))

print("Loading notes for val candidate patients...")
candidate_pids = set(annotated.keys()) - train_pids
print(f"  {len(candidate_pids):,} candidate patients (annotated minus train)")

patient_notes = defaultdict(list)
skipped_pids  = set()

with open(PRE_ANNOT) as f:
    for i, line in enumerate(f):
        row = json.loads(line)
        pid = row["PatientUid"]

        if pid not in candidate_pids:
            continue
        if pid in skipped_pids:
            continue

        text      = row.get("Note", "")
        tok_count = count_tokens(text)

        if tok_count > MAX_NOTE_TOKENS:
            skipped_pids.add(pid)
            if pid in patient_notes:
                del patient_notes[pid]
            continue

        patient_notes[pid].append((row, tok_count))

        if (i + 1) % 100_000 == 0:
            print(f"  ...{i+1:,} lines | candidates kept: {len(patient_notes):,}", flush=True)

print(f"  {len(skipped_pids):,} patients skipped (oversized note)")
print(f"  {len(patient_notes):,} candidate patients with eligible notes")

def patient_token_count(pid: str) -> int:
    return sum(tc for _, tc in patient_notes[pid])

print("\nSampling val patients per diagnosis...")
selected_pids = set()
MIN_PER_DX_VAL = 1  # 1% size of train cohort 

for dx in DIAGNOSES:
    present_key = f"{dx}_present"
    candidates = [
        pid for pid in patient_notes
        if annotated[pid].get(present_key) and pid not in selected_pids
    ]
    n_select = min(MIN_PER_DX_VAL, len(candidates))
    sampled  = random.sample(candidates, n_select)
    selected_pids.update(sampled)
    print(f"  {dx}: {len(candidates):,} available, sampled {n_select}")

current_tokens = sum(patient_token_count(p) for p in selected_pids)
print(f"\nAfter per-dx seed: {len(selected_pids):,} patients | {current_tokens:,} tokens")

if current_tokens < TARGET_TOKENS:
    remaining = [pid for pid in patient_notes if pid not in selected_pids]
    random.shuffle(remaining)
    for pid in remaining:
        if current_tokens >= TARGET_TOKENS:
            break
        selected_pids.add(pid)
        current_tokens += patient_token_count(pid)
    print(f"After top-up: {len(selected_pids):,} patients | {current_tokens:,} tokens")

overlap = selected_pids & train_pids
assert not overlap, f"FATAL: {len(overlap)} val patients overlap with train!"
print(f"✓ Zero patient overlap with training cohort")

print(f"\nWriting val cohort to {OUTPUT}...")
written = 0
with open(OUTPUT, "w") as out:
    for pid in selected_pids:
        for note_row, _ in patient_notes[pid]:
            out.write(json.dumps(note_row) + "\n")
            written += 1
print(f"  {written:,} notes written | {current_tokens:,} tokens")