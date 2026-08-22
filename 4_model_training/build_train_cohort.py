import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

random.seed(42)

ANNOTATED  = config.ANNOTATED_COHORT_PATH
PRE_ANNOT  = config.PRE_ANNOTATION_PATH
OUTPUT     = config.TRAIN_COHORT_PATH
TOKENIZER  = config.TOKENIZER_PATH

TARGET_TOKENS   = 1_000_000_000
MIN_PER_DX      = 100
MAX_NOTE_TOKENS = 8_096

DIAGNOSES = config.SENSITIVE_DIAGNOSES


print(f"Loading tokenizer from {TOKENIZER}...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=True)

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))

print("Loading annotated cohort...")
annotated = {}
with open(ANNOTATED) as f:
    for line in f:
        row = json.loads(line)
        annotated[row["PatientUid"]] = row
print(f"  {len(annotated):,} annotated patients")

print(f"Loading notes and filtering patients with any note > {MAX_NOTE_TOKENS:,} tokens...")

patient_notes      = defaultdict(list)  # pid -> [(row, token_count)]
skipped_pids       = set()              # pids excluded due to oversized note
skipped_note_counts = {}                # pid -> number of oversized notes seen

with open(PRE_ANNOT) as f:
    for i, line in enumerate(f):
        row = json.loads(line)
        pid = row["PatientUid"]

        if pid not in annotated:
            continue
        if pid in skipped_pids:
            continue

        text      = row.get("Note", "")
        tok_count = count_tokens(text)

        if tok_count > MAX_NOTE_TOKENS:
            skipped_note_counts[pid] = skipped_note_counts.get(pid, 0) + 1
            skipped_pids.add(pid)
            if pid in patient_notes:
                del patient_notes[pid]
            continue

        patient_notes[pid].append((row, tok_count))

        if (i + 1) % 100_000 == 0:
            print(f"  ...{i+1:,} lines read | {len(skipped_pids):,} patients skipped so far", flush=True)

total_notes = sum(len(v) for v in patient_notes.values())
print(f"  {len(skipped_pids):,} patients skipped (had at least one note > {MAX_NOTE_TOKENS:,} tokens)")
print(f"  {len(patient_notes):,} patients remaining")
print(f"  {total_notes:,} total notes kept")

def patient_token_count(pid: str) -> int:
    return sum(tc for _, tc in patient_notes[pid])


print("\nSelecting eval-ready patients per diagnosis...")
dx_stats      = {}
selected_pids = set()

for dx in DIAGNOSES:
    present_key  = f"{dx}_present"
    in_start_key = f"{dx}_in_note_start"
    in_hpi_key   = f"{dx}_in_hpi"

    present_pids    = []
    eval_ready_pids = []

    for pid, row in annotated.items():
        if pid not in patient_notes:
            continue
        if row.get(present_key):
            present_pids.append(pid)
            if not row.get(in_start_key) and not row.get(in_hpi_key):
                eval_ready_pids.append(pid)

    n_select = min(MIN_PER_DX, len(eval_ready_pids))
    sampled  = random.sample(eval_ready_pids, n_select)
    selected_pids.update(sampled)

    dx_stats[dx] = {
        "present":    len(present_pids),
        "eval_ready": len(eval_ready_pids),
        "sampled":    n_select,
    }

print(f"After per-dx selection: {len(selected_pids):,} unique patients selected")

print("Counting tokens for seed patients...")
current_tokens = sum(patient_token_count(p) for p in selected_pids)
print(f"  Tokens from seed patients: {current_tokens:,} / {TARGET_TOKENS:,}")

if current_tokens < TARGET_TOKENS:
    print("Topping up with additional sensitive-dx patients...")
    any_dx_pids = [
        pid for pid, row in annotated.items()
        if pid in patient_notes
        and pid not in selected_pids
        and any(row.get(f"{dx}_present") for dx in DIAGNOSES)
    ]
    random.shuffle(any_dx_pids)

    for pid in any_dx_pids:
        if current_tokens >= TARGET_TOKENS:
            break
        selected_pids.add(pid)
        current_tokens += patient_token_count(pid)

    print(f"  After sensitive top-up: {len(selected_pids):,} patients | {current_tokens:,} tokens")

if current_tokens < TARGET_TOKENS:
    print("Still under target — drawing from all remaining annotated patients...")
    remaining_pids = [
        pid for pid in annotated
        if pid in patient_notes and pid not in selected_pids
    ]
    random.shuffle(remaining_pids)

    for pid in remaining_pids:
        if current_tokens >= TARGET_TOKENS:
            break
        selected_pids.add(pid)
        current_tokens += patient_token_count(pid)

    print(f"  After general top-up: {len(selected_pids):,} patients | {current_tokens:,} tokens")

if current_tokens < TARGET_TOKENS:
    print(f"  WARNING: corpus exhausted at {current_tokens:,} tokens — below {TARGET_TOKENS:,} target.")

print(f"\nWriting training cohort to {OUTPUT}...")
written = 0
with open(OUTPUT, "w") as out:
    for pid in selected_pids:
        for note_row, _ in patient_notes[pid]:
            out.write(json.dumps(note_row) + "\n")
            written += 1
print(f"  {written:,} notes written")

print("\n" + "=" * 100)
print(f"{'Diagnosis':<35} {'Present':>8} {'Eval-Ready':>10} {'Sampled':>8} {'In Cohort':>10} {'Skipped':>8}")
print("=" * 100)
for dx, s in dx_stats.items():
    in_cohort = sum(1 for pid in selected_pids if annotated[pid].get(f"{dx}_present"))
    n_skipped = sum(
        1 for pid in skipped_pids
        if annotated.get(pid, {}).get(f"{dx}_present")
    )
    print(
        f"{dx:<35} {s['present']:>8,} {s['eval_ready']:>10,} "
        f"{s['sampled']:>8,} {in_cohort:>10,} {n_skipped:>8,}"
    )
print("=" * 100)
print(f"\nTotal unique patients in training cohort:        {len(selected_pids):,}")
print(f"Total notes in training cohort:                  {written:,}")
print(f"Total tokens in training cohort:                 {current_tokens:,}")
print(f"Total patients skipped (oversized note):         {len(skipped_pids):,}")
print(f"Total oversized notes encountered:               {sum(skipped_note_counts.values()):,}")
