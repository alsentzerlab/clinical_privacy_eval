#!/usr/bin/env python3
"""
Builds a review queue of the cases where the automatic gold label is
questionable, shows you the evidence, and takes a one-key verdict.

Queues (control with --queues):
  sym_pos   Eval cohort POSITIVE patients whose only evidence for their cohort's
            diagnosis is symptom spans.
  neg_evid  Eval cohort NEGATIVE patients that have ANY evidence for their own
            diagnosis. 
  judge     LLM judge outputs for the sym_pos patients 
  
Controls:
  p  gold positive — the patient does have this diagnosis
  n  gold negative — they do not
  u  unsure / needs a second reader
  s  skip for now (stays in the queue next run)
  b  back one item
  q  save and quit
"""

import argparse
import csv
import json
import os
import sys
import textwrap
from collections import defaultdict
from datetime import datetime

SPAN_TYPES = ["diagnosis_spans", "icd10_spans", "medication_spans", "symptom_spans"]
LABEL = {"diagnosis_spans": "DIAGNOSIS", "icd10_spans": "ICD-10",
         "medication_spans": "MEDICATION", "symptom_spans": "SYMPTOM"}

DX_PRETTY = {
    "major_depressive_disorder": "MDD", "ptsd": "PTSD", "anxiety": "Anxiety",
    "bipolar_disorder": "Bipolar", "abortion": "Abortion", "hiv": "HIV",
}

FIELDS = ["timestamp", "queue", "diagnosis", "PatientUid", "eval_split",
          "gold_label", "evidence", "prior", "verdict", "note"]


def as_list(v):
    return v if isinstance(v, list) else []


def wrap(s, width=88, indent="      "):
    s = str(s).replace("\n", " ").strip()
    return textwrap.fill(s, width=width, initial_indent=indent,
                         subsequent_indent=indent + "  ")


# queue construction

def load_cohorts(cohort_dir, dx_filter):
    """[(dx, uid, split, label)]"""
    out = []
    for path in sorted(__import__("glob").glob(
            os.path.join(cohort_dir, "*_eval_cohort.jsonl"))):
        dx = os.path.basename(path)[: -len("_eval_cohort.jsonl")]
        if dx_filter and dx not in dx_filter:
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                out.append((dx, str(rec.get("PatientUid")),
                            rec.get("eval_split"), rec.get("eval_label")))
    return out


def load_annotations(path, uids):
    src = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            uid = str(rec.get("PatientUid"))
            if uid in uids:
                src[uid] = rec
    return src


def evidence_of(rec, dx):
    spans = {st: as_list(rec.get(f"{dx}_{st}")) for st in SPAN_TYPES}
    exp = bool(spans["diagnosis_spans"] or spans["icd10_spans"])
    med = bool(spans["medication_spans"])
    sym = bool(spans["symptom_spans"])
    parts = [k for k, v in (("exp", exp), ("med", med), ("sym", sym)) if v]
    return spans, ("+".join(parts) if parts else "none")


UID_KEYS = ('"patient_id":', '"PatientUid":', '"patient_uid":')


def _peek_uid(line):
    """
    Pull the patient id out of a raw JSONL line without json.loads.
    Judge records embed long generations, so full parsing every line is the
    slow part; this lets us skip lines we don't want.
    Returns None if the key isn't found (caller falls back to parsing).
    """
    for key in UID_KEYS:
        i = line.find(key)
        if i == -1:
            continue
        j = line.find('"', i + len(key))
        if j == -1:
            return None
        k = line.find('"', j + 1)
        if k == -1:
            return None
        return line[j + 1:k]
    return None


def _rec_uid(rec):
    return str(rec.get("patient_id") or rec.get("PatientUid")
               or rec.get("patient_uid") or "")


def load_judge(judge_dir, wanted, dx_filter=None,
               probes=("patient_note", "note_continuation"),
               priors=None, judge_evidence=None, any_patient=False):
    """
    (dx, uid) -> [ {prior, probe, judge, generation} ]

    judge_evidence: if given, keep only records whose llm_judge.evidence_types
    is exactly this set — e.g. {"symptoms"} for verdicts the judge reached on
    symptom language alone.
    any_patient: ignore `wanted` and take every patient (used when the judge's
    own evidence is the filter rather than the gold label).
    """
    import glob as _g
    out = defaultdict(list)
    files = []
    for path in sorted(_g.glob(os.path.join(judge_dir, "**", "*.jsonl"),
                               recursive=True)):
        stem = os.path.basename(path)[:-6]
        if "__" not in stem:
            continue
        prior, probe = stem.split("__", 1)
        if probe not in probes:
            continue
        if priors and prior not in priors:
            continue
        dx = os.path.basename(os.path.dirname(path))
        if dx_filter and dx not in dx_filter:
            continue
        files.append((path, dx, prior, probe))

    wanted_uids = {u for _, u in wanted}
    print(f"scanning {len(files)} judge files ...", file=sys.stderr)

    kept = seen = 0
    for n, (path, dx, prior, probe) in enumerate(files, 1):
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                if not any_patient:
                    uid = _peek_uid(line)
                    if uid is not None and (
                        uid not in wanted_uids or (dx, uid) not in wanted
                    ):
                        continue
                else:
                    uid = None
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("_record_type") or rec.get("error_type"):
                    continue
                uid = _rec_uid(rec)
                if not any_patient and (dx, uid) not in wanted:
                    continue
                judge = rec.get("llm_judge") or {}
                seen += 1
                if judge_evidence is not None:
                    et = judge.get("evidence_types")
                    if not isinstance(et, list) or set(et) != judge_evidence:
                        continue
                kept += 1
                out[(dx, uid)].append({
                    "prior": prior,
                    "probe": probe,
                    "judge": judge,
                    "generation": rec.get("generation") or "",
                    "eval_dx_label": rec.get("eval_dx_label"),
                    "eval_split": rec.get("eval_split"),
                })
        print(f"  [{n}/{len(files)}] {dx}/{prior}__{probe}", file=sys.stderr)
    if judge_evidence is not None:
        print(f"  {kept:,} of {seen:,} judge records match "
              f"evidence_types == {sorted(judge_evidence)}", file=sys.stderr)
    return out


# display

def show_patient(item, idx, total, done_count):
    dx, uid, split, label, spans, evid = (
        item["dx"], item["uid"], item["split"], item["label"],
        item["spans"], item["evidence"]
    )
    print("\n" * 2 + "=" * 92)
    print(f"[{idx + 1}/{total}]  reviewed so far: {done_count}      "
          f"queue: {item['queue']}")
    print("=" * 92)
    print(f"  Diagnosis   : {DX_PRETTY.get(dx, dx)}   ({dx})")
    print(f"  Patient     : {uid}")
    print(f"  Cohort      : {split} / {label}      auto-evidence: {evid}")
    print("-" * 92)
    for st in SPAN_TYPES:
        if not spans[st]:
            continue
        print(f"  {LABEL[st]} spans ({len(spans[st])}):")
        for s in spans[st]:
            print(wrap(s))
    if item["queue"] == "neg_evid":
        print("\n  ^ labelled NEGATIVE but has the above evidence.")
    elif item["queue"] in ("sym_pos", "sym_only"):
        print(f"\n  ^ auto-labelled {label.upper()} with symptom language as "
              f"the only evidence.")
    print("-" * 92)


def show_judge(item, idx, total, done_count):
    print("\n" * 2 + "=" * 92)
    print(f"[{idx + 1}/{total}]  reviewed so far: {done_count}      queue: judge")
    print("=" * 92)
    print(f"  Diagnosis   : {DX_PRETTY.get(item['dx'], item['dx'])}")
    print(f"  Patient     : {item['uid']}")
    print(f"  Cohort      : {item['split']} / {item['label']}   "
          f"gold evidence: {item['evidence']}")
    print(f"  Prior       : {item['prior']}   probe: {item['probe']}")
    print("-" * 92)
    print("  GOLD symptom spans:")
    for s in item["spans"]["symptom_spans"]:
        print(wrap(s))
    j = item["judge"]
    print("\n  JUDGE:")
    print(f"    evidence_types       : {j.get('evidence_types')}")
    print(f"    diagnosis_mentioned  : {j.get('diagnosis_mentioned')}")
    print(f"    patient_has_diagnosis: {j.get('patient_has_diagnosis')}")
    for st in ("diagnosis_spans", "symptom_spans", "medication_spans"):
        for s in j.get(st) or []:
            print(f"    [judge {st.split('_')[0]}]")
            print(wrap(s))
    for k in ("reasoning", "explanation", "rationale"):
        if j.get(k):
            print(f"    {k}:")
            print(wrap(j[k]))
    if item.get("generation"):
        print("\n  MODEL GENERATION (truncated):")
        print(wrap(item["generation"][:1200]))
    print("-" * 92)


# main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cohort_dir")
    ap.add_argument("--annotated", required=True,
                    help="one-record-per-patient annotation file")
    ap.add_argument("--out", required=True, help="verdict CSV (appended)")
    ap.add_argument("--queues", nargs="+",
                    default=["sym_pos", "neg_evid"],
                    choices=["sym_only", "sym_pos", "neg_evid", "judge"],
                    help="sym_only = every cohort patient whose evidence for "
                         "their own dx is symptom spans alone, positive and "
                         "negative alike")
    ap.add_argument("--dx", nargs="+", help="restrict to these diagnoses")
    ap.add_argument("--judge-dir", help="checkpoint dir for the judge queue")
    ap.add_argument("--priors", nargs="+",
                    help="restrict the judge queue to these priors, e.g. "
                         "encounter_info_cc_hpi")
    ap.add_argument("--judge-evidence", nargs="+",
                    help="keep only judge records whose llm_judge.evidence_types "
                         "is exactly this set, e.g. --judge-evidence symptoms. "
                         "Filters on the JUDGE's own basis rather than the gold "
                         "evidence, and covers every patient.")
    ap.add_argument("--shuffle", action="store_true",
                    help="randomize order (reduces ordering bias)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cohort = load_cohorts(args.cohort_dir, set(args.dx) if args.dx else None)
    uids = {uid for _, uid, _, _ in cohort}
    print(f"{len(cohort):,} cohort rows, {len(uids):,} unique patients",
          file=sys.stderr)
    ann = load_annotations(args.annotated, uids)

    # build items
    items = []
    for dx, uid, split, label in cohort:
        rec = ann.get(uid)
        if rec is None:
            continue
        spans, evid = evidence_of(rec, dx)
        base = {"dx": dx, "uid": uid, "split": split, "label": label,
                "spans": spans, "evidence": evid, "prior": "", "probe": ""}
        if evid == "sym" and "sym_only" in args.queues:
            items.append({**base, "queue": "sym_only"})
        if label == "positive" and evid == "sym" and "sym_pos" in args.queues:
            items.append({**base, "queue": "sym_pos"})
        if label == "negative" and evid != "none" and "neg_evid" in args.queues:
            items.append({**base, "queue": "neg_evid"})

    if "judge" in args.queues:
        if not args.judge_dir:
            raise SystemExit("--judge-dir required for the judge queue")

        if args.judge_evidence:
            # filter on the JUDGE's own basis, not the gold evidence
            jr = load_judge(args.judge_dir, set(),
                            dx_filter=set(args.dx) if args.dx else None,
                            priors=set(args.priors) if args.priors else None,
                            judge_evidence=set(args.judge_evidence),
                            any_patient=True)
        else:
            symkeys = {(d["dx"], d["uid"]) for d in items
                       if d["queue"] in ("sym_pos", "sym_only")}
            if not symkeys:
                symkeys = {
                    (dx, uid) for dx, uid, split, label in cohort
                    if uid in ann and evidence_of(ann[uid], dx)[1] == "sym"
                }
            jr = load_judge(args.judge_dir, symkeys,
                            dx_filter=set(args.dx) if args.dx else None,
                            priors=set(args.priors) if args.priors else None)

        for (dx, uid), entries in jr.items():
            match = next((d for d in items
                          if d["dx"] == dx and d["uid"] == uid), None)
            if match is None:
                rec = ann.get(uid)
                if rec is not None:
                    spans, evid = evidence_of(rec, dx)
                else:
                    spans = {st: [] for st in SPAN_TYPES}
                    evid = "?"
                row = next((r for r in cohort if r[0] == dx and r[1] == uid),
                           (dx, uid, entries[0].get("eval_split"),
                            entries[0].get("eval_dx_label")))
                match = {"dx": dx, "uid": uid, "split": row[2], "label": row[3],
                         "spans": spans, "evidence": evid}
            for e in entries:
                items.append({**match, "queue": "judge", "prior": e["prior"],
                              "probe": e["probe"], "judge": e["judge"],
                              "generation": e["generation"]})

    if args.shuffle:
        import random
        random.Random(args.seed).shuffle(items)
    else:
        items.sort(key=lambda d: (d["queue"], d["dx"], d["split"], d["prior"]))

    # resume
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for row in csv.DictReader(f):
                done.add((row["queue"], row["diagnosis"], row["PatientUid"],
                          row.get("prior", "")))
        print(f"resuming — {len(done)} already reviewed", file=sys.stderr)

    todo = [d for d in items
            if (d["queue"], d["dx"], d["uid"], d.get("prior", "")) not in done]

    counts = defaultdict(int)
    for d in items:
        counts[d["queue"]] += 1
    print("\nqueue sizes:")
    for q, n in sorted(counts.items()):
        rem = sum(1 for d in todo if d["queue"] == q)
        print(f"  {q:<10} {n:>5} total, {rem:>5} remaining")
    if not todo:
        print("\nnothing left to review.")
        return

    new = os.path.exists(args.out) is False
    out_f = open(args.out, "a", newline="")
    writer = csv.DictWriter(out_f, fieldnames=FIELDS)
    if new:
        writer.writeheader()
        out_f.flush()

    print("\n  p=positive  n=negative  u=unsure  s=skip  b=back  q=quit")
    print("  (append a comment after the key, e.g. 'n generic fatigue only')")

    i = 0
    reviewed = 0
    while i < len(todo):
        item = todo[i]
        if item["queue"] == "judge":
            show_judge(item, i, len(todo), reviewed)
        else:
            show_patient(item, i, len(todo), reviewed)

        try:
            raw = input("  verdict > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nsaving and exiting.")
            break
        if not raw:
            continue
        key, _, comment = raw.partition(" ")
        key = key.lower()

        if key == "q":
            print("saving and exiting.")
            break
        if key == "s":
            i += 1
            continue
        if key == "b":
            i = max(0, i - 1)
            continue
        if key not in ("p", "n", "u"):
            print("  ? use p / n / u / s / b / q")
            continue

        writer.writerow({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "queue": item["queue"],
            "diagnosis": item["dx"],
            "PatientUid": item["uid"],
            "eval_split": item["split"],
            "gold_label": item["label"],
            "evidence": item["evidence"],
            "prior": item.get("prior", ""),
            "verdict": {"p": "positive", "n": "negative", "u": "unsure"}[key],
            "note": comment.strip(),
        })
        out_f.flush()
        reviewed += 1
        i += 1

    out_f.close()
    print(f"\n{reviewed} verdicts written to {args.out}")
    print("PHI reminder: that CSV holds no note text, only ids and verdicts — "
          "but keep it on the cluster anyway.")


if __name__ == "__main__":
    main()
