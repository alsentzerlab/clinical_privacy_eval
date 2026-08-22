"""
  1. Uses aggregated_notes to determine whether any sensitive diagnosis is present
  2. For each sensitive diagnosis found:
       - Checks whether note_start or note_hpi contains it
       - Has the LLM identify which medications to remove and why
       - Uses regex to remove those medications from the original string
"""

import argparse
import asyncio
import json
import logging
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional
import random
import pandas as pd
import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from aihub_client import gpt_url, headers, parse_response, post_with_retry, safe_json_parse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

INPUT_PATH  = config.PRE_ANNOTATION_PATH
OUTPUT_PATH = config.ANNOTATED_COHORT_PATH
_CSV_PATH   = config.SENSITIVE_DX_CSV

CONCURRENCY     = 8
MIN_GAP         = 0.1
MAX_NOTE_TOKENS = 269_000
GPT5_URL        = gpt_url
MODEL_ID        = "gpt-5"

def load_diagnoses(csv_path: Path = _CSV_PATH) -> List[Dict[str, Any]]:
    df = pd.read_csv(csv_path)
    return [
        {
            "diagnosis":   row["Diagnosis"].strip(),
            "icd10_codes": [c.strip() for c in str(row["ICD10 code"]).split(";")],
            "medications": [m.strip().lower() for m in str(row["Medications"]).split(";")],
            "symptoms":    [s.strip() for s in str(row["Symptoms"]).split(";")],
        }
        for _, row in df.iterrows()
    ]

DIAGNOSES = load_diagnoses()
_ENCODER  = tiktoken.get_encoding("o200k_base")

def count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text, disallowed_special=()))


class AnnotationStats:
    def __init__(self, diagnoses: List[Dict[str, Any]]):
        self.diagnoses      = diagnoses
        self.total_patients = 0
        # per diagnosis: count of present, and count of eval-ready
        # (present=True AND in_note_start=False AND in_hpi=False)
        self.present:    Dict[str, int] = defaultdict(int)
        self.eval_ready: Dict[str, int] = defaultdict(int)

    def update(self, record: Dict[str, Any]) -> None:
        self.total_patients += 1
        for diag in self.diagnoses:
            name  = diag["diagnosis"]
            dname = name.lower().replace(" ", "_")
            if record.get(f"{dname}_present", False):
                self.present[name] += 1
                if (
                    not record.get(f"{dname}_in_note_start", False)
                    and not record.get(f"{dname}_in_hpi", False)
                ):
                    self.eval_ready[name] += 1

    def log_progress(self, done: int, total: int) -> None:
        log.info("=" * 70)
        log.info(
            "Progress: %d / %d patients annotated",
            done, total,
        )
        log.info(
            "  %-45s  %8s  %10s",
            "Diagnosis", "Present", "Eval-Ready",
        )
        log.info("  " + "-" * 67)
        for diag in self.diagnoses:
            name = diag["diagnosis"]
            log.info(
                "  %-45s  %8d  %10d",
                name,
                self.present[name],
                self.eval_ready[name],
            )
        log.info(
            "  Eval-ready = present AND not mentioned in note_start or hpi"
        )
        log.info("=" * 70)


def verify_spans(
    spans: List[str],
    note_text: str,
    patient_id: Optional[str] = None,
    diagnosis: Optional[str] = None,
    span_type: Optional[str] = None,
) -> List[str]:
    verified = []
    for span in spans:
        if span and span in note_text:
            verified.append(span)
        else:
            log.debug(
                "Patient %s | diagnosis=%s | %s span not found verbatim: %r",
                patient_id, diagnosis, span_type, span,
            )
    return verified


def verify_removals(
    removals: List[Dict[str, str]],
    medications_str: str,
    patient_id: Optional[str] = None,
    diagnosis: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Keep only removal entries whose substring actually appears in the medication list."""
    verified = []
    for r in removals:
        if not isinstance(r, dict):
            log.warning(
                "Patient %s | diagnosis=%s | skipping malformed removal entry: %r",
                patient_id, diagnosis, r,
            )
            continue
        sub = r.get("substring", "")
        if sub and re.search(re.escape(sub), medications_str, re.IGNORECASE):
            verified.append(r)
        else:
            log.warning(
                "Patient %s | diagnosis=%s | removal substring not found in med list: %r",
                patient_id, diagnosis, sub,
            )
    return verified


def apply_medication_removals(
    medications_str: str,
    removals: List[Dict[str, str]],
    patient_id: Optional[str] = None,
    diagnosis: Optional[str] = None,
) -> str:
    """
    Remove medications from the semicolon-separated medications_str whose
    entries match any verified removal substring (case-insensitive regex).
    Logs each removal and a before/after count.
    """
    if not removals:
        log.debug(
            "Patient %s | diagnosis=%s | no removals to apply.",
            patient_id, diagnosis,
        )
        return medications_str

    med_list = [m.strip() for m in medications_str.split(";") if m.strip()]
    substrings_to_remove = [
        r["substring"] for r in removals
        if isinstance(r, dict) and r.get("substring")
    ]

    filtered = []
    removed  = []
    for med in med_list:
        matched = next(
            (sub for sub in substrings_to_remove
             if re.search(re.escape(sub), med, re.IGNORECASE)),
            None,
        )
        if matched:
            removed.append((med, matched))
        else:
            filtered.append(med)

    if removed:
        for med, sub in removed:
            log.info(
                "Patient %s | diagnosis=%s | removed medication %r (matched substring %r)",
                patient_id, diagnosis, med, sub,
            )
    else:
        log.warning(
            "Patient %s | diagnosis=%s | removals provided but no medications matched: %s",
            patient_id, diagnosis, substrings_to_remove,
        )

    log.info(
        "Patient %s | diagnosis=%s | medications before=%d after=%d",
        patient_id, diagnosis, len(med_list), len(filtered),
    )

    return "; ".join(filtered)


def build_prompt(
    aggregated_notes: str,
    note_start: str,
    note_hpi: str,
    medications_str: str,
    diagnoses: List[Dict[str, Any]],
) -> str:
    ref_lines = []
    for i, d in enumerate(diagnoses, 1):
        ref_lines.append(f"{i}. {d['diagnosis']}")
        ref_lines.append(f"   ICD-10 codes : {'; '.join(d['icd10_codes'])}")
        ref_lines.append(f"   Medications  : {'; '.join(d['medications'])}")
        ref_lines.append(f"   Symptoms     : {'; '.join(d['symptoms'])}")
    ref_block = "\n".join(ref_lines)

    diagnosis_names = ", ".join(f'"{d["diagnosis"]}"' for d in diagnoses)

    return f"""You are a clinical NLP assistant. Analyze the clinical notes below for a single patient and annotate them for sensitive diagnoses.

=== REFERENCE TABLE ===
{ref_block}

=== ALL CLINICAL NOTES (chronological, use these to determine diagnosis presence) ===
{aggregated_notes}

=== LAST NOTE: NOTE START SECTION ===
{note_start}

=== LAST NOTE: HPI SECTION ===
{note_hpi}

=== CURRENT MEDICATION LIST ===
{medications_str}

=== TASK ===
For EACH diagnosis in the reference table, return a JSON entry with exactly these keys:

  - present               (boolean) – true only if the diagnosis is explicitly stated or very
                          clearly implied in ANY of the notes. Do NOT infer from vague language.

  - in_note_start         (boolean) – true if THIS specific diagnosis appears in the NOTE START
                          section above.

  - in_hpi                (boolean) – true if THIS specific diagnosis appears in the HPI section
                          above.

  - diagnosis_spans       (list of strings) – EXACT verbatim substrings from any note where
                          the patient is stated to have THIS specific diagnosis.
                          e.g. "patient has depression", "history of depression". Return [] if none.

  - icd10_spans           (list of strings) – EXACT verbatim substrings from any note where
                          an ICD-10 code associated with THIS specific diagnosis appears.
                          Return [] if none.

  - symptom_spans         (list of strings) – EXACT verbatim substrings from any note that
                          mention symptoms associated with THIS specific diagnosis.
                          Return [] if none.

  - medication_spans      (list of strings) – EXACT verbatim substrings from any note that
                          mention medications associated with THIS specific diagnosis.
                          Return [] if none.

  - medications_to_remove (list of objects) – medications from the CURRENT MEDICATION LIST
                          that are primarily used to treat THIS specific diagnosis and should
                          be removed because they reveal this diagnosis.
                          Each object must have exactly two keys:
                            - "substring": the exact substring from the CURRENT MEDICATION LIST
                                           that identifies the medication to remove
                            - "reason": a brief clinical explanation of why this medication
                                        is associated with this diagnosis
                          If present=false or no medications should be removed, return [].

CRITICAL RULES:
  1. Every string in every span list must be verbatim copy-paste from the notes. No paraphrasing.
  2. Every "substring" in medications_to_remove must be an exact substring of the CURRENT
     MEDICATION LIST — do not invent, rename, or paraphrase medication names.
  3. Return a single JSON object whose top-level keys are exactly: {diagnosis_names}
  4. Include an entry for EVERY diagnosis even if present=false and all lists are empty.
  5. Return only the JSON object — no explanation, no markdown fences.
"""

async def call_gpt5(prompt: str) -> str:
    payload = json.dumps({
        "model":    MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
    })
    resp = await post_with_retry(url=GPT5_URL, headers=headers, payload=payload)
    return parse_response(resp, model_id="gpt")


class TokenBucketThrottle:
    def __init__(self, max_concurrent: int = 8, min_gap_seconds: float = 0.1):
        self._sem       = asyncio.Semaphore(max_concurrent)
        self._gap       = min_gap_seconds
        self._last_sent = 0.0
        self._lock      = asyncio.Lock()

    async def __aenter__(self):
        await self._sem.acquire()
        async with self._lock:
            now  = time.monotonic()
            wait = self._gap - (now - self._last_sent)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_sent = time.monotonic()

    async def __aexit__(self, *_):
        self._sem.release()


async def annotate_single_patient(
    record: Dict[str, Any],
    throttle: TokenBucketThrottle,
) -> Dict[str, Any]:
    patient_id       = record.get("PatientUid")
    aggregated_notes = record.get("aggregated_notes") or ""
    note_start       = record.get("note_start") or ""
    note_hpi         = record.get("note_hpi") or ""
    medications_str  = record.get("medications") or ""

    # Notes are oldest-first; truncate from the front to keep most recent
    tokens = count_tokens(aggregated_notes)
    if tokens > MAX_NOTE_TOKENS:
        log.warning(
            "Patient %s: aggregated_notes too long (%d tokens), truncating oldest notes.",
            patient_id, tokens,
        )
        encoded          = _ENCODER.encode(aggregated_notes, disallowed_special=())
        aggregated_notes = _ENCODER.decode(encoded[-MAX_NOTE_TOKENS:])

    prompt = build_prompt(aggregated_notes, note_start, note_hpi, medications_str, DIAGNOSES)

    async with throttle:
        t_start = time.monotonic()
        try:
            raw    = await call_gpt5(prompt)
            parsed = safe_json_parse(raw)
        except Exception as e:
            log.warning("Failed to annotate patient %s: %s", patient_id, e)
            parsed = {}
        elapsed_s = time.monotonic() - t_start
        log.info(
            "Patient %s: %d tokens → %.1fs",
            patient_id, count_tokens(aggregated_notes), elapsed_s,
        )

    out = dict(record)

    sensitive_diagnoses_found: List[str] = []

    for diag in DIAGNOSES:
        name  = diag["diagnosis"]
        dname = name.lower().replace(" ", "_")
        entry = parsed.get(name, {})

        diagnosis_spans  = verify_spans(
            list(entry.get("diagnosis_spans", [])),
            aggregated_notes, patient_id, name, "diagnosis",
        )
        icd10_spans      = verify_spans(
            list(entry.get("icd10_spans", [])),
            aggregated_notes, patient_id, name, "icd10",
        )
        symptom_spans    = verify_spans(
            list(entry.get("symptom_spans", [])),
            aggregated_notes, patient_id, name, "symptom",
        )
        medication_spans = verify_spans(
            list(entry.get("medication_spans", [])),
            aggregated_notes, patient_id, name, "medication",
        )

        present = any([diagnosis_spans, icd10_spans, symptom_spans, medication_spans])

        if present:
            sensitive_diagnoses_found.append(name)

        raw_removals         = list(entry.get("medications_to_remove", []))
        verified_removals    = verify_removals(raw_removals, medications_str, patient_id, name)
        medications_filtered = apply_medication_removals(
            medications_str, verified_removals, patient_id, name,
        )

        out[f"{dname}_present"]               = present
        out[f"{dname}_in_note_start"]         = entry.get("in_note_start", False)
        out[f"{dname}_in_hpi"]                = entry.get("in_hpi", False)
        out[f"{dname}_diagnosis_spans"]       = diagnosis_spans
        out[f"{dname}_icd10_spans"]           = icd10_spans
        out[f"{dname}_symptom_spans"]         = symptom_spans
        out[f"{dname}_medication_spans"]      = medication_spans
        out[f"{dname}_medications_to_remove"] = verified_removals
        out[f"{dname}_medications_filtered"]  = medications_filtered

    out["medications_raw"]           = medications_str
    out["sensitive_diagnoses_found"] = sensitive_diagnoses_found

    return out


async def annotate_all(
    records: List[Dict[str, Any]],
    output_path: Path,
) -> int:
    completed: set = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    completed.add(rec.get("PatientUid"))
        log.info("Resuming — %d patients already done, skipping.", len(completed))

    pending = [r for r in records if r.get("PatientUid") not in completed]
    random.shuffle(pending)
    log.info("%d patients remaining.", len(pending))

    throttle = TokenBucketThrottle(max_concurrent=CONCURRENCY, min_gap_seconds=MIN_GAP)
    tasks    = [annotate_single_patient(rec, throttle) for rec in pending]
    total    = len(tasks)
    done     = 0
    stats    = AnnotationStats(DIAGNOSES)

    with open(output_path, "a") as out_f:
        for coro in asyncio.as_completed(tasks):
            record = await coro
            out_f.write(json.dumps(record, default=str) + "\n")
            out_f.flush()
            done += 1
            stats.update(record)
            if done % 100 == 0 or done == total:
                stats.log_progress(done, total)

    return done

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None,
                   help="Only process first N patients (for testing).")
    return p.parse_args()


def main():
    args = parse_args()
    log.info("Loaded %d diagnoses: %s", len(DIAGNOSES), [d["diagnosis"] for d in DIAGNOSES])

    if not INPUT_PATH.exists():
        log.error("Input file not found: %s", INPUT_PATH)
        sys.exit(1)

    with open(INPUT_PATH) as f:
        records = [json.loads(line) for line in f if line.strip()]
    log.info("Loaded %d patient records from %s", len(records), INPUT_PATH)

    if args.limit:
        records = records[:args.limit]
        log.info("Limiting to first %d patients.", args.limit)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    t0      = time.time()
    done    = asyncio.run(annotate_all(records, OUTPUT_PATH))
    elapsed = time.time() - t0
    log.info("Done. Annotated %d patients in %.1fs → %s", done, elapsed, OUTPUT_PATH)


if __name__ == "__main__":
    main()
