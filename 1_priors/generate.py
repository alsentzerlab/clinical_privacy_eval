import sys
import json
import argparse
import torch
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_prompt_from_priors as _priors  # noqa: E402
build_prompt_record   = _priors.build_prompt_record
PRIOR_TIERS           = _priors.PRIOR_TIERS
PROBE_TYPES           = _priors.PROBE_TYPES
UNSTRUCTURED_PRIORS   = _priors.UNSTRUCTURED_PRIORS
is_valid_combination  = _priors.is_valid_combination


EVAL_COHORT_DIR = config.EVAL_COHORT_DIR
TRAIN_PATH      = config.TRAIN_COHORT_PATH
CPT_MODEL_BASE  = config.CPT_MODEL_BASE
BASE_MODEL_PATH = Path(config.BASE_MODEL_PATH)
RESULTS_DIR     = config.RESULTS_DIR

SENSITIVE_DIAGNOSES = config.SENSITIVE_DIAGNOSES

NOTE_CONTINUATION_TOKENS = 1000
NOTE_PROBE_TOKENS        = 1000
REPETITION_WINDOW_TOKENS = 20


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior_tier",  required=True, choices=PRIOR_TIERS)
    ap.add_argument("--probe_type",  required=True, choices=PROBE_TYPES)
    ap.add_argument(
        "--diagnosis", nargs="+", default=None,
        choices=SENSITIVE_DIAGNOSES + ["all"],
        help="One or more diagnoses, or 'all' (default: all).",
    )
    ap.add_argument(
        "--checkpoint", default="final",
        help="'final', an epoch number, or an absolute path to a checkpoint dir",
    )
    ap.add_argument("--out_dir", type=str, default=None,
                    help="Override output directory")
    return ap.parse_args()


def resolve_model_path(checkpoint: str) -> Path:
    p = Path(checkpoint)
    if p.is_absolute() and p.exists():
        return p

    if checkpoint == "final":
        out = CPT_MODEL_BASE / "final"
        if not out.exists():
            raise FileNotFoundError(f"Final model not found: {out}")
        return out

    exact = CPT_MODEL_BASE / f"checkpoint-{checkpoint}"
    if exact.exists():
        return exact

    ckpt_dirs = sorted(CPT_MODEL_BASE.glob("checkpoint-*"),
                       key=lambda d: int(d.name.split("-")[1]))
    try:
        epoch_idx = int(checkpoint) - 1
        return ckpt_dirs[epoch_idx]
    except (ValueError, IndexError):
        raise FileNotFoundError(
            f"Cannot resolve checkpoint '{checkpoint}' in {CPT_MODEL_BASE}.\n"
            f"Available: {[d.name for d in ckpt_dirs]}"
        )
        
def load_eval_cohort(diagnosis: str) -> list[dict]:
    path = EVAL_COHORT_DIR / f"{diagnosis}_eval_cohort.jsonl"
    if not path.exists():
        print(f"  [WARN] eval cohort not found: {path}")
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_train_notes() -> dict[str, list[str]]:
    patient_notes = defaultdict(list)
    with open(TRAIN_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                rec  = json.loads(line)
                uid  = str(rec.get("PatientUid", ""))
                note = rec.get("Note", "").strip()
                if uid and note:
                    patient_notes[uid].append(note)
    return patient_notes


def load_completed_patient_ids(out_path: Path) -> tuple[set[str], set[str]]:
    """
    Returns (success_ids, error_ids).
    """
    success_ids: set[str] = set()
    error_ids:   set[str] = set()
    if not out_path.exists():
        return success_ids, error_ids

    with open(out_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("_record_type"):
                continue
            pid = str(rec.get("patient_id") or rec.get("PatientUid") or "")
            if not pid:
                continue
            if rec.get("error_type"):
                error_ids.add(pid)
            else:
                success_ids.add(pid)

    error_ids -= success_ids
    return success_ids, error_ids


def parse_age_from_dob(dob_str: str) -> float | None:
    if not dob_str or str(dob_str).strip() in ("", "nan", "None", "null"):
        return None
    ref = datetime(2024, 1, 1)
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m-%d-%Y", "%Y/%m/%d", "%d-%b-%Y", "%d/%m/%Y"):
        try:
            from datetime import datetime as dt
            dob = dt.strptime(str(dob_str).strip(), fmt)
            age = (ref - dob).days / 365.25
            return round(age, 1) if 0 < age < 130 else None
        except ValueError:
            continue
    return None


def normalize_record(raw: dict, dx: str) -> dict:
    label = raw.get("eval_label", "")

    if label == "positive":
        medications = (
            raw.get(f"{dx}_medications_filtered")
            or raw.get("medications")
            or raw.get("medications_raw")
            or ""
        )
    else:
        medications = (
            raw.get("medications_raw")
            or raw.get("medications")
            or ""
        )

    age = parse_age_from_dob(raw.get("extracted_dob", ""))

    return {
        "patient_id":     str(raw.get("PatientUid", "")),
        "encounter_date": raw.get("Encounter_Date", ""),
        "name":               raw.get("extracted_name", ""),
        "age":                age,
        "gender":             raw.get("patient_gender", ""),
        "marital_status":     raw.get("marital_status", ""),
        "occupation":         raw.get("occupation", ""),
        "children":           raw.get("children", ""),
        "current_medications": medications,
        "note_start": raw.get("note_start", ""),
        "note_hpi":   raw.get("note_hpi", ""),
        "last_note_text": raw.get("last_note", ""),
        "eval_dx_assignment": dx,
        "eval_dx_label":      label,
        "eval_split":         raw.get("eval_split", ""),
        "num_train_notes":    int(raw.get("num_notes", 0) or 0),

        **{k: v for k, v in raw.items() if k.endswith("_present")},
    }

def truncate_at_repetition(ids: list[int], window: int = REPETITION_WINDOW_TOKENS) -> list[int]:
    if len(ids) < window * 2:
        return ids
    seen = {}
    for i in range(len(ids) - window + 1):
        ngram = tuple(ids[i: i + window])
        if ngram in seen:
            return ids[:i]
        seen[ngram] = i
    return ids


@torch.no_grad()
def greedy_generate(model, tok, prompt: str, max_new_tokens: int) -> tuple[str, str, int]:
    inputs     = tok(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]
    print(f"    [generate] prompt_len={prompt_len} tokens, max_new={max_new_tokens}")

    out = model.generate(
        **inputs,
        max_new_tokens = max_new_tokens,
        do_sample      = False,
        temperature    = 0.0,
        pad_token_id   = tok.eos_token_id,
        eos_token_id   = tok.eos_token_id,
    )
    gen_ids = out[0, prompt_len:].tolist()
    if gen_ids and gen_ids[-1] == tok.eos_token_id:
        gen_ids = gen_ids[:-1]

    raw_ids    = gen_ids
    raw_text   = tok.decode(raw_ids, skip_special_tokens=True)
    clean_ids  = truncate_at_repetition(raw_ids)
    clean_text = tok.decode(clean_ids, skip_special_tokens=True)
    was_trunc  = len(clean_ids) < len(raw_ids)

    print(
        f"    [generate] raw={len(raw_ids)}  post_trunc={len(clean_ids)}  "
        f"truncated={was_trunc}  ({len(clean_text)} chars)"
    )
    return raw_text, clean_text, len(clean_ids)


@torch.no_grad()
def compute_remaining_note_tokens(rec: dict, prior_tier: str, tok) -> int:
    full_note = rec.get("last_note_text", "") or rec.get("last_note", "")
    if not full_note:
        return NOTE_CONTINUATION_TOKENS

    if prior_tier not in UNSTRUCTURED_PRIORS:
        return NOTE_CONTINUATION_TOKENS

    prefix, marker = _priors.get_unstructured_prefix_and_marker(rec, prior_tier)
    if prefix is None:
        return NOTE_CONTINUATION_TOKENS

    idx = full_note.find(prefix)
    if idx == -1:
        idx = full_note.find(prefix.rstrip())
        if idx == -1:
            return NOTE_CONTINUATION_TOKENS
        prefix_end = idx + len(prefix.rstrip())
    else:
        prefix_end = idx + len(prefix)

    after_prefix = full_note[prefix_end : prefix_end + 400]
    marker_idx = after_prefix.find(marker)
    if marker_idx == -1:
        return NOTE_CONTINUATION_TOKENS
    char_end = prefix_end + marker_idx + len(marker)

    remainder = full_note[char_end:]
    if not remainder.strip():
        return NOTE_CONTINUATION_TOKENS

    remaining_ids = tok.encode(remainder, add_special_tokens=False)
    return max(len(remaining_ids), 1)


def run_diagnosis(
    dx:                  str,
    args,
    model,
    tok,
    patient_train_notes: dict,
    model_path:          Path,
    ckpt_tag:            str,
) -> None:
    out_dir = (Path(args.out_dir) / dx) if args.out_dir else (RESULTS_DIR / ckpt_tag / dx)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path  = out_dir / f"{args.prior_tier}__{args.probe_type}.jsonl"
    done_path = out_path.with_suffix(".done")

    if done_path.exists():
        print(f"  [skip] .done file exists: {out_path}")
        return

    all_raw_records = load_eval_cohort(dx)
    if not all_raw_records:
        print(f"  [skip] no eval cohort records for {dx}")
        return

    all_cohort_ids = {str(r.get("PatientUid", "")) for r in all_raw_records}
    success_ids, error_ids = load_completed_patient_ids(out_path)
    missing_ids = all_cohort_ids - success_ids   # retry errors too

    if not missing_ids:
        print(f"  [skip] all {len(success_ids)} records already written: {out_path}")
        done_path.write_text(str(out_path))
        return

    file_mode = "a" if (out_path.exists() and success_ids) else "w"

    if success_ids or error_ids:
        print(
            f"  [resume] {len(success_ids)} ok  {len(error_ids)} errored  "
            f"{len(missing_ids - error_ids)} never written  "
            f"— reprocessing {len(missing_ids)} records  (mode={file_mode})"
        )

    raw_records = [r for r in all_raw_records
                   if str(r.get("PatientUid", "")) in missing_ids]

    print(f"\n{'='*70}")
    print(f"[dx={dx}] prior={args.prior_tier}  probe={args.probe_type}")
    print(f"[dx={dx}] output → {out_path}")
    print(f"{'='*70}")

    from collections import Counter
    group_counts = Counter(
        f"{r.get('eval_split')}/{r.get('eval_label')}" for r in raw_records
    )
    print(f"  [dx={dx}] {len(raw_records)} records to process "
          f"({len(success_ids)} already done): {dict(group_counts)}")

    skipped = errors = written = 0

    with open(out_path, file_mode, buffering=1) as out_f:
        for i, raw_rec in enumerate(raw_records):
            try:
                rec = normalize_record(raw_rec, dx)

                prompt_rec = build_prompt_record(
                    rec        = rec,
                    prior_tier = args.prior_tier,
                    probe_type = args.probe_type,
                )

                if prompt_rec is None:
                    skipped += 1
                    if skipped <= 5 or skipped % 50 == 0:
                        print(
                            f"  [skip] record {i} uid={raw_rec.get('PatientUid')} "
                            f"split={raw_rec.get('eval_split')} "
                            f"label={raw_rec.get('eval_label')} "
                            f"— build_prompt_record returned None"
                        )
                    continue

                patient_id  = rec["patient_id"]
                train_notes = patient_train_notes.get(patient_id, [])

                show_preview = (i < 3 or (i + 1) % 100 == 0)
                if show_preview:
                    print(
                        f"  [record {i+1}/{len(raw_records)}] "
                        f"uid={patient_id}  "
                        f"split={rec['eval_split']}  label={rec['eval_dx_label']}  "
                        f"train_notes={len(train_notes)}"
                    )
                    pp = prompt_rec["prompt"].replace("\n", "↵ ")
                    print(f"    [prompt] {pp[:300]}{'...' if len(pp) > 300 else ''}")

                result = {
                    "patient_id":         patient_id,
                    "encounter_date":     rec["encounter_date"],
                    "prior_tier":         args.prior_tier,
                    "probe_type":         args.probe_type,
                    "eval_dx_assignment": dx,
                    "eval_dx_label":      rec["eval_dx_label"],
                    "eval_split":         rec["eval_split"],
                    "num_train_notes":    rec["num_train_notes"],
                    "ground_truth":       prompt_rec["ground_truth"],
                    "model_path":         str(model_path),
                    "ckpt_tag":           ckpt_tag,
                    "prompt":             prompt_rec["prompt"],
                    "medications_used":   rec["current_medications"],
                }

                if args.probe_type == "note_continuation":
                    max_tok = compute_remaining_note_tokens(rec, args.prior_tier, tok)
                    print(f"    [note_continuation] remaining_note_tokens={max_tok}")
                else:
                    max_tok = NOTE_PROBE_TOKENS

                gen_raw, gen_clean, num_gen = greedy_generate(
                    model, tok, prompt_rec["prompt"], max_tok,
                )

                result["generation"]         = gen_clean
                result["generation_raw"]     = gen_raw
                result["repetition_removed"] = gen_raw != gen_clean
                result["num_gen_tokens"]     = num_gen
                result["truncated"]          = num_gen < max_tok

            except Exception as e:
                import traceback
                err = {
                    "patient_id": raw_rec.get("PatientUid"),
                    "eval_split": raw_rec.get("eval_split"),
                    "eval_label": raw_rec.get("eval_label"),
                    "error":      str(e),
                    "error_type": type(e).__name__,
                }
                out_f.write(json.dumps(err) + "\n")
                errors += 1
                print(f"  [ERROR] record {i}  {type(e).__name__}: {e}")
                traceback.print_exc()

            if (i + 1) % 50 == 0:
                print(
                    f"  [{i+1}/{len(raw_records)}] written={written}  "
                    f"errors={errors}  skipped={skipped}"
                )

    done_path.write_text(str(out_path))
    print(f"\n[done: {dx}] written={written}  skipped={skipped}  errors={errors}")
    print(f"[done: {dx}] → {out_path}")


def main():
    args = parse_args()

    if not is_valid_combination(args.prior_tier, args.probe_type):
        print(
            f"[skip] ({args.prior_tier}, {args.probe_type}) is an invalid combination. Exiting."
        )
        return

    if args.diagnosis is None or "all" in (args.diagnosis or []):
        diagnoses = SENSITIVE_DIAGNOSES
    else:
        diagnoses = list(args.diagnosis)

    model_path = resolve_model_path(args.checkpoint)
    ckpt_tag   = model_path.name

    print("=" * 70)
    print(f"[config] model_path:  {model_path}")
    print(f"[config] ckpt_tag:    {ckpt_tag}")
    print(f"[config] prior_tier:  {args.prior_tier}")
    print(f"[config] probe_type:  {args.probe_type}")
    print(f"[config] diagnoses:   {diagnoses}")
    print("=" * 70)

    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    print("\n[data] Loading training notes ...")
    patient_train_notes = load_train_notes()
    print(
        f"[data] {len(patient_train_notes)} patients, "
        f"{sum(len(v) for v in patient_train_notes.values())} notes"
    )

    tok_candidates = [BASE_MODEL_PATH, CPT_MODEL_BASE / "final"]
    tok = None
    for tok_source in tok_candidates:
        if not tok_source.exists():
            continue
        try:
            print(f"\n[tokenizer] Trying {tok_source} ...")
            tok = AutoTokenizer.from_pretrained(
                str(tok_source), trust_remote_code=True,
            )
            print(f"[tokenizer] Loaded from {tok_source}  vocab={tok.vocab_size}")
            break
        except Exception as e:
            print(f"[tokenizer] Failed ({type(e).__name__}: {e}) — trying next path")

    if tok is None:
        hub_name = config.TOKENIZER_PATH
        print(f"\n[tokenizer] All local paths failed. Trying HF hub: {hub_name}")
        tok = AutoTokenizer.from_pretrained(hub_name, trust_remote_code=True)
        print(f"[tokenizer] Loaded from hub  vocab={tok.vocab_size}")

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"\n[model] Loading CPT model from {model_path} ...")
    load_kwargs = dict(
        torch_dtype         = torch.bfloat16,
        attn_implementation = "flash_attention_2",
        trust_remote_code   = True,
        device_map          = {"": 0},
    )
    try:
        from transformers import Qwen3_5ForCausalLM
        model = Qwen3_5ForCausalLM.from_pretrained(str(model_path), **load_kwargs)
        print("[model] Loaded via Qwen3_5ForCausalLM")
    except (ImportError, AttributeError):
        raise RuntimeError(
            "Qwen3_5ForCausalLM not found in your transformers installation.\n"
            "Run: pip install git+https://github.com/huggingface/transformers.git\n"
            "Then retry."
        )
    model.eval()
    print(f"[model] dtype={next(model.parameters()).dtype}  "
          f"device={next(model.parameters()).device}")

    for dx in diagnoses:
        run_diagnosis(
            dx                  = dx,
            args                = args,
            model               = model,
            tok                 = tok,
            patient_train_notes = patient_train_notes,
            model_path          = model_path,
            ckpt_tag            = ckpt_tag,
        )

    print("\n[all done]")


if __name__ == "__main__":
    main()
    
