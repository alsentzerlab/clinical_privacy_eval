import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from psmpy import PsmPy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


warnings.filterwarnings("ignore")

TRAIN_COHORT_PATH = config.TRAIN_COHORT_PATH
ANNOTATED_PATH    = config.ANNOTATED_COHORT_PATH
OUTPUT_DIR        = config.EVAL_COHORT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_TARGET    = 100
RANDOM_SEED = 42

# Before running PSM, randomly subsample pool 
MAX_CONTROL_MULTIPLIER = 40       
MIN_CONTROL_POOL       = 2000     
MAX_TREATED_MULTIPLIER = 20      
MIN_TREATED_POOL       = 1000    

SENSITIVE_DIAGNOSES = config.SENSITIVE_DIAGNOSES

QC_FIELDS = [
    "extracted_name",
    "extracted_dob",
    "patient_gender",
    "raceeth",
    "medications_raw",
    "medications",
    "num_notes",
    "note_hpi",
    "note_start",
]

def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def coerce_bool(val) -> bool | None:
    if isinstance(val, bool):
        return val
    if isinstance(val, int):
        return bool(val)
    if isinstance(val, str):
        s = val.strip().lower()
        if s in ("true",  "1", "yes"): return True
        if s in ("false", "0", "no"):  return False
    return None


def parse_age(dob_str, reference_date=None) -> float | None:
    if not dob_str or str(dob_str).strip() in ("", "nan", "None", "null"):
        return None
    ref = reference_date or datetime(2024, 1, 1)
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m-%d-%Y",
                "%Y/%m/%d", "%d-%b-%Y", "%d/%m/%Y"):
        try:
            dob = datetime.strptime(str(dob_str).strip(), fmt)
            age = (ref - dob).days / 365.25
            return age if 0 < age < 130 else None
        except ValueError:
            continue
    return None


def binarise_sex(gender_str) -> int | None:
    g = str(gender_str).strip().lower()
    if g in ("m", "male"):   return 0
    if g in ("f", "female"): return 1
    return None


def safe_int(val) -> int | None:
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def is_missing(val) -> bool:
    if val is None:
        return True
    if isinstance(val, float) and np.isnan(val):
        return True
    return str(val).strip().lower() in ("", "nan", "none", "null", "n/a", "na")


def field_qc_flags(record: dict) -> dict:
    return {f"qc_{f}_ok": not is_missing(record.get(f)) for f in QC_FIELDS}


def deduplicate_groups(
    pt_ids:  list,
    nt_ids:  list,
    pnt_ids: list,
    nnt_ids: list,
    pos_train_pool:     list,
    neg_train_pool:     list,
    pos_nontrain_pool:  list,
    neg_nontrain_pool:  list,
    n_max: int,
    rng: np.random.Generator,
    dx: str,
) -> tuple[list, list, list, list]:
    """
    Guarantee that all four groups together contain exactly 4*n_max
    *unique* PatientUids, with no patient appearing in more than one group.

    Priority order for conflict resolution (preserves PSM quality):
      1. pt_ids  (pos_train)   
      2. pnt_ids (pos_nontrain) 
      3. nt_ids  (neg_train)    
      4. nnt_ids (neg_nontrain)
    """

    def _dedup_list(ids: list) -> list:
        seen, out = set(), []
        for uid in ids:
            if uid not in seen:
                seen.add(uid)
                out.append(uid)
        return out

    def _fill_to_n(ids: list, pool: list, claimed: set, label: str) -> list:
        if len(ids) >= n_max:
            return ids[:n_max]
        needed    = n_max - len(ids)
        available = [uid for uid in pool if uid not in claimed and uid not in ids]
        if len(available) < needed:
            print(f"    [WARN][dedup] {dx}/{label}: need {needed} top-up "
                  f"but only {len(available)} unclaimed available; "
                  f"final size will be {len(ids) + len(available)} < {n_max}.")
            needed = len(available)
        extra = rng.choice(available, needed, replace=False).tolist()
        return ids + extra

    pt_ids  = _dedup_list(pt_ids)
    pnt_ids = _dedup_list(pnt_ids)
    nt_ids  = _dedup_list(nt_ids)
    nnt_ids = _dedup_list(nnt_ids)

    claimed: set[str] = set()
    pt_ids  = pt_ids[:n_max]
    claimed.update(pt_ids)
    pnt_ids = [uid for uid in pnt_ids if uid not in claimed]
    claimed.update(pnt_ids)
    nt_ids  = [uid for uid in nt_ids  if uid not in claimed]
    claimed.update(nt_ids)
    nnt_ids = [uid for uid in nnt_ids if uid not in claimed]
    claimed.update(nnt_ids)

    n_conflicts = (
        (N_TARGET - len(pt_ids))
        + max(0, N_TARGET - len(pnt_ids))
        + max(0, N_TARGET - len(nt_ids))
        + max(0, N_TARGET - len(nnt_ids))
    )
    if n_conflicts > 0:
        print(f"    [dedup] {dx}: removed {n_conflicts} cross-group duplicate(s); "
              f"topping up affected groups …")

    pt_ids  = _fill_to_n(pt_ids,  pos_train_pool,    claimed, "pos_train")
    claimed.update(pt_ids)
    pnt_ids = _fill_to_n(pnt_ids, pos_nontrain_pool, claimed, "pos_nontrain")
    claimed.update(pnt_ids)
    nt_ids  = _fill_to_n(nt_ids,  neg_train_pool,    claimed, "neg_train")
    claimed.update(nt_ids)
    nnt_ids = _fill_to_n(nnt_ids, neg_nontrain_pool, claimed, "neg_nontrain")
    claimed.update(nnt_ids)

    all_ids  = pt_ids + pnt_ids + nt_ids + nnt_ids
    n_unique = len(set(all_ids))
    n_total  = len(all_ids)
    if n_unique < n_total:
        raise RuntimeError(
            f"[dedup] {dx}: STILL {n_total - n_unique} duplicate(s) after "
            f"deduplication — this is a bug, please report."
        )
    print(f"    [dedup] {dx}: {n_unique} unique patients across 4 groups "
          f"(pt={len(pt_ids)}, pnt={len(pnt_ids)}, "
          f"nt={len(nt_ids)}, nnt={len(nnt_ids)})")

    return pt_ids, nt_ids, pnt_ids, nnt_ids

train_records = load_jsonl(TRAIN_COHORT_PATH)
annot_records = load_jsonl(ANNOTATED_PATH)

train_uids = {str(r["PatientUid"]) for r in train_records}
print(f"  Training cohort  : {len(train_uids):,} unique patients")
print(f"  Annotated cohort : {len(annot_records):,} records")

rows = []
for r in annot_records:
    uid     = str(r.get("PatientUid", ""))
    age     = parse_age(r.get("extracted_dob", ""))
    sex     = binarise_sex(r.get("patient_gender", ""))
    n_notes = safe_int(r.get("num_notes"))

    row: dict = {
        "PatientUid": uid,
        "in_train"  : int(uid in train_uids),
        "age"       : age,
        "sex"       : sex,
        "num_notes" : n_notes,
        "medications_raw_ok": not is_missing(r.get("medications_raw")),
    }
    for dx in SENSITIVE_DIAGNOSES:
        row[f"{dx}_present"]                  = coerce_bool(r.get(f"{dx}_present"))
        row[f"{dx}_in_note_start"]            = coerce_bool(r.get(f"{dx}_in_note_start"))
        row[f"{dx}_in_hpi"]                   = coerce_bool(r.get(f"{dx}_in_hpi"))
        row[f"{dx}_medications_filtered_ok"]  = not is_missing(r.get(f"{dx}_medications_filtered"))
    rows.append(row)

base_df = pd.DataFrame(rows)
n_rows_pre_dedup = len(base_df)
n_uids_pre_dedup = base_df["PatientUid"].nunique()
print(f"  Rows (pre-dedup): {n_rows_pre_dedup:,}   unique PatientUids: {n_uids_pre_dedup:,}")

base_df = base_df.drop_duplicates("PatientUid", keep="first").reset_index(drop=True)
print(f"  Rows (post-dedup): {len(base_df):,}   "
      f"collapsed {n_rows_pre_dedup - len(base_df):,} duplicate records")
print(f"  in_train: {base_df['in_train'].sum():,}")
print(f"  age missing    : {base_df['age'].isna().sum():,}")
print(f"  sex missing    : {base_df['sex'].isna().sum():,}")
print(f"  num_notes miss : {base_df['num_notes'].isna().sum():,}\n")

COVARIATES = ["age", "sex", "num_notes"]

uid_to_raw = {str(r["PatientUid"]): r for r in annot_records}


def group_stats(df: pd.DataFrame, label: str) -> dict:
    age_vals = df["age"].dropna()
    sex_vals = df["sex"].dropna()
    nn_vals  = df["num_notes"].dropna()
    return {
        "group"         : label,
        "n"             : len(df),
        "age_mean"      : round(age_vals.mean(), 1)   if len(age_vals) else None,
        "age_median"    : round(age_vals.median(), 1) if len(age_vals) else None,
        "age_std"       : round(age_vals.std(),  1)   if len(age_vals) else None,
        "age_missing"   : int(df["age"].isna().sum()),
        "pct_female"    : round((sex_vals == 1).mean() * 100, 1) if len(sex_vals) else None,
        "sex_missing"   : int(df["sex"].isna().sum()),
        "notes_mean"    : round(nn_vals.mean(), 1)    if len(nn_vals) else None,
        "notes_median"  : round(nn_vals.median(), 1)  if len(nn_vals) else None,
        "notes_std"     : round(nn_vals.std(),  1)    if len(nn_vals) else None,
        "notes_missing" : int(df["num_notes"].isna().sum()),
        "in_train_n"    : int((df["in_train"] == 1).sum()),
        "in_train_pct"  : round((df["in_train"] == 1).mean() * 100, 1),
    }


def smd(a: pd.Series, b: pd.Series) -> float | None:
    a, b = a.dropna(), b.dropna()
    if len(a) < 2 or len(b) < 2:
        return None
    pooled_sd = np.sqrt((a.std() ** 2 + b.std() ** 2) / 2)
    if pooled_sd == 0:
        return 0.0
    return round((a.mean() - b.mean()) / pooled_sd, 3)


print("\n" + "=" * 88)
print("PER-DIAGNOSIS POOL STATISTICS")
print("=" * 88)

dx_stats_rows = []

for dx in SENSITIVE_DIAGNOSES:
    eval_ready_mask = (
        (base_df[f"{dx}_present"]                 == True)  &
        (base_df[f"{dx}_in_note_start"]           == False) &
        (base_df[f"{dx}_in_hpi"]                  == False) &
        (base_df[f"{dx}_medications_filtered_ok"] == True)
    )
    negative_mask = (
        (base_df[f"{dx}_present"]      == False) &
        (base_df["medications_raw_ok"] == True)
    )

    pos_train_df    = base_df[eval_ready_mask & (base_df["in_train"] == 1)]
    pos_nontrain_df = base_df[eval_ready_mask & (base_df["in_train"] == 0)]
    neg_train_df    = base_df[negative_mask   & (base_df["in_train"] == 1)]
    neg_nontrain_df = base_df[negative_mask   & (base_df["in_train"] == 0)]
    pos_all_df      = base_df[eval_ready_mask]
    neg_all_df      = base_df[negative_mask]

    print(f"\n{'─'*88}")
    print(f"  {dx.upper()}")
    print(f"{'─'*88}")
    print(f"  {'Group':<28} {'N':>6}  {'Age µ±σ':>12}  {'%F':>6}  {'Notes µ':>8}  {'InTrain':>8}")
    print(f"  {'-'*28} {'-'*6}  {'-'*12}  {'-'*6}  {'-'*8}  {'-'*8}")

    for label, sub_df in [
        ("pos_train",     pos_train_df),
        ("pos_non_train", pos_nontrain_df),
        ("neg_train",     neg_train_df),
        ("neg_non_train", neg_nontrain_df),
    ]:
        s = group_stats(sub_df, label)
        age_str   = f"{s['age_mean']}±{s['age_std']}"   if s["age_mean"]   is not None else "—"
        pct_f_str = f"{s['pct_female']}%"                if s["pct_female"] is not None else "—"
        notes_str = f"{s['notes_mean']}"                 if s["notes_mean"] is not None else "—"
        train_str = f"{s['in_train_n']} ({s['in_train_pct']}%)"
        print(f"  {label:<28} {s['n']:>6,}  {age_str:>12}  {pct_f_str:>6}  {notes_str:>8}  {train_str:>8}")
        dx_stats_rows.append({"diagnosis": dx, **s})

    print(f"\n  SMD pos vs neg (pre-match):")
    for cov in COVARIATES:
        s_val = smd(pos_all_df[cov], neg_all_df[cov])
        if s_val is not None:
            flag = "✓" if abs(s_val) <= 0.1 else "△" if abs(s_val) <= 0.2 else "✗"
            print(f"    {flag} {cov:<14}: {s_val:+.3f}")
        else:
            print(f"    ? {cov:<14}: insufficient data")

    print(f"  SMD pos_train vs pos_non_train (pre-match):")
    for cov in COVARIATES:
        s_val = smd(pos_train_df[cov], pos_nontrain_df[cov])
        if s_val is not None:
            flag = "✓" if abs(s_val) <= 0.1 else "△" if abs(s_val) <= 0.2 else "✗"
            print(f"    {flag} {cov:<14}: {s_val:+.3f}")
        else:
            print(f"    ? {cov:<14}: insufficient data")

    feasible = (len(pos_train_df) >= N_TARGET and len(pos_nontrain_df) >= N_TARGET)
    print(f"\n  Pool feasibility (need {N_TARGET} each):")
    print(f"    pos_train={len(pos_train_df):,}  pos_non_train={len(pos_nontrain_df):,}  "
          f"neg_train={len(neg_train_df):,}  neg_non_train={len(neg_nontrain_df):,}  "
          f"→ {'✓ FEASIBLE' if feasible else '✗ WILL SKIP'}")

print(f"\n{'='*88}\n")

dx_stats_df   = pd.DataFrame(dx_stats_rows)
dx_stats_path = OUTPUT_DIR / "dx_pool_stats.csv"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
dx_stats_df.to_csv(dx_stats_path, index=False)
print(f"Per-diagnosis pool stats → {dx_stats_path}\n")

def _topup(matched_ids: list, full_pool: list, n_max: int,
           rng: np.random.Generator, label: str) -> list:
    """Top up matched_ids to n_max using unmatched members of the FULL pool."""
    if len(matched_ids) >= n_max:
        return matched_ids[:n_max]
    needed      = n_max - len(matched_ids)
    matched_set = set(matched_ids)
    unmatched   = [uid for uid in full_pool if uid not in matched_set]
    if len(unmatched) == 0:
        print(f"    [WARN] No unmatched {label} left to top up; "
              f"returning {len(matched_ids)} (< {n_max}).")
        return matched_ids
    n_sample = min(needed, len(unmatched))
    if n_sample < needed:
        print(f"    [WARN] {label} pool exhausted — could only reach "
              f"{len(matched_ids) + n_sample} / {n_max}.")
    extra = rng.choice(unmatched, n_sample, replace=False).tolist()
    return matched_ids + extra


def _subsample_for_psm(df: pd.DataFrame, treatment_col: str, id_col: str,
                       rng: np.random.Generator,
                       n_max: int = N_TARGET) -> pd.DataFrame:
    treated_cap = max(MIN_TREATED_POOL, MAX_TREATED_MULTIPLIER * n_max)
    control_cap = max(MIN_CONTROL_POOL, MAX_CONTROL_MULTIPLIER * n_max)

    treated_df = df[df[treatment_col] == 1]
    control_df = df[df[treatment_col] == 0]

    n_t_orig = len(treated_df)
    n_c_orig = len(control_df)

    if n_t_orig > treated_cap:
        keep_t = rng.choice(treated_df[id_col].values, treated_cap, replace=False)
        treated_df = treated_df[treated_df[id_col].isin(keep_t)]
        print(f"    [subsample] treated {n_t_orig:,} → {len(treated_df):,} "
              f"(cap={treated_cap:,})")
    if n_c_orig > control_cap:
        keep_c = rng.choice(control_df[id_col].values, control_cap, replace=False)
        control_df = control_df[control_df[id_col].isin(keep_c)]
        print(f"    [subsample] control {n_c_orig:,} → {len(control_df):,} "
              f"(cap={control_cap:,})")

    if n_t_orig <= treated_cap and n_c_orig <= control_cap:
        print(f"    [subsample] no cap needed (treated={n_t_orig:,}, control={n_c_orig:,})")

    return pd.concat([treated_df, control_df], ignore_index=True)


def run_psm(df: pd.DataFrame, treatment_col: str, covariates: list[str],
            id_col: str = "PatientUid", n_max: int = N_TARGET,
            seed: int = RANDOM_SEED) -> tuple[list, list]:
    rng = np.random.default_rng(seed)

    # Preserve FULL pools for _topup (matches v6 semantics)
    full_treated_pool = df[df[treatment_col] == 1][id_col].tolist()
    full_control_pool = df[df[treatment_col] == 0][id_col].tolist()

    for side, pool in (("treated", full_treated_pool), ("control", full_control_pool)):
        if len(pool) < n_max:
            print(f"    [WARN] {side} pool only has {len(pool)} patients "
                  f"(target={n_max}); will top up with all available.")

    if len(full_treated_pool) < 5 or len(full_control_pool) < 5:
        print(f"    [WARN] Pool too small for PSM — using random sampling.")
        t = rng.choice(full_treated_pool, min(n_max, len(full_treated_pool)),
                       replace=False).tolist()
        c = rng.choice(full_control_pool, min(n_max, len(full_control_pool)),
                       replace=False).tolist()
        return t, c

    df_psm = _subsample_for_psm(df, treatment_col, id_col, rng, n_max=n_max)

    work = (df_psm[[id_col, treatment_col] + covariates]
            .dropna()
            .reset_index(drop=True))
    work["_idx"] = work.index.astype(int)

    n_treated = int((work[treatment_col] == 1).sum())
    n_control = int((work[treatment_col] == 0).sum())
    print(f"    PSM: n_treated={n_treated}, n_control={n_control} (post-subsample)")

    try:
        psm = PsmPy(work, treatment=treatment_col, indx="_idx", exclude=[id_col])
        psm.logistic_ps(balance=False)
        psm.kdtree_matched(matcher="propensity_logit", replacement=False,
                           caliper=None, drop_unmatched=True)

        matched = psm.matched_ids.dropna()

        print(f"    matched_ids columns: {matched.columns.tolist()}")
        idx_to_uid = work.set_index("_idx")[id_col].to_dict()
        idx_to_trt = work.set_index("_idx")[treatment_col].to_dict()

        sample_minor = int(matched["_idx"].iloc[0])
        minor_is_treated = (idx_to_trt.get(sample_minor) == 1)
        if minor_is_treated:
            treated_col_name, control_col_name = "_idx", "matched_ID"
        else:
            treated_col_name, control_col_name = "matched_ID", "_idx"
        print(f"    minor=treated: {minor_is_treated} → "
              f"treated='{treated_col_name}', control='{control_col_name}'")

        t_ids, c_ids = [], []
        for _, row in matched.iterrows():
            t_uid = idx_to_uid.get(int(row[treated_col_name]))
            c_uid = idx_to_uid.get(int(row[control_col_name]))
            if t_uid and c_uid:
                t_ids.append(t_uid)
                c_ids.append(c_uid)

        try:
            from psmpy.functions import cohenD
            print(f"    SMD after matching:")
            for cov in covariates:
                smd_val = cohenD(psm.df_matched, treatment_col, cov)
                flag = "✓" if abs(smd_val) <= 0.1 else "△" if abs(smd_val) <= 0.2 else "✗"
                qual = "balanced" if abs(smd_val) <= 0.1 else \
                       "acceptable" if abs(smd_val) <= 0.2 else "IMBALANCED"
                print(f"      {flag} {cov:<12}: SMD={smd_val:+.3f}  ({qual})")
        except Exception as smd_exc:
            print(f"    [SMD unavailable: {smd_exc}]")

        if len(t_ids) > n_max:
            sel   = rng.choice(len(t_ids), n_max, replace=False)
            t_ids = [t_ids[i] for i in sel]
            c_ids = [c_ids[i] for i in sel]

        t_ids = _topup(t_ids, full_treated_pool, n_max, rng, "treated")
        c_ids = _topup(c_ids, full_control_pool, n_max, rng, "control")

        return t_ids, c_ids

    except Exception as exc:
        import traceback
        print(f"    [WARN] PSMpy error: {exc!r}")
        traceback.print_exc()
        print(f"    Falling back to random sampling.")
        t = rng.choice(full_treated_pool, min(n_max, len(full_treated_pool)),
                       replace=False).tolist()
        c = rng.choice(full_control_pool, min(n_max, len(full_control_pool)),
                       replace=False).tolist()
        return t, c


def build_output_records(uid_list: list, split: str, label: str, dx: str) -> list:
    out = []
    for uid in uid_list:
        raw = uid_to_raw.get(uid, {})
        rec = dict(raw)
        rec["eval_split"] = split
        rec["eval_label"] = label
        rec["diagnosis"]  = dx
        rec.update(field_qc_flags(raw))
        rec[f"qc_{dx}_medications_filtered_ok"] = not is_missing(raw.get(f"{dx}_medications_filtered"))
        out.append(rec)
    return out


summary_rows = []

for dx in SENSITIVE_DIAGNOSES:
    print(f"=== {dx} ===")

    eval_ready_mask = (
        (base_df[f"{dx}_present"]                 == True)  &
        (base_df[f"{dx}_in_note_start"]           == False) &
        (base_df[f"{dx}_in_hpi"]                  == False) &
        (base_df[f"{dx}_medications_filtered_ok"] == True)
    )
    negative_mask = (
        (base_df[f"{dx}_present"]     == False) &
        (base_df["medications_raw_ok"] == True)
    )

    pos_train    = base_df[eval_ready_mask & (base_df["in_train"] == 1)]
    pos_nontrain = base_df[eval_ready_mask & (base_df["in_train"] == 0)]
    neg_train    = base_df[negative_mask   & (base_df["in_train"] == 1)]
    neg_nontrain = base_df[negative_mask   & (base_df["in_train"] == 0)]

    print(f"  eval-ready pos : train={len(pos_train)}, non-train={len(pos_nontrain)}")
    print(f"  negatives      : train={len(neg_train)}, non-train={len(neg_nontrain)}")

    if len(pos_train) < N_TARGET or len(pos_nontrain) < N_TARGET:
        short = []
        if len(pos_train)    < N_TARGET: short.append(f"pos_train={len(pos_train)}")
        if len(pos_nontrain) < N_TARGET: short.append(f"pos_nontrain={len(pos_nontrain)}")
        print(f"  [WARN] Pool(s) below N_TARGET={N_TARGET}: {', '.join(short)}. "
              f"Skipping diagnosis.\n")
        summary_rows.append({
            "diagnosis": dx,
            "pos_train_avail": len(pos_train),
            "pos_nontrain_avail": len(pos_nontrain),
            "final_pos_train": 0, "final_neg_train": 0,
            "final_pos_nontrain": 0, "final_neg_nontrain": 0,
            "total_records": 0,
        })
        continue

    print("  Step 3: PSM pos-train vs pos-non-train …")
    pos_combined = pd.concat([
        pos_train.assign(treatment=1),
        pos_nontrain.assign(treatment=0),
    ], ignore_index=True)
    pt_ids, pnt_ids = run_psm(pos_combined, "treatment", COVARIATES)
    print(f"    → pos train={len(pt_ids)}, pos non-train={len(pnt_ids)}")

    print("  Step 4a: PSM pos-train vs neg-train …")
    train_combined = pd.concat([
        base_df[base_df["PatientUid"].isin(pt_ids)].assign(treatment=1),
        neg_train.assign(treatment=0),
    ], ignore_index=True)
    _, nt_ids = run_psm(train_combined, "treatment", COVARIATES)
    print(f"    → neg train={len(nt_ids)}")

    print("  Step 4b: PSM pos-non-train vs neg-non-train …")
    nontrain_combined = pd.concat([
        base_df[base_df["PatientUid"].isin(pnt_ids)].assign(treatment=1),
        neg_nontrain.assign(treatment=0),
    ], ignore_index=True)
    _, nnt_ids = run_psm(nontrain_combined, "treatment", COVARIATES)
    print(f"    → neg non-train={len(nnt_ids)}")

    rng_dedup = np.random.default_rng(RANDOM_SEED)
    pt_ids, nt_ids, pnt_ids, nnt_ids = deduplicate_groups(
        pt_ids  = pt_ids,
        nt_ids  = nt_ids,
        pnt_ids = pnt_ids,
        nnt_ids = nnt_ids,
        pos_train_pool    = pos_train["PatientUid"].tolist(),
        neg_train_pool    = neg_train["PatientUid"].tolist(),
        pos_nontrain_pool = pos_nontrain["PatientUid"].tolist(),
        neg_nontrain_pool = neg_nontrain["PatientUid"].tolist(),
        n_max = N_TARGET,
        rng   = rng_dedup,
        dx    = dx,
    )

    all_ids = pt_ids + pnt_ids + nt_ids + nnt_ids
    assert len(all_ids) == len(set(all_ids)), \
        f"{dx}: duplicate PatientUids remain after deduplication — bug!"
    assert len(pt_ids)  == N_TARGET, f"{dx} pos_train: {len(pt_ids)} ≠ {N_TARGET}"
    assert len(pnt_ids) == N_TARGET, f"{dx} pos_nontrain: {len(pnt_ids)} ≠ {N_TARGET}"
    assert len(nt_ids)  == N_TARGET, f"{dx} neg_train: {len(nt_ids)} ≠ {N_TARGET}"
    assert len(nnt_ids) == N_TARGET, f"{dx} neg_nontrain: {len(nnt_ids)} ≠ {N_TARGET}"
    print(f"  ✓ Uniqueness assertion passed: {len(set(all_ids))} unique patients")


    def _sub(ids):
        return base_df[base_df["PatientUid"].isin(ids)]
    print(f"  Post-match covariate summary:")
    print(f"    {'Group':<20} {'N':>5}  {'Age µ±σ':>12}  {'%F':>6}  {'Notes µ':>8}")
    for label, ids in [("pos_train", pt_ids), ("pos_non_train", pnt_ids),
                       ("neg_train", nt_ids), ("neg_non_train", nnt_ids)]:
        s = group_stats(_sub(ids), label)
        age_str   = f"{s['age_mean']}±{s['age_std']}"  if s["age_mean"]   is not None else "—"
        pct_f_str = f"{s['pct_female']}%"               if s["pct_female"] is not None else "—"
        notes_str = f"{s['notes_mean']}"                if s["notes_mean"] is not None else "—"
        print(f"    {label:<20} {s['n']:>5}  {age_str:>12}  {pct_f_str:>6}  {notes_str:>8}")
    print(f"  Post-match SMD (pos_train vs neg_train):")
    for cov in COVARIATES:
        s_val = smd(_sub(pt_ids)[cov], _sub(nt_ids)[cov])
        flag  = "✓" if s_val is not None and abs(s_val) <= 0.1 else \
                "△" if s_val is not None and abs(s_val) <= 0.2 else "✗"
        print(f"    {flag} {cov:<14}: {s_val:+.3f}" if s_val is not None else f"    ? {cov}: —")
    print(f"  Post-match SMD (pos_non_train vs neg_non_train):")
    for cov in COVARIATES:
        s_val = smd(_sub(pnt_ids)[cov], _sub(nnt_ids)[cov])
        flag  = "✓" if s_val is not None and abs(s_val) <= 0.1 else \
                "△" if s_val is not None and abs(s_val) <= 0.2 else "✗"
        print(f"    {flag} {cov:<14}: {s_val:+.3f}" if s_val is not None else f"    ? {cov}: —")

    output_records = (
        build_output_records(pt_ids,  "train",     "positive", dx) +
        build_output_records(nt_ids,  "train",     "negative", dx) +
        build_output_records(pnt_ids, "non_train", "positive", dx) +
        build_output_records(nnt_ids, "non_train", "negative", dx)
    )

    out_path = OUTPUT_DIR / f"{dx}_eval_cohort.jsonl"
    with open(out_path, "w") as f:
        for rec in output_records:
            f.write(json.dumps(rec) + "\n")

    summary_rows.append({
        "diagnosis"          : dx,
        "pos_train_avail"    : len(pos_train),
        "pos_nontrain_avail" : len(pos_nontrain),
        "final_pos_train"    : len(pt_ids),
        "final_neg_train"    : len(nt_ids),
        "final_pos_nontrain" : len(pnt_ids),
        "final_neg_nontrain" : len(nnt_ids),
        "total_records"      : len(output_records),
        "output_file"        : str(out_path),
    })
    print(f"  → {len(output_records)} records saved to {out_path}\n")

summary_df   = pd.DataFrame(summary_rows)
summary_path = OUTPUT_DIR / "matching_summary.csv"
summary_df.to_csv(summary_path, index=False)

print("\n" + "=" * 88)
print("FINAL SUMMARY — COHORT COUNTS")
print("=" * 88)
print(f"\n  {'Diagnosis':<35} {'PT':>5} {'PNT':>5} {'NT':>5} {'NNT':>5} {'Total':>7} {'Status':>8}")
print(f"  {'-'*35} {'-'*5} {'-'*5} {'-'*5} {'-'*5} {'-'*7} {'-'*8}")
for _, row in summary_df.iterrows():
    status = "✓" if row["total_records"] == 4 * N_TARGET else \
             "⚠ SKIP" if row["total_records"] == 0 else "⚠ SHORT"
    print(f"  {row['diagnosis']:<35} "
          f"{int(row['final_pos_train']):>5} "
          f"{int(row['final_pos_nontrain']):>5} "
          f"{int(row['final_neg_train']):>5} "
          f"{int(row['final_neg_nontrain']):>5} "
          f"{int(row['total_records']):>7} "
          f"{status:>8}")

completed = summary_df[summary_df["total_records"] == 4 * N_TARGET]
skipped   = summary_df[summary_df["total_records"] == 0]
print(f"\n  Completed: {len(completed)} diagnoses  |  Skipped: {len(skipped)} diagnoses")
print(f"  Total patients written: {summary_df['total_records'].sum():,}")
print(f"\n  PT=pos_train  PNT=pos_non_train  NT=neg_train  NNT=neg_non_train")
print(f"\nSummary CSV  → {summary_path}")
print(f"Pool stats   → {dx_stats_path}")
